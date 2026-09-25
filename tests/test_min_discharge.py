"""路口放行能力测算核心算法测试。

确定性用例覆盖：最小放行量的二分边界、deadline 含端点、deadline 迫使更大放行量、
红灯不放行导致无解、期初队列先于批次、同秒批次按输入顺序先入队再放行、
跨零点绿灯、常绿、offset 相位、半开范围终点、空请求、大车辆数不逐辆展开。
另含 2000 组随机参数下与逐辆展开暴力参考实现（含逐 d 判定与二分）的逐项对照。
"""

import random
from collections import deque

from app.greenwave import compute_min_discharge


def brute_force(
    cycle,
    offset,
    green_start,
    green_end,
    forecast_from,
    forecast_to,
    initial_queue,
    arrivals,
):
    """逐辆展开 + 逐 d 二分的参考实现，返回 (min_discharge, pass_times) 或 None。"""

    def green(now):
        if green_start == green_end:
            return True
        phase = (now + offset) % cycle
        if green_start < green_end:
            return green_start <= phase < green_end
        return phase >= green_start or phase < green_end

    total = initial_queue + sum(count for _t, count, _dl in arrivals)
    by_second = {}
    for index, (moment, _count, _dl) in enumerate(arrivals):
        by_second.setdefault(moment, []).append(index)

    def simulate(discharge):
        queue = deque([-1] * initial_queue)
        remaining = [count for _t, count, _dl in arrivals]
        pass_times = [None] * len(arrivals)
        feasible = True
        for now in range(forecast_from, forecast_to):
            for index in by_second.get(now, []):
                queue.extend([index] * arrivals[index][1])
            if queue and green(now):
                for _ in range(min(discharge, len(queue))):
                    batch = queue.popleft()
                    if batch >= 0:
                        remaining[batch] -= 1
                        if remaining[batch] == 0:
                            pass_times[batch] = now
                            if now > arrivals[batch][2]:
                                feasible = False
        if queue:
            feasible = False
        return feasible, pass_times

    if total == 0:
        return 1, []
    if not simulate(total)[0]:
        return None
    lo, hi = 1, total
    while lo < hi:
        mid = (lo + hi) // 2
        if simulate(mid)[0]:
            hi = mid
        else:
            lo = mid + 1
    _ok, pass_times = simulate(lo)
    return lo, pass_times


def solve(**kw):
    """以常绿默认值调用 compute_min_discharge，关键字覆盖。"""
    params = {
        "cycle": 60,
        "offset": 0,
        "green_start": 0,
        "green_end": 0,  # 常绿
        "forecast_from": 0,
        "forecast_to": 10,
        "initial_queue": 0,
        "arrivals": [],
    }
    params.update(kw)
    return compute_min_discharge(**params)


# ---------- 确定性用例 ----------


def test_basic_minimum_is_one():
    # 常绿每秒 1 辆即可：期初 2 辆先走，t=1 到 3 辆在 t=4 走完，deadline 宽松
    assert solve(initial_queue=2, arrivals=[(1, 3, 9)]) == (1, [4])


def test_deadline_forces_larger_discharge():
    # 同上但 deadline=3：d=1 时 t=4 走完已晚，d=2 时期初在 t=0..1、
    # 该批在 t=2 全部走完（t=1 到达后先放 1 辆，t=2 放剩余 2 辆）
    assert solve(initial_queue=2, arrivals=[(1, 3, 3)]) == (2, [2])
    # deadline 恰为 t=2（含端点）时 d=2 仍可行
    assert solve(initial_queue=2, arrivals=[(1, 3, 2)]) == (2, [2])
    # 期初 2 辆在 t=0 已全部放完，故 t=1 当秒队列只有该批 3 辆：
    # deadline=1 只需 d=3
    assert solve(initial_queue=2, arrivals=[(1, 3, 1)]) == (3, [1])


def test_arrival_second_deadline_with_initial_queue():
    # 期初 1 辆挡在队首，批次 t=0 到 1 辆且 deadline=0：当秒须同时放过
    # 期初 1 辆与该批 1 辆，d=2；d=1 时该批要等到 t=1，晚于 deadline
    assert solve(initial_queue=1, arrivals=[(0, 1, 0)]) == (2, [0])


def test_red_light_no_green_before_deadline_is_null():
    # 绿灯 [0,10)：t=20 到达，deadline=25，范围内 [20,30) 全程红灯
    result = solve(
        green_start=0,
        green_end=10,
        forecast_to=30,
        arrivals=[(20, 1, 25)],
    )
    assert result is None


def test_all_red_range_with_initial_queue_is_null():
    # 绿灯 [0,10)，范围 [20,40) 全程红灯：期初队列无法清空
    result = solve(
        green_start=0,
        green_end=10,
        forecast_from=20,
        forecast_to=40,
        initial_queue=1,
    )
    assert result is None


def test_deadline_boundary_at_last_green_second():
    # 绿灯 [0,10)：t=8 到 3 辆，d=1 时 t=8,9 只走 2 辆，第三辆要到 t=60，
    # deadline=60 可行（含端点）且 d=1；范围须覆盖 t=60
    result = solve(
        green_start=0,
        green_end=10,
        forecast_from=0,
        forecast_to=61,
        arrivals=[(8, 3, 60)],
    )
    assert result == (1, [60])
    # deadline=9 时第三辆来不及，d=2 可在 t=9 放完（t=8 放 1、t=9 放 2）
    result = solve(
        green_start=0,
        green_end=10,
        forecast_from=0,
        forecast_to=61,
        arrivals=[(8, 3, 9)],
    )
    assert result == (2, [9])


def test_half_open_range_end_not_discharged():
    # 范围终点不含：绿灯恰在 forecast_to 才到来时无解
    result = solve(
        green_start=0,
        green_end=10,
        forecast_from=20,
        forecast_to=60,
        arrivals=[(20, 1, 59)],
    )
    assert result is None


def test_same_second_batches_input_order_matters_for_deadline():
    # t=0 两批：批 A 2 辆 deadline=0，批 B 1 辆 deadline=2
    result = solve(arrivals=[(0, 2, 0), (0, 1, 2)])
    assert result == (2, [0, 1])  # A 当秒全走（d=2），B 在 t=1 走完
    # 顺序对调：B(1 辆, dl=2) 排在 A(2 辆, dl=0) 之前，A 当秒全走须连同
    # B 一起放，d=3；两批都在 t=0 走完
    result = solve(arrivals=[(0, 1, 2), (0, 2, 0)])
    assert result == (3, [0, 0])


def test_wrapping_green_min_discharge():
    # 跨零点绿灯 [50,60)∪[0,10)：t=55 到 3 辆，deadline=56，每秒放行 2
    # 即可（t=55 放 2、t=56 放 1）
    result = solve(
        green_start=50,
        green_end=10,
        forecast_from=50,
        forecast_to=60,
        arrivals=[(55, 3, 56)],
    )
    assert result == (2, [56])
    # deadline=55：当秒 3 辆须全走，d=3
    result = solve(
        green_start=50,
        green_end=10,
        forecast_from=50,
        forecast_to=60,
        arrivals=[(55, 3, 55)],
    )
    assert result == (3, [55])


def test_wrapping_green_uses_second_segment():
    # 跨零点绿灯 [50,60)∪[0,10)，范围 [0,30)：t=0..9 为绿，期初 10 辆
    # d=1 在 t=9 恰放完
    result = solve(
        green_start=50,
        green_end=10,
        forecast_from=0,
        forecast_to=30,
        initial_queue=10,
    )
    assert result == (1, [])


def test_offset_shifts_green_window():
    # offset=30、绿灯 [0,10)：绝对绿灯段为 [30,40)
    # t=30 到 2 辆，deadline=30：d=2 当秒全走
    result = solve(
        offset=30,
        green_start=0,
        green_end=10,
        forecast_from=30,
        forecast_to=45,
        arrivals=[(30, 2, 30)],
    )
    assert result == (2, [30])
    # deadline=30 但期初已有 1 辆：当秒须放 3 辆
    result = solve(
        offset=30,
        green_start=0,
        green_end=10,
        forecast_from=30,
        forecast_to=45,
        initial_queue=1,
        arrivals=[(30, 2, 30)],
    )
    assert result == (3, [30])


def test_initial_queue_only_requires_one_under_always_green():
    assert solve(forecast_to=5, initial_queue=5) == (1, [])


def test_empty_request_returns_discharge_one():
    assert solve() == (1, [])


def test_large_counts_without_per_vehicle_expansion():
    # 常绿、10^12 辆、范围 1001 秒：最小放行量 ceil(10^12 / 1001)
    result = solve(
        forecast_to=1001,
        arrivals=[(0, 10**12, 1000)],
    )
    assert result == (999_001_000, [1000])
    # deadline=999：只有 t=0..999 共 1000 秒可用，ceil(10^12 / 1000) = 10^9
    result = solve(
        forecast_to=1001,
        arrivals=[(0, 10**12, 999)],
    )
    assert result == (10**9, [999])


def test_multiple_batches_pass_times_in_input_order():
    # 常绿 d=1：t=0 批 1 辆 t=0 走；t=2 批 2 辆 t=2、t=3 走完
    result = solve(
        forecast_to=10,
        arrivals=[(0, 1, 9), (2, 2, 9)],
    )
    assert result == (1, [0, 3])


def test_later_arrival_with_tight_deadline_behind_earlier_queue():
    # t=0 到 3 辆（dl=9 宽松），t=1 到 1 辆（dl=1）。t=0 已先放 d 辆：
    # d=2 时前批在 t=0 放 2、t=1 当秒剩 1 辆在前，后批要在 t=1 通过须连同
    # 前批剩余 1 辆一起放，需 d=2，前批 t=1 走完、后批同在 t=1 走完
    result = solve(arrivals=[(0, 3, 9), (1, 1, 1)])
    assert result == (2, [1, 1])
    # 后批 deadline=0：t=0 当秒后批尚未到达，t=1 才入队，最早只能在 t=1
    # 通过——即使无限放行也无解
    result = solve(arrivals=[(0, 3, 9), (1, 1, 0)])
    assert result is None
    # 前批把放行能力全部占满的场景：t=0 批 3 辆 dl=0 须当秒全走（d>=3），
    # 同秒后批 1 辆 dl=0：FIFO 下当秒须放 4 辆
    result = solve(arrivals=[(0, 3, 0), (0, 1, 0)])
    assert result == (4, [0, 0])


# ---------- 随机对照：逐辆展开暴力实现 ----------


def _random_signal(rng):
    cycle = rng.choice([1, 2, 3, 7, 12, 60, 90])
    offset = rng.randrange(cycle)
    kind = rng.randrange(3)
    if kind == 0:  # 常绿
        green_start = green_end = rng.randrange(cycle)
    elif kind == 1:  # 普通绿灯
        green_start = rng.randrange(cycle)
        green_end = rng.randrange(cycle)
        if green_start == green_end:
            green_end = (green_end + 1) % cycle
        if green_start > green_end:
            green_start, green_end = green_end, green_start
    else:  # 跨零点绿灯
        green_start = rng.randrange(cycle)
        green_end = rng.randrange(cycle)
        if green_start == green_end:
            green_start = (green_start + 1) % cycle
        if green_start < green_end:
            green_start, green_end = green_end, green_start
    return cycle, offset, green_start, green_end


def test_random_against_per_vehicle_brute_force():
    rng = random.Random(20260925)
    for _ in range(2000):
        cycle, offset, green_start, green_end = _random_signal(rng)
        forecast_from = rng.randrange(0, 40)
        forecast_to = forecast_from + rng.randrange(1, 90)
        initial = rng.randrange(0, 6)
        raw = [
            (rng.randrange(forecast_from, forecast_to), rng.randrange(1, 5))
            for _ in range(rng.randrange(0, 8))
        ]
        arrivals = [
            (moment, count, rng.randrange(moment, forecast_to))
            for moment, count in raw
        ]
        expected = brute_force(
            cycle, offset, green_start, green_end,
            forecast_from, forecast_to, initial, arrivals,
        )
        actual = compute_min_discharge(
            cycle=cycle,
            offset=offset,
            green_start=green_start,
            green_end=green_end,
            forecast_from=forecast_from,
            forecast_to=forecast_to,
            initial_queue=initial,
            arrivals=arrivals,
        )
        assert actual == expected

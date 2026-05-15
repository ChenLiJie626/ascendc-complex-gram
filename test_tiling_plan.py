#!/usr/bin/env python3
"""Tests for ComplexGram tiling/synchronization planning.

This test does not compile AscendC. It verifies the host-side tiling contract used
by the kernel template:
- 20 cube cores, 40 vector cores.
- vector pair mapping: vector 2*c and 2*c+1 consume cube c's tile tasks.
- group-tile task space covers every B[g, row, col] tile exactly once.
"""
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Tiling:
    n: int
    k: int
    bm: int
    bn: int
    cube_blocks: int
    vector_blocks: int
    tile_m_num: int
    tile_n_num: int
    total_tasks: int
    workspace_bytes: int
    ping_pong: int = 2
    flag_slots_per_cube: int = 4


def make_tiling(n: int, bm: int = 32, bn: int = 32,
                cube_blocks: int = 20, vector_blocks: int = 40) -> Tiling:
    k = 8 * n
    tile_m_num = ceil(k / bm)
    tile_n_num = ceil(k / bn)
    total_tasks = 17 * tile_m_num * tile_n_num
    # Four float temporary planes for one group-tile task and 16 slices.
    workspace_bytes = total_tasks * 16 * 4 * bm * bn * 4
    return Tiling(n, k, bm, bn, cube_blocks, vector_blocks,
                  tile_m_num, tile_n_num, total_tasks, workspace_bytes)


def decode_task(task_id: int, t: Tiling):
    tile_per_group = t.tile_m_num * t.tile_n_num
    g = task_id // tile_per_group
    r = task_id % tile_per_group
    tm = r // t.tile_n_num
    tn = r % t.tile_n_num
    return g, tm, tn


def cube_tasks(cube_id: int, t: Tiling):
    return list(range(cube_id, t.total_tasks, t.cube_blocks))


def vector_tasks(vector_id: int, t: Tiling):
    pair_cube = vector_id // 2
    lane = vector_id % 2
    return pair_cube, lane, cube_tasks(pair_cube, t)


def flag_ids(cube_id: int, lane: int, buf: int, t: Tiling):
    """Two ready flags per ping-pong buffer and two done flags per buffer.

    ready flag: cube -> vector lane.
    done flag : vector lane -> cube, allowing the cube to reuse the buffer only
                after both lanes have consumed it.
    """
    assert 0 <= lane < 2
    assert 0 <= buf < t.ping_pong
    base = cube_id * t.flag_slots_per_cube * t.ping_pong + buf * t.flag_slots_per_cube
    ready = base + lane
    done = base + 2 + lane
    return ready, done


def scheduled_buffer(task_index_within_cube: int, t: Tiling):
    return task_index_within_cube % t.ping_pong


def test_cube_vector_pair_mapping():
    t = make_tiling(n=16)
    assert t.cube_blocks == 20
    assert t.vector_blocks == 40
    for c in range(20):
        assert vector_tasks(2 * c, t)[0:2] == (c, 0)
        assert vector_tasks(2 * c + 1, t)[0:2] == (c, 1)
        assert vector_tasks(2 * c, t)[2] == cube_tasks(c, t)
        assert vector_tasks(2 * c + 1, t)[2] == cube_tasks(c, t)


def test_group_tile_task_space_covers_all_tiles_once():
    t = make_tiling(n=16, bm=32, bn=32)
    seen = set()
    for c in range(20):
        for task in cube_tasks(c, t):
            seen.add(decode_task(task, t))
    expected = {
        (g, tm, tn)
        for g in range(17)
        for tm in range(t.tile_m_num)
        for tn in range(t.tile_n_num)
    }
    assert seen == expected
    assert len(seen) == t.total_tasks


def test_lane_splits_tile_elements_without_overlap():
    t = make_tiling(n=16, bm=32, bn=32)
    elems = t.bm * t.bn
    lane0 = set(range(0, (elems + 1) // 2))
    lane1 = set(range((elems + 1) // 2, elems))
    assert lane0.isdisjoint(lane1)
    assert lane0 | lane1 == set(range(elems))


def test_single_kernel_flag_ids_are_unique_per_cube_lane_and_buffer():
    t = make_tiling(n=16)
    ids = set()
    for c in range(t.cube_blocks):
        for buf in range(t.ping_pong):
            for lane in range(2):
                ready, done = flag_ids(c, lane, buf, t)
                assert ready not in ids
                ids.add(ready)
                assert done not in ids
                ids.add(done)
    assert len(ids) == t.cube_blocks * t.ping_pong * t.flag_slots_per_cube


def test_single_kernel_ping_pong_reuse_waits_for_both_vector_lanes():
    t = make_tiling(n=16)
    for c in range(t.cube_blocks):
        tasks = cube_tasks(c, t)
        # Before reusing a ping-pong buffer for task j, cube must wait for the
        # done flags posted by both paired vector lanes for task j-2.
        for local_idx, task in enumerate(tasks):
            buf = scheduled_buffer(local_idx, t)
            if local_idx >= t.ping_pong:
                prev = tasks[local_idx - t.ping_pong]
                assert scheduled_buffer(local_idx - t.ping_pong, t) == buf
                assert prev != task
                waits = [flag_ids(c, lane, buf, t)[1] for lane in range(2)]
                assert len(set(waits)) == 2


if __name__ == "__main__":
    test_cube_vector_pair_mapping()
    test_group_tile_task_space_covers_all_tiles_once()
    test_lane_splits_tile_elements_without_overlap()
    test_single_kernel_flag_ids_are_unique_per_cube_lane_and_buffer()
    test_single_kernel_ping_pong_reuse_waits_for_both_vector_lanes()
    print("tiling plan tests passed")

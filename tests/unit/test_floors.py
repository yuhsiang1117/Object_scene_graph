"""Online floor estimation (osg.mapping.floors)."""
import numpy as np
import pytest

from osg.mapping.floors import FloorEstimator, height_histogram_peaks

CAM_H = 0.88


def est(**kw):
    return FloorEstimator(camera_height=CAM_H, **kw)


def drive(e, floor_ys, steps_each=10, start_step=0):
    """Feed `steps_each` frames at each floor height in turn."""
    step = start_step
    out = []
    for y in floor_ys:
        for _ in range(steps_each):
            out.append(e.update(y + CAM_H, step))
            step += 1
    return out


def ramp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# ------------------------------------------------------- single-floor no-op


def test_bootstrap_matches_the_latched_floor_y():
    """With one floor, height_of(0) is identically what nav_agent used to latch
    on the first frame -- the single-floor path is unchanged by construction."""
    e = est()
    cam_y = 1.42
    assert e.update(cam_y, 0) == 0
    assert e.height_of(0) == pytest.approx(cam_y - CAM_H)


def test_flat_scene_never_creates_a_second_floor():
    e = est()
    drive(e, [0.0], steps_each=200)
    assert len(e.levels) == 1
    assert e.current == 0
    assert e.transitions == []


def test_small_height_noise_stays_on_one_floor():
    """Sunken living rooms and raised thresholds must not register as floors."""
    e = est()
    rng = np.random.default_rng(0)
    for i in range(300):
        e.update(0.0 + CAM_H + float(rng.normal(0, 0.08)), i)
    assert len(e.levels) == 1


def test_split_level_below_merge_is_not_a_new_floor():
    e = est()
    drive(e, [0.0, 0.45, 0.0], steps_each=40)
    assert len(e.levels) == 1


# ----------------------------------------------------------- climbing stairs


def test_climbing_registers_one_new_floor():
    e = est()
    drive(e, [0.0], steps_each=20)
    for i, y in enumerate(ramp(0.0, 2.8, 20)):
        e.update(y + CAM_H, 20 + i)
    drive(e, [2.8], steps_each=20, start_step=40)
    assert len(e.levels) == 2
    assert e.current != 0
    assert e.height_of(e.current) == pytest.approx(2.8, abs=0.4)


def test_index_is_frozen_while_on_stairs():
    """Mid-staircase the floor id must not move -- everything downstream
    (costmaps, room ids, blacklists) is keyed on it."""
    e = est()
    drive(e, [0.0], steps_each=20)
    seen = [e.update(y + CAM_H, 20 + i) for i, y in enumerate(ramp(0.0, 1.2, 8))]
    assert set(seen) == {0}
    assert e.on_stairs


def test_half_a_staircase_and_back_creates_nothing():
    e = est()
    drive(e, [0.0], steps_each=20)
    step = 20
    for y in ramp(0.0, 1.4, 8) + ramp(1.4, 0.0, 8):
        e.update(y + CAM_H, step)
        step += 1
    drive(e, [0.0], steps_each=20, start_step=step)
    assert len(e.levels) == 1
    assert e.current == 0


def test_returning_to_a_known_floor_reuses_its_id():
    e = est()
    drive(e, [0.0], steps_each=20)
    drive(e, [2.8], steps_each=20, start_step=20)
    up = e.current
    drive(e, [0.0], steps_each=20, start_step=40)
    assert e.current == 0
    assert len(e.levels) == 2
    drive(e, [2.8], steps_each=20, start_step=60)
    assert e.current == up  # not a third level


def test_dwell_is_required_before_a_switch_commits():
    e = est(min_dwell_steps=6)
    drive(e, [0.0], steps_each=20)
    drive(e, [2.8], steps_each=20, start_step=20)   # floor 1 now known
    drive(e, [0.0], steps_each=20, start_step=40)
    assert e.current == 0
    # Fewer frames than the dwell on the far floor -> no commit
    for i in range(3):
        e.update(2.8 + CAM_H, 60 + i)
    assert e.current == 0


def test_no_thrash_across_the_level_boundary():
    """Oscillating around a boundary must not produce a stream of switches."""
    e = est()
    drive(e, [0.0], steps_each=20)
    drive(e, [2.8], steps_each=20, start_step=20)
    step = 40
    for _ in range(30):
        for y in (0.0, 2.8):
            e.update(y + CAM_H, step)
            step += 1
    assert len(e.levels) == 2
    assert len(e.transitions) < 10  # not one per sample


def test_new_level_m_blocks_a_landing():
    """A 1.4 m landing must not register as a floor at the default, and does
    once new_level_m is lowered below it."""
    strict = est()
    lenient = est(new_level_m=0.6)
    for e in (strict, lenient):
        drive(e, [0.0], steps_each=20)
        drive(e, [1.4], steps_each=30, start_step=20)
    assert len(strict.levels) == 1
    assert len(lenient.levels) == 2


def test_landings_in_a_real_descent_are_not_floors():
    """Regression from XB4GS9ShBRE ep3: one descent 2.796 -> 0.196 registered
    FOUR floors at new_level_m=0.6, inventing landings at 1.879 and 1.253.
    The real scene has storeys 2.7-2.9 m apart."""
    heights = [2.796, 1.879, 1.253, 0.196]
    e = est()
    for i, y in enumerate(heights):
        drive(e, [y], steps_each=25, start_step=i * 25)
    assert len(e.levels) == 2, f"expected 2 storeys, got {e.sorted_levels()}"
    got = sorted(h for _, h in e.sorted_levels())
    assert got[0] == pytest.approx(0.196, abs=0.05)
    assert got[1] == pytest.approx(2.796, abs=0.05)


def test_level_height_refines_to_the_true_floor():
    """Regression from XB4GS9ShBRE ep3: the descent registered its new level at
    0.832 (part-way down the stairs) for a floor whose true height is 0.196.
    Uncorrected, that 0.64 m error exceeds level_tol_m, so the agent would read
    as permanently on-stairs while standing on the floor."""
    e = est()
    drive(e, [2.796], steps_each=30)
    step = 30
    for y in ramp(2.796, 0.196, 12):  # descend, registering somewhere en route
        e.update(y + CAM_H, step)
        step += 1
    drive(e, [0.196], steps_each=60, start_step=step)  # then walk the floor

    assert len(e.levels) == 2
    lower = min(e.levels.values())
    assert lower == pytest.approx(0.196, abs=0.1), f"level stuck at {lower}"
    assert not e.on_stairs, "standing on the real floor must not read as stairs"


def test_refinement_never_drags_a_separate_storey():
    e = est()
    drive(e, [0.0], steps_each=40)
    drive(e, [2.8], steps_each=40, start_step=40)
    drive(e, [0.0], steps_each=200, start_step=80)  # lots of time downstairs
    heights = sorted(e.levels.values())
    assert heights[1] == pytest.approx(2.8, abs=0.1), "upper storey was dragged down"


def walk(e, y, n, start_step=0, step_m=0.25, x0=0.0):
    """Feed n frames at height y while walking in a straight line."""
    for i in range(n):
        e.update(y + CAM_H, start_step + i, xy=(x0 + i * step_m, 0.0))


def test_a_storey_below_new_level_m_commits_if_there_is_room_to_walk():
    """The fix for aborted climbs: three episodes stalled at ~1.6 m, under
    new_level_m=1.8, so no floor was ever committed and the agent wandered back
    down. Horizontal room proves it is a real storey."""
    e = est()
    walk(e, 0.0, 30)
    walk(e, 1.6, 40, start_step=30)          # 40 steps x 0.25 m = 10 m of walking
    assert len(e.levels) == 2, "a walkable storey at 1.6 m was not committed"
    assert e.current != 0


def test_a_landing_at_the_same_height_is_still_rejected():
    """The counterpart: identical height, but only a landing's worth of room.
    This is what makes the horizontal-run rule safe."""
    e = est()
    walk(e, 0.0, 30)
    # A 1.2 m landing: the agent shuffles back and forth, never getting far.
    for i in range(40):
        e.update(1.6 + CAM_H, 30 + i, xy=(0.6 if i % 2 else 0.0, 0.0))
    assert len(e.levels) == 1, f"landing registered as a storey: {e.sorted_levels()}"


def test_horizontal_run_can_be_disabled():
    e = est(min_horizontal_run_m=0.0)
    walk(e, 0.0, 30)
    walk(e, 1.6, 40, start_step=30)
    assert len(e.levels) == 1  # only new_level_m applies


def test_full_storey_still_commits_without_walking():
    """The `new_level_m` route must not need horizontal evidence."""
    e = est()
    drive(e, [0.0], steps_each=30)
    drive(e, [2.8], steps_each=30, start_step=30)   # no xy fed at all
    assert len(e.levels) == 2


def test_merge_m_still_rejects_a_sunken_room_however_far_you_walk():
    """Horizontal room must not promote a 0.45 m split level to a storey."""
    e = est()
    walk(e, 0.0, 30)
    walk(e, 0.45, 60, start_step=30)
    assert len(e.levels) == 1


def test_a_dwell_rule_alone_cannot_reject_a_landing():
    """The agent lingers 25+ steps on a landing, so no dwell length separates
    it from a floor -- the separation threshold is what does the work."""
    e = est(new_level_m=0.6, min_dwell_steps=20)
    drive(e, [0.0], steps_each=20)
    drive(e, [1.2], steps_each=60, start_step=20)
    assert len(e.levels) == 2  # dwell did not save it
    assert len(est().levels) == 0  # ...but the default threshold would have


# -------------------------------------------------------------- stable ids


def test_a_basement_found_later_does_not_renumber_existing_floors():
    e = est()
    drive(e, [0.0], steps_each=20)
    drive(e, [2.8], steps_each=20, start_step=20)
    ground, upper = 0, e.current
    ground_y, upper_y = e.height_of(ground), e.height_of(upper)
    drive(e, [0.0], steps_each=20, start_step=40)
    drive(e, [-2.7], steps_each=30, start_step=60)
    assert e.height_of(ground) == ground_y
    assert e.height_of(upper) == upper_y
    assert e.current not in (ground, upper)
    # Ids are creation-ordered; sorted_levels gives the height order.
    assert [fid for fid, _ in e.sorted_levels()] == [e.current, ground, upper]


def test_floor_of_height_picks_the_nearest_level():
    e = est()
    drive(e, [0.0], steps_each=20)
    drive(e, [2.8], steps_each=20, start_step=20)
    assert e.floor_of_height(0.1) == 0        # an object 10 cm off the ground
    assert e.floor_of_height(3.3) == e.current  # one sitting on a table upstairs


# ------------------------------------------------------------- histogram


def test_histogram_finds_two_peaks():
    rng = np.random.default_rng(0)
    ys = np.concatenate([rng.normal(0.0, 0.02, 500), rng.normal(2.8, 0.02, 500)])
    peaks = height_histogram_peaks(ys)
    assert len(peaks) == 2
    assert peaks[0] == pytest.approx(0.0, abs=0.1)
    assert peaks[1] == pytest.approx(2.8, abs=0.1)


def test_histogram_handles_degenerate_input():
    assert height_histogram_peaks([]) == []
    assert height_histogram_peaks([1.0, 1.0]) == [1.0]


def test_observe_points_adds_but_never_switches():
    """The secondary source may pre-register an unvisited floor; it must not
    move the agent's committed floor."""
    e = est()
    drive(e, [0.0], steps_each=20)
    rng = np.random.default_rng(1)
    added = e.observe_points(np.concatenate([
        rng.normal(0.0, 0.02, 400), rng.normal(2.8, 0.02, 400)
    ]))
    assert len(added) == 1
    assert len(e.levels) == 2
    assert e.current == 0

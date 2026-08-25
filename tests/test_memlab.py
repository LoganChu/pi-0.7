"""Tests for the 2-D MEM lab.

The encoder tests are the load-bearing ones: they pin the three claims taken
from the paper, so that a refactor that quietly breaks weight sharing or the
history drop fails here rather than in a figure.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from memlab.cost import flops, joint_flops, tokens_per_layer
from memlab.encoder import EncoderConfig, MemEncoder
from memlab.oracle import collect, run_episode
from memlab.render import cell_on_screen, filmstrip_steps, rollout, save_gif
from memlab.tasks import COUNTER_SPOTS, TABLE_CELLS, make_task
from memlab.world import (
    FRAME_PX,
    Cell,
    Kitchen,
    Obj,
    default_layout,
    nearest_interactable,
    reachable,
    standing_cell,
)


# -- world ---------------------------------------------------------------


def test_observation_shape_and_dtype() -> None:
    kitchen = Kitchen()
    frame = kitchen.reset({0: Obj("bread", container=0)})
    assert frame.shape == (FRAME_PX, FRAME_PX, 3)
    assert frame.dtype == np.uint8


def test_agent_cannot_walk_through_geometry() -> None:
    kitchen = Kitchen()
    kitchen.reset({})
    for _ in range(60):
        kitchen.step([0.0, -0.22, 0.0])  # straight into the top counter run
    row = int(np.floor(kitchen.pos[1]))
    assert kitchen.layout.grid[row, int(np.floor(kitchen.pos[0]))] == Cell.FLOOR


def test_closed_cabinet_hides_its_contents() -> None:
    kitchen = Kitchen()
    kitchen.reset({0: Obj("bread", container=2)})
    assert kitchen._visible_cell(kitchen.objects[0]) is None
    kitchen.cabinets[2].is_open = True
    assert kitchen._visible_cell(kitchen.objects[0]) == kitchen.cabinets[2].cell


def test_grip_is_edge_triggered() -> None:
    """Holding the gripper open must not re-fire the interaction every step."""
    kitchen = Kitchen()
    kitchen.reset({0: Obj("bread", container=2)})
    kitchen.pos = np.array([5.5, 2.4])
    for _ in range(5):
        kitchen.step([0.0, 0.0, 1.0])
    assert sum(e.kind == "open" for e in kitchen.events) == 1


def test_episodes_are_deterministic() -> None:
    first = collect("find_object", 3, seed=7, keep_frames=False)
    second = collect("find_object", 3, seed=7, keep_frames=False)
    for a, b in zip(first, second):
        assert np.array_equal(a.actions, b.actions)


# -- reach geometry ------------------------------------------------------


def test_standing_cell_agrees_with_the_simulator() -> None:
    """Anywhere `standing_cell` proposes, the simulator must act on that cell."""
    layout = default_layout()
    for target in list(layout.cabinets) + list(TABLE_CELLS) + list(COUNTER_SPOTS):
        stand = standing_cell(layout, target)
        assert stand is not None, target
        centre = np.array([stand[1] + 0.5, stand[0] + 0.5])
        assert nearest_interactable(layout, centre) == target


def test_ambiguous_cell_is_rejected_not_silently_mishandled() -> None:
    """(10, 1) ties with the sink at (8, 1), so it must report unreachable."""
    layout = default_layout()
    assert not reachable(layout, (10, 1))
    assert (10, 1) not in COUNTER_SPOTS


# -- tasks and oracle ----------------------------------------------------


@pytest.mark.parametrize("task", ["find_object", "tidy_up"])
def test_oracle_solves_its_own_tasks(task: str) -> None:
    episodes = collect(task, 12, seed=11, keep_frames=False)
    assert all(e.success for e in episodes)


def test_actions_are_continuous_and_bounded() -> None:
    episode = collect("find_object", 1, seed=2, keep_frames=False)[0]
    assert episode.actions.shape[1] == 3
    assert np.abs(episode.actions[:, :2]).max() <= 0.22 + 1e-9
    assert np.abs(episode.actions[:, 2]).max() <= 1.0


def test_frames_actions_and_subtasks_stay_aligned() -> None:
    episode = collect("find_object", 1, seed=5)[0]
    assert len(episode.frames) == len(episode.actions) == len(episode.subtasks)


def test_search_length_varies_across_episodes() -> None:
    """If every demo opened the same number of cabinets, memory would be moot."""
    episodes = collect("find_object", 20, seed=4, keep_frames=False)
    opens = {sum(e.kind == "open" for e in ep.events) for ep in episodes}
    assert len(opens) > 1


def test_tidy_up_requires_closing_every_door() -> None:
    rng = np.random.default_rng(0)
    spec = make_task("tidy_up", rng)
    episode = run_episode(spec, keep_frames=False)
    assert episode.success
    assert any(e.kind == "close" for e in episode.events)


# -- encoder: the three claims -------------------------------------------


def test_output_is_the_current_frames_tokens() -> None:
    cfg = EncoderConfig()
    encoder = MemEncoder(cfg).eval()
    frames = torch.randint(0, 255, (2, 6, 64, 64, 3), dtype=torch.uint8)
    with torch.no_grad():
        out = encoder(frames)
    assert out.shape == (2, cfg.n_patches, cfg.dim)


def test_sharing_weights_adds_no_parameters() -> None:
    """Claim 3: no new learnable parameters over a single-image ViT."""
    single = MemEncoder(EncoderConfig(temporal_every=0)).n_params()
    shared = MemEncoder(EncoderConfig(temporal_every=4)).n_params()
    separate = MemEncoder(EncoderConfig(share_time_space_weights=False)).n_params()
    assert shared == single
    assert separate > single


def test_history_reaches_the_output_only_through_temporal_attention() -> None:
    frames = torch.randint(0, 255, (2, 6, 64, 64, 3), dtype=torch.uint8)
    altered = frames.clone()
    altered[:, 0] = 0  # change the oldest frame only

    torch.manual_seed(0)
    with_temporal = MemEncoder(EncoderConfig(temporal_every=4)).eval()
    torch.manual_seed(0)
    without = MemEncoder(EncoderConfig(temporal_every=0)).eval()

    with torch.no_grad():
        moved = (with_temporal(frames) - with_temporal(altered)).abs().max()
        still = (without(frames) - without(altered)).abs().max()
    assert moved > 1e-6
    assert still == 0.0


def test_temporal_layers_above_the_cut_are_inert() -> None:
    cfg = EncoderConfig(temporal_every=4, drop_history_after=6)
    assert cfg.temporal_layers == (3, 7, 11)
    assert cfg.effective_temporal_layers == (3,)
    assert EncoderConfig(temporal_every=12).effective_temporal_layers == ()


def test_dropping_changes_the_output() -> None:
    """The drop is a real architectural change, not just a cost accounting one."""
    frames = torch.randint(0, 255, (1, 8, 64, 64, 3), dtype=torch.uint8)
    torch.manual_seed(0)
    dropped = MemEncoder(EncoderConfig(drop_history_after=6)).eval()
    torch.manual_seed(0)
    kept = MemEncoder(EncoderConfig(drop_history_after=12)).eval()
    with torch.no_grad():
        assert not torch.allclose(dropped(frames), kept(frames))


# -- cost model ----------------------------------------------------------


def test_token_schedule_steps_down_at_the_cut() -> None:
    cfg = EncoderConfig(drop_history_after=6, depth=12)
    assert tokens_per_layer(cfg, 18) == [18] * 6 + [1] * 6
    assert tokens_per_layer(cfg, 18, drop=False) == [18] * 12


def test_dropping_is_cheaper_and_the_gap_grows() -> None:
    cfg = EncoderConfig()
    gaps = [
        flops(cfg, t, drop=False).total - flops(cfg, t).total for t in (2, 8, 18)
    ]
    assert gaps == sorted(gaps)
    assert all(g > 0 for g in gaps)


# -- visibility and rendering --------------------------------------------


def test_contents_of_a_closed_cabinet_are_not_visible() -> None:
    kitchen = Kitchen()
    kitchen.reset({0: Obj("bread", container=2)})
    kitchen.pos = np.array([5.5, 2.4])  # standing right at cabinet 2
    assert 0 not in kitchen.visible_objects()
    kitchen.step([0.0, 0.0, 1.0])
    assert 0 in kitchen.visible_objects()


def test_held_and_distant_objects() -> None:
    kitchen = Kitchen()
    kitchen.reset({0: Obj("mug", cell=(10, 8))})
    kitchen.pos = np.array([2.5, 2.5])  # far corner
    assert 0 not in kitchen.visible_objects()
    kitchen.objects[0].held, kitchen.objects[0].cell = True, None
    kitchen.holding = 0
    assert 0 in kitchen.visible_objects()


def test_cell_on_screen_matches_the_simulator() -> None:
    """The geometry helper must agree with what the world actually draws."""
    kitchen = Kitchen()
    kitchen.reset({0: Obj("mug", cell=(10, 8))})
    positions, truth = [], []
    for x in np.arange(1.5, 10.5, 0.5):
        kitchen.pos = np.array([x, 8.5])
        positions.append(kitchen.pos.copy())
        truth.append(0 in kitchen.visible_objects())
    predicted = cell_on_screen(np.stack(positions), (10, 8))
    assert list(predicted) == truth


def test_keep_global_records_an_aligned_god_view() -> None:
    episode = run_episode(
        make_task("find_object", np.random.default_rng(0)), keep_global=True
    )
    assert len(episode.global_frames) == len(episode)
    assert episode.global_frames.shape[1:] == (96, 96, 3)
    assert len(episode.visible) == len(episode)


def test_global_frames_are_off_by_default() -> None:
    """Recording the god view is opt-in; training on it would be cheating."""
    episode = collect("find_object", 1, seed=0)[0]
    assert len(episode.global_frames) == 0


def test_filmstrip_steps_are_sorted_and_in_range() -> None:
    episode = rollout("find_object", 0)
    steps = filmstrip_steps(episode, 6)
    assert steps == sorted(set(steps))
    assert all(0 <= s < len(episode) for s in steps)


def test_save_gif_writes_a_playable_file(tmp_path) -> None:
    episode = rollout("find_object", 0)
    out = save_gif(episode, tmp_path / "run.gif", stride=8)
    assert out.exists() and out.stat().st_size > 0
    with Image.open(out) as img:
        assert img.format == "GIF"
        assert img.n_frames > 1


def test_save_gif_refuses_without_global_frames() -> None:
    episode = collect("find_object", 1, seed=0)[0]
    with pytest.raises(ValueError, match="keep_global"):
        save_gif(episode)


def test_joint_attention_is_the_expensive_baseline() -> None:
    cfg = EncoderConfig()
    assert joint_flops(cfg, 18).total > flops(cfg, 18, drop=False).total
    # ...but identical at a single frame, where there is no history to factorize.
    assert joint_flops(cfg, 1).total == pytest.approx(flops(cfg, 1).total, rel=1e-9)

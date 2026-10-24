"""Unit tests for Log-Odds Metric Occupancy Grid Mapping and Clearance."""

import pytest
import numpy as np

from scene_graph.mapping.occupancy_grid import OccupancyGrid, OccupancyGridConfig


def test_initial_state_unknown():
    cfg = OccupancyGridConfig(resolution_m=0.10, width_m=2.0, height_m=2.0, origin_x_m=0.0, origin_y_m=0.0)
    grid = OccupancyGrid(cfg)
    assert np.all(grid.grid == OccupancyGrid.UNKNOWN)
    assert np.all(grid.log_odds == 0.0)


def test_hit_updates_occupied():
    cfg = OccupancyGridConfig(resolution_m=0.10, width_m=2.0, height_m=2.0, origin_x_m=0.0, origin_y_m=0.0)
    grid = OccupancyGrid(cfg)

    # Cast a ray hitting (1.0, 1.0)
    grid.update_ray(start_x=0.0, start_y=0.0, end_x=1.0, end_y=1.0, is_hit=True)
    c_hit, r_hit = grid.world_to_grid(1.0, 1.0)

    # Hit endpoint should be marked OCCUPIED
    assert grid.grid[r_hit, c_hit] == OccupancyGrid.OCCUPIED
    assert grid.log_odds[r_hit, c_hit] >= cfg.l_occ_thresh

    # Cells along ray should be marked FREE
    c_mid, r_mid = grid.world_to_grid(0.5, 0.5)
    assert grid.grid[r_mid, c_mid] == OccupancyGrid.FREE
    assert grid.log_odds[r_mid, c_mid] <= cfg.l_free_thresh


def test_ghost_map_clearance_transition():
    """Verify that a cell previously marked OCCUPIED transitions back to FREE when vacated."""
    cfg = OccupancyGridConfig(resolution_m=0.10, width_m=2.0, height_m=2.0, origin_x_m=0.0, origin_y_m=0.0)
    grid = OccupancyGrid(cfg)

    c, r = grid.world_to_grid(0.5, 0.5)

    # Step 1: Obstacle appears at (0.5, 0.5)
    grid.update_ray(start_x=0.0, start_y=0.5, end_x=0.5, end_y=0.5, is_hit=True)
    assert grid.grid[r, c] == OccupancyGrid.OCCUPIED
    assert grid.log_odds[r, c] >= cfg.l_occ_thresh

    # Step 2: Obstacle moves away. Sensor rays now penetrate through (0.5, 0.5) to a distant wall at (1.5, 0.5)
    # Each penetrating ray subtracts l_free (-0.40).
    for _ in range(4):
        grid.update_ray(start_x=0.0, start_y=0.5, end_x=1.5, end_y=0.5, is_hit=False)

    # Cell must clear from OCCUPIED back to FREE!
    assert grid.grid[r, c] == OccupancyGrid.FREE
    assert grid.log_odds[r, c] <= cfg.l_free_thresh


def test_log_odds_clamping_bounds():
    cfg = OccupancyGridConfig(resolution_m=0.10, width_m=1.0, height_m=1.0, origin_x_m=0.0, origin_y_m=0.0)
    grid = OccupancyGrid(cfg)

    c, r = grid.world_to_grid(0.5, 0.5)
    for _ in range(25):
        grid.update_ray(start_x=0.0, start_y=0.5, end_x=0.5, end_y=0.5, is_hit=True)

    assert grid.log_odds[r, c] == pytest.approx(cfg.l_max)

    for _ in range(50):
        grid.update_ray(start_x=0.0, start_y=0.5, end_x=0.9, end_y=0.5, is_hit=False)

    assert grid.log_odds[r, c] == pytest.approx(cfg.l_min)


def test_probabilistic_multi_session_fusion():
    """Verify that multi-session fusion resolves relocated obstacles properly."""
    cfg = OccupancyGridConfig(resolution_m=0.10, width_m=2.0, height_m=2.0, origin_x_m=0.0, origin_y_m=0.0)
    grid_session_1 = OccupancyGrid(cfg)
    grid_session_2 = OccupancyGrid(cfg)

    c, r = cfg.origin_x_m, cfg.origin_y_m
    c_tgt, r_tgt = grid_session_1.world_to_grid(0.8, 0.8)

    # Session 1: Chair was at (0.8, 0.8)
    grid_session_1.update_ray(start_x=0.0, start_y=0.8, end_x=0.8, end_y=0.8, is_hit=True)
    assert grid_session_1.grid[r_tgt, c_tgt] == OccupancyGrid.OCCUPIED

    # Session 2: Chair was relocated; 4 free observations at (0.8, 0.8)
    for _ in range(4):
        grid_session_2.update_ray(start_x=0.0, start_y=0.8, end_x=1.6, end_y=0.8, is_hit=False)
    assert grid_session_2.grid[r_tgt, c_tgt] == OccupancyGrid.FREE

    # Fuse Session 2 into Session 1
    grid_session_1.fuse_log_odds(grid_session_2)

    # Net log-odds should be negative, confirming the ghost chair was cleared!
    assert grid_session_1.grid[r_tgt, c_tgt] == OccupancyGrid.FREE

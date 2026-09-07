# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List

import numpy as np

from ascentnav.vendor.vlfm.mapping.traj_visualizer import TrajectoryVisualizer


class BaseMap:
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = 0.0
    _map_dtype: np.dtype = np.dtype(np.float32)

    def __init__(self, size: int = 1000, pixels_per_meter: int = 20, *args: Any, **kwargs: Any):
        """
        Args:
            size: The size of the map in pixels.
        """
        self.pixels_per_meter = pixels_per_meter
        self.size = size
        self._map = np.zeros((size, size), dtype=self._map_dtype)
        self._episode_pixel_origin = np.array([size // 2, size // 2])
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)
        # DEVIATION from the vendored file. `_camera_positions` is declared as a
        # CLASS attribute above and `update_agent_traj` appends to it in place,
        # so without this rebinding every map ever constructed -- every floor,
        # every episode, in one process -- shares one trajectory. VLFM only ever
        # builds one map per process, so it never shows there; here it painted
        # the previous episode's path onto the next episode's map and the
        # previous floor's path onto the new floor's. Read only by `visualize`
        # (checked: no decision touches it), so this is cosmetic in effect --
        # but the visualisation is what the stair behaviour is being judged from.
        self._camera_positions: List[np.ndarray] = []
        self._last_camera_yaw = 0.0

    def reset(self) -> None:
        self._map.fill(0)
        self._camera_positions = []
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)

    def update_agent_traj(self, robot_xy: np.ndarray, robot_heading: float) -> None:
        self._camera_positions.append(robot_xy)
        self._last_camera_yaw = robot_heading

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """Converts an array of (x, y) coordinates to pixel coordinates.

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        px[:, 0] = self._map.shape[0] - px[:, 0]
        return px.astype(int)

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """Converts an array of pixel coordinates to (x, y) coordinates.

        Args:
            px: The array of pixel coordinates to convert.

        Returns:
            The array of (x, y) coordinates.
        """
        px_copy = px.copy()
        px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0]
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        return points[:, ::-1]

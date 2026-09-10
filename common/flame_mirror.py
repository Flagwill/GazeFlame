"""Shared FLAME head renderer for the GazeFlame pipeline.

Renders a 3D FLAME head whose eyeballs follow a gaze direction with an
offscreen pyrender context, returning a BGR numpy image ready for cv2.

Also provides ``compose_with_head``: side-by-side composition of an annotated
input frame and the FLAME head panel so both visualizations live in one
cv2 window.

The offscreen renderer's OpenGL context is bound to the thread that creates
it, so ``FlameMirror`` must be created and used on the same thread.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FLAME_ROOT = REPO_ROOT / "flame"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FLAME_ROOT))


class FlameMirror:
    """Reusable FLAME head whose eyeballs track a live gaze direction."""

    def __init__(self, device="cpu", width=320, height=320):
        import pyrender

        import gaze_viewer
        from config import FLAMEConfig
        from FLAME import FLAME

        self._gv = gaze_viewer
        self.width = width
        self.height = height

        config = FLAMEConfig()
        config.batch_size = 1
        # FLAMEConfig 默认路径相对 flame/, 转成绝对路径以免依赖 cwd。
        config.flame_model_path = str(FLAME_ROOT / config.flame_model_path)
        config.static_landmark_embedding_path = str(
            FLAME_ROOT / config.static_landmark_embedding_path
        )
        config.dynamic_landmark_embedding_path = str(
            FLAME_ROOT / config.dynamic_landmark_embedding_path
        )

        self.device = torch.device(device)
        self.flame = FLAME(config).to(self.device)
        self.shape = torch.zeros(
            1, config.shape_params, dtype=torch.float32, device=self.device
        )
        self.expression = torch.zeros(
            1, config.expression_params, dtype=torch.float32, device=self.device
        )
        self.pose = torch.zeros(
            1, config.pose_params, dtype=torch.float32, device=self.device
        )
        self.neck_pose = (
            torch.zeros(1, 3, dtype=torch.float32, device=self.device)
            if config.optimize_neckpose
            else None
        )
        self._camera_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._renderer = pyrender.OffscreenRenderer(width, height)

    def render(self, gaze: np.ndarray) -> np.ndarray:
        """Render the head looking along ``gaze``; return a BGR numpy image."""
        gv = self._gv
        target_gaze = gv.normalize(np.asarray(gaze, dtype=np.float32))

        eye_axis_angle = gv.forward_to_axis_angle(target_gaze)
        eye_pose_np = np.concatenate(
            [eye_axis_angle, eye_axis_angle], axis=0
        ).reshape(1, 6)
        eye_pose = torch.tensor(eye_pose_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            vertices, landmarks, joints = self.flame(
                self.shape, self.expression, self.pose, self.neck_pose, eye_pose
            )

        vertices_np = vertices[0].detach().cpu().numpy()
        landmarks_np = landmarks[0].detach().cpu().numpy()
        joints_np = joints[0].detach().cpu().numpy()

        left_center, right_center, face_scale = gv.estimate_eye_centers(
            landmarks_np, joints_np
        )
        if np.dot(target_gaze, (self._camera_pos - left_center)) < 0:
            target_gaze = -target_gaze

        scene = gv.build_scene(
            vertices_np, self.flame.faces, left_center, right_center,
            target_gaze, face_scale,
        )
        color, _ = self._renderer.render(scene)
        # pyrender 输出是 RGB, cv2 的 imshow/imwrite 需要 BGR。
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Single-window composition
# ---------------------------------------------------------------------------
def compose_with_head(
    left_bgr: np.ndarray,
    head_bgr: np.ndarray | None,
    sep_w: int = 4,
) -> np.ndarray:
    """Concatenate ``left_bgr`` and a FLAME head panel side by side.

    The head is resized to match the left panel's height (or a fixed-size dark
    placeholder when no face has been detected yet), so the composed image
    keeps a stable layout in a single window.
    """
    h = left_bgr.shape[0]
    if head_bgr is not None and head_bgr.ndim == 3 and head_bgr.shape[0] > 0:
        hh, hw = head_bgr.shape[:2]
        panel_w = max(64, int(hw * (h / float(hh))))
        panel = cv2.resize(head_bgr, (panel_w, h))
    else:
        panel = np.zeros((h, max(h // 2, 96), 3), dtype=np.uint8)

    sep = np.full((h, sep_w, 3), 0, dtype=np.uint8)
    return np.hstack([left_bgr, sep, panel])

"""Shared FLAME head renderer for the GazeFlame pipeline.

Renders a 3D FLAME head whose eyeballs follow a gaze direction with an
offscreen pyrender context, returning a BGR numpy image ready for cv2.

Realtime design (important for fps):
    The pyrender Scene is built ONCE in ``__init__`` and reused by every
    ``render()`` call: the head mesh, sclerae, lids and pupil/arrow meshes live
    in fixed nodes and only the head/eye/lid node poses are updated per frame.
    Rebuilding the whole scene per frame cost ~44ms (trimesh + from_trimesh)
    AND invalidated pyrender's per-context buffer cache, pushing a 640x640
    render to ~140ms and capping ``run_webcam --flame`` at ~10fps.
    The head mesh is frozen at eye_pose=0: in FLAME the eye pose only rotates
    the (skin-colored) eyeball verts hidden in the sockets — the visible
    white/iris are the separate sclera/pupil spheres — so freezing it is
    visually equivalent (verified by pixel diff; residual <1px).
    ``_install_color_only_readback`` additionally skips the unused depth
    ``glReadPixels`` (~28ms at 640x640 on this GPU); if the installed pyrender
    internals ever differ, it silently falls back to the stock path.

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


def _install_color_only_readback(offscreen_renderer) -> bool:
    """让 pyrender 的离屏渲染只回读颜色缓冲，跳过没人用的深度缓冲。

    ``Renderer._read_main_framebuffer`` 每帧固定做两次 ``glReadPixels``
    (RGB color + float32 depth)；FlameMirror 只用 color，640x640 的同步
    深度回读在本机 (radeonsi) 要 ~28ms/帧。这里对**实例**打补丁，只
    blit + 回读 color。若 pyrender 内部结构对不上（版本变化）或平台是
    darwin（有 _resize_image 逻辑），直接放弃补丁返回 False，走原路径。
    """
    if sys.platform == "darwin":
        return False
    try:
        import types

        from OpenGL.GL import (
            glBindFramebuffer,
            glBlitFramebuffer,
            glReadPixels,
            GL_COLOR_BUFFER_BIT,
            GL_DRAW_FRAMEBUFFER,
            GL_LINEAR,
            GL_READ_FRAMEBUFFER,
            GL_RGB,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
        )
        from pyrender.constants import RenderFlags

        renderer = offscreen_renderer._renderer
        if not all(hasattr(renderer, a) for a in ("_main_fb", "_main_fb_ms", "_main_fb_dims")):
            return False

        def _read_color_only(self, scene, flags):  # noqa: ANN001 - 仿实例方法
            width = int(self._main_fb_dims[0])
            height = int(self._main_fb_dims[1])
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb_ms)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glBlitFramebuffer(
                0, 0, width, height, 0, 0, width, height,
                GL_COLOR_BUFFER_BIT, GL_LINEAR,
            )
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)
            if flags & RenderFlags.RGBA:
                fmt = GL_RGBA
                channels = 4
            else:
                fmt = GL_RGB
                channels = 3
            buf = glReadPixels(0, 0, width, height, fmt, GL_UNSIGNED_BYTE)
            color_im = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, channels))
            return np.flip(color_im, axis=0), None

        renderer._read_main_framebuffer = types.MethodType(_read_color_only, renderer)
        return True
    except Exception:  # noqa: BLE001 - 补丁失败只影响速度, 不影响正确性
        return False


class FlameMirror:
    """Reusable FLAME head whose eyeballs track a live gaze direction.

    The 3D scene (head + per-eye sclera/iris/pupil/lids/arrow) is constructed
    once in ``__init__``; ``render()`` only updates node poses and reuses the
    cached GPU buffers.
    """

    def __init__(self, device="cpu", width=320, height=320, show_arrows=True):
        import pyrender
        import trimesh

        import gaze_viewer
        from config import FLAMEConfig
        from FLAME import FLAME

        self._gv = gaze_viewer
        self.width = width
        self.height = height
        self.show_arrows = bool(show_arrows)

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
        self._color_only = _install_color_only_readback(self._renderer)

        # ---- 静态场景: 只在这里建一次, render() 复用 ---------------------
        # shape/expression/pose 恒为 0, eye_pose 只动皮肤色的眼球顶点,
        # 所以头部网格与眼睛中心/比例全部按 eye_pose=0 预算一次。
        zero_eye = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            vertices, landmarks, joints = self.flame(
                self.shape, self.expression, self.pose, self.neck_pose, zero_eye
            )
        vertices_np = vertices[0].detach().cpu().numpy()
        landmarks_np = landmarks[0].detach().cpu().numpy()
        joints_np = joints[0].detach().cpu().numpy()
        left_center, right_center, face_scale = self._gv.estimate_eye_centers(
            landmarks_np, joints_np
        )
        self._eye_centers = (left_center, right_center)
        self._face_scale = float(face_scale)
        # The old fixed z=1 camera made the FLAME head occupy only a small
        # fraction of the panel because this model is roughly 0.15 units wide.
        # Scale the camera to the model once; render() still only updates nodes.
        camera_distance = max(self._face_scale * 2.4, 0.18)
        self._camera_pos = np.array([0.0, 0.0, camera_distance], dtype=np.float32)

        eye_dist = float(np.linalg.norm(left_center - right_center))
        sclera_radius = max(eye_dist * 0.18, 0.008)
        pupil_radius = sclera_radius * 0.45
        eye_offset = sclera_radius * 0.2
        self._sclera_radius = sclera_radius
        self._eye_offset = eye_offset
        gaze_len = face_scale * 0.6
        line_radius = max(face_scale * 0.005, 0.0025)
        arrow_len = face_scale * 0.12
        cone_radius = line_radius * 3.0

        gv = self._gv
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])

        head_colors = np.ones([vertices_np.shape[0], 4]) * [0.83, 0.72, 0.62, 1.0]
        tri_mesh = trimesh.Trimesh(vertices_np, self.flame.faces, vertex_colors=head_colors)
        self._head_root = pyrender.Node(name="head_root")
        scene.add_node(self._head_root)
        scene.add(
            pyrender.Mesh.from_trimesh(tri_mesh, smooth=True),
            parent_node=self._head_root,
        )

        # 每只眼睛一个 "gaze root" 节点 (世界平移=socket 中心, 姿态=视线旋转)。
        # 子网格全部沿局部 +Z 建一次; render() 里 set_pose 整组转动。球体绕
        # 自身旋转不变形, 与原"每帧重建"的差别仅 sclera 前推量随视线转动,
        # <1px (见模块 docstring)。
        origin = np.zeros(3, dtype=np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        arrow_parts = []
        line = gv.make_cylinder_between(origin, z_axis * gaze_len,
                                        radius=line_radius, color=gv.GAZE_COLOR)
        if line is not None:
            arrow_parts.append(line)
        arrow_parts.append(
            gv.make_arrowhead(z_axis * gaze_len, z_axis, length=arrow_len,
                              radius=cone_radius, color=gv.GAZE_COLOR)
        )
        arrow_mesh = (
            arrow_parts[0] if len(arrow_parts) == 1
            else trimesh.util.concatenate(arrow_parts)
        )

        self._gaze_roots = []
        for center in (left_center, right_center):
            root = pyrender.Node(name=f"gaze_root_{center[0]:.4f}")
            scene.add_node(root, parent_node=self._head_root)

            sclera = trimesh.creation.uv_sphere(radius=sclera_radius)
            sclera.apply_scale([1.0, 0.92, 1.12])
            sclera.visual.vertex_colors = np.tile(
                np.array(gv.SCLERA_COLOR, dtype=np.float32),
                (sclera.vertices.shape[0], 1),
            )
            sclera.apply_translation([0.0, 0.0, eye_offset])
            scene.add(pyrender.Mesh.from_trimesh(sclera), parent_node=root)

            iris = trimesh.creation.uv_sphere(radius=sclera_radius * 0.64)
            iris.visual.vertex_colors = np.tile(
                np.array((0.20, 0.10, 0.045, 1.0), dtype=np.float32),
                (iris.vertices.shape[0], 1),
            )
            iris.apply_translation([0.0, 0.0, eye_offset + sclera_radius * 0.82])
            scene.add(pyrender.Mesh.from_trimesh(iris), parent_node=root)

            pupil = trimesh.creation.uv_sphere(radius=pupil_radius * 0.72)
            pupil.visual.vertex_colors = np.tile(
                np.array((0.018, 0.012, 0.008, 1.0), dtype=np.float32),
                (pupil.vertices.shape[0], 1),
            )
            pupil.apply_translation([0.0, 0.0, eye_offset + sclera_radius * 1.14])
            scene.add(pyrender.Mesh.from_trimesh(pupil), parent_node=root)

            highlight = trimesh.creation.uv_sphere(radius=sclera_radius * 0.12)
            highlight.visual.vertex_colors = np.tile(
                np.array((1.0, 1.0, 1.0, 1.0), dtype=np.float32),
                (highlight.vertices.shape[0], 1),
            )
            highlight.apply_translation(
                [-sclera_radius * 0.24, sclera_radius * 0.27,
                 eye_offset + sclera_radius * 1.22]
            )
            scene.add(pyrender.Mesh.from_trimesh(highlight), parent_node=root)

            # Skin-colored upper/lower lids are moved toward the center during
            # a blink.  They remain part of the cached scene.
            lid_color = np.array((0.70, 0.53, 0.43, 1.0), dtype=np.float32)
            upper_lid = trimesh.creation.uv_sphere(radius=1.0)
            upper_lid.apply_scale([sclera_radius * 1.25,
                                   sclera_radius * 0.34,
                                   sclera_radius * 0.23])
            upper_lid.visual.vertex_colors = np.tile(
                lid_color, (upper_lid.vertices.shape[0], 1)
            )
            upper_node = pyrender.Node(name=f"upper_lid_{center[0]:.4f}")
            scene.add_node(upper_node, parent_node=root)
            scene.add(pyrender.Mesh.from_trimesh(upper_lid), parent_node=upper_node)

            lower_lid = trimesh.creation.uv_sphere(radius=1.0)
            lower_lid.apply_scale([sclera_radius * 1.25,
                                   sclera_radius * 0.34,
                                   sclera_radius * 0.23])
            lower_lid.visual.vertex_colors = np.tile(
                lid_color, (lower_lid.vertices.shape[0], 1)
            )
            lower_node = pyrender.Node(name=f"lower_lid_{center[0]:.4f}")
            scene.add_node(lower_node, parent_node=root)
            scene.add(pyrender.Mesh.from_trimesh(lower_lid), parent_node=lower_node)

            if self.show_arrows:
                scene.add(pyrender.Mesh.from_trimesh(arrow_mesh.copy()), parent_node=root)
            self._gaze_roots.append(
                (root, np.asarray(center, dtype=np.float64), upper_node, lower_node)
            )

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        cam_pose = np.eye(4)
        cam_pose[2, 3] = camera_distance
        scene.add(camera, pose=cam_pose)
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=cam_pose
        )
        for offset in ((1.5, 1.0, 1.0), (-1.5, -1.0, 0.5)):
            light_pose = np.eye(4)
            light_pose[:3, 3] = offset
            scene.add(
                pyrender.PointLight(color=np.ones(3), intensity=1.2), pose=light_pose
            )
        self._scene = scene

    def render(
        self,
        gaze: np.ndarray,
        head_pose: np.ndarray | None = None,
        blink: float = 0.0,
    ) -> np.ndarray:
        """Render head pose, converged eyes and blink state to a BGR image.

        ``gaze`` is expressed in the render/world frame.  ``head_pose`` is an
        FLAME axis-angle vector in that frame; eye directions are converted
        back into head-local coordinates before their individual roots rotate.
        """
        gv = self._gv
        gaze_value = np.asarray(gaze, dtype=np.float32)
        if not np.all(np.isfinite(gaze_value)):
            raise ValueError("gaze contains NaN or infinity")
        target_gaze = gv.normalize(gaze_value)
        if np.dot(target_gaze, (self._camera_pos - self._eye_centers[0])) < 0:
            target_gaze = -target_gaze

        if head_pose is None:
            head_rotation = np.eye(3, dtype=np.float64)
        else:
            pose = np.asarray(head_pose, dtype=np.float64)
            if not np.all(np.isfinite(pose)):
                pose = np.zeros(3, dtype=np.float64)
            if pose.shape == (3, 3):
                head_rotation = pose
            else:
                head_rotation, _ = cv2.Rodrigues(pose.reshape(3, 1))
        head_pose_matrix = np.eye(4, dtype=np.float64)
        head_pose_matrix[:3, :3] = head_rotation
        self._scene.set_pose(self._head_root, head_pose_matrix)

        # A finite target, rather than parallel rays, makes the pupils gently
        # converge.  This is more natural at close range and remains stable for
        # all directions produced by the 2D gaze model.
        local_gaze = head_rotation.T @ target_gaze.astype(np.float64)
        convergence_distance = max(self._face_scale * 8.0, 0.1)
        target = local_gaze * convergence_distance
        blink = float(np.clip(blink, 0.0, 1.0))
        for root, center, upper_node, lower_node in self._gaze_roots:
            eye_direction = gv.normalize(target - center)
            axis_angle = gv.forward_to_axis_angle(eye_direction)
            rot, _ = cv2.Rodrigues(axis_angle.astype(np.float64))
            eye_pose = np.eye(4, dtype=np.float64)
            eye_pose[:3, :3] = rot
            eye_pose[:3, 3] = center
            self._scene.set_pose(root, eye_pose)

            lid_z = self._eye_offset + self._sclera_radius * 1.38
            lid_gap = self._sclera_radius * 1.98 * (1.0 - blink)
            for node, sign in ((upper_node, 1.0), (lower_node, -1.0)):
                lid_pose = np.eye(4, dtype=np.float64)
                lid_pose[:3, 3] = [0.0, sign * lid_gap * 0.5, lid_z]
                self._scene.set_pose(node, lid_pose)

        color, _ = self._renderer.render(self._scene)
        # pyrender 输出是 RGB, cv2 的 imshow/imwrite 需要 BGR。
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        """Release the offscreen context on the thread that owns it."""
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            renderer.delete()
            self._renderer = None


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

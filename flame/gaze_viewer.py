"""
Command-line FLAME gaze viewer.

Renders the FLAME head with the two eyeball joints oriented toward a given
3D gaze direction and shows the direction with a colored arrow per eye.

Examples:
    # Interactive window (requires an X11 display)
    python gaze_viewer.py --gaze-x 0 --gaze-y 0.2 --gaze-z 1

    # Render to a PNG without opening a window (headless / automation)
    python gaze_viewer.py --gaze-x 0 --gaze-y 0.2 --gaze-z 1 \
        --save-image out.png --headless
"""

from __future__ import annotations

import argparse

import numpy as np
import pyrender
import torch
import trimesh

from FLAME import FLAME
from config import add_flame_args, config_from_args

Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float64)


# --------------------------------------------------------------------------
# Vector / rotation helpers
# --------------------------------------------------------------------------
def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return ``v`` normalized; raise if its norm is effectively zero."""
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Gaze direction norm is zero. Please provide a non-zero vector.")
    return v / n


def rotation_matrix_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into an axis-angle vector."""
    trace = np.trace(rot)
    cos_theta = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)

    if np.pi - theta < 1e-5:
        # 180-degree case: pick axis from the diagonal terms.
        axis = np.sqrt(np.maximum((np.diag(rot) + 1.0) * 0.5, 0.0))
        if axis[0] < 1e-4 and axis[1] < 1e-4 and axis[2] < 1e-4:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        axis = normalize(axis)
        return (axis * theta).astype(np.float32)

    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    )
    axis = axis / (2.0 * np.sin(theta))
    axis = normalize(axis)
    return (axis * theta).astype(np.float32)


def forward_to_axis_angle(target_dir: np.ndarray) -> np.ndarray:
    """Axis-angle that rotates the forward vector (0, 0, 1) onto ``target_dir``."""
    src = FORWARD
    dst = normalize(target_dir.astype(np.float64))

    cross = np.cross(src, dst)
    dot = np.dot(src, dst)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0.0:
            return np.zeros(3, dtype=np.float32)
        return np.array([np.pi, 0.0, 0.0], dtype=np.float32)

    vx = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    rot = np.eye(3) + vx + vx.dot(vx) * (1.0 / (1.0 + dot))
    return rotation_matrix_to_axis_angle(rot)


def align_to_axis_transform(axis: np.ndarray) -> np.ndarray:
    """4x4 transform mapping the +Z axis onto ``axis``."""
    axis = normalize(axis.astype(np.float64))
    v = np.cross(Z_AXIS, axis)
    c = np.dot(Z_AXIS, axis)

    if np.linalg.norm(v) < 1e-8:
        rot = np.eye(3)
    else:
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )
        rot = np.eye(3) + vx + vx.dot(vx) * (1.0 / (1.0 + c))

    transform = np.eye(4)
    transform[:3, :3] = rot
    return transform


def make_cylinder_between(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float = 0.003,
    sections: int = 16,
    color: tuple | None = None,
):
    """A cylinder mesh spanning from ``p0`` to ``p1`` (None if too short)."""
    if color is None:
        color = (0.1, 0.3, 0.9, 1.0)

    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-8:
        return None

    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    cyl.visual.vertex_colors = np.tile(
        np.array(color, dtype=np.float32), (cyl.vertices.shape[0], 1)
    )

    transform = align_to_axis_transform(vec / length)
    transform[:3, 3] = p0 + 0.5 * vec
    cyl.apply_transform(transform)
    return cyl


def make_arrowhead(
    base: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    sections: int = 16,
    color: tuple = (0.1, 0.3, 0.9, 1.0),
):
    """A cone (arrow tip) whose base sits at ``base`` and points along ``direction``."""
    cone = trimesh.creation.cone(radius=radius, height=length, sections=sections)
    cone.visual.vertex_colors = np.tile(
        np.array(color, dtype=np.float32), (cone.vertices.shape[0], 1)
    )

    transform = align_to_axis_transform(direction)
    transform[:3, 3] = base
    cone.apply_transform(transform)
    return cone


# --------------------------------------------------------------------------
# Eye center estimation
# --------------------------------------------------------------------------
def estimate_eye_centers(
    landmarks: np.ndarray, joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Approximate the left/right eye centers from face landmarks and joints.

    Landmarks are split around the median x with a vertical tolerance, then
    each half's centroid is snapped to the nearest FLAME joint.

    Returns:
        ``(left_center, right_center, face_scale)``.
    """
    lm_xs = landmarks[:, 0]
    lm_ys = landmarks[:, 1]

    median_x = np.median(lm_xs)
    median_y = np.median(lm_ys)
    face_scale = max(float(np.ptp(lm_xs)), float(np.ptp(lm_ys)), 1e-4)
    y_tol = face_scale * 0.25

    left_idx = np.where(
        (landmarks[:, 0] < median_x) & (np.abs(landmarks[:, 1] - median_y) < y_tol)
    )[0]
    right_idx = np.where(
        (landmarks[:, 0] >= median_x) & (np.abs(landmarks[:, 1] - median_y) < y_tol)
    )[0]

    left_centroid = (
        landmarks[left_idx].mean(axis=0)
        if left_idx.size > 0
        else landmarks[landmarks[:, 0] < median_x].mean(axis=0)
    )
    right_centroid = (
        landmarks[right_idx].mean(axis=0)
        if right_idx.size > 0
        else landmarks[landmarks[:, 0] >= median_x].mean(axis=0)
    )

    left_joint_idx = np.argmin(np.linalg.norm(joints - left_centroid[None, :], axis=1))
    right_joint_idx = np.argmin(
        np.linalg.norm(joints - right_centroid[None, :], axis=1)
    )

    return joints[left_joint_idx], joints[right_joint_idx], face_scale


# --------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------
GAZE_COLOR = (0.1, 0.3, 0.9, 1.0)


def build_scene(
    vertices: np.ndarray,
    faces: np.ndarray,
    left_center: np.ndarray,
    right_center: np.ndarray,
    gaze_dir: np.ndarray,
    face_scale: float,
):
    """Build a pyrender scene: head + per-eye gaze arrow."""
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])

    head_colors = np.ones([vertices.shape[0], 4]) * [0.83, 0.72, 0.62, 1.0]
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=head_colors)
    scene.add(pyrender.Mesh.from_trimesh(tri_mesh, smooth=True))

    gaze_len = face_scale * 0.6
    sphere_radius = max(face_scale * 0.03, 0.006)
    line_radius = max(face_scale * 0.005, 0.0025)
    arrow_len = face_scale * 0.12
    cone_radius = line_radius * 3.0

    for center in (left_center, right_center):
        end = center + gaze_dir * gaze_len

        sphere = trimesh.creation.uv_sphere(radius=sphere_radius)
        sphere.apply_translation(center)
        sphere.visual.vertex_colors = list(GAZE_COLOR)
        scene.add(pyrender.Mesh.from_trimesh(sphere))

        line = make_cylinder_between(center, end, radius=line_radius, color=GAZE_COLOR)
        if line is not None:
            scene.add(pyrender.Mesh.from_trimesh(line))

        tip = make_arrowhead(end, gaze_dir, length=arrow_len, radius=cone_radius,
                             color=GAZE_COLOR)
        scene.add(pyrender.Mesh.from_trimesh(tip))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 1.0
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
    return scene


def render_scene_to_image(scene, width: int = 1280, height: int = 720) -> np.ndarray:
    """Render the scene offscreen and return an RGB numpy image."""
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()
    return color


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FLAME gaze direction viewer")
    add_flame_args(parser)

    parser.add_argument("--gaze-x", type=float, required=True,
                        help="Gaze direction X component")
    parser.add_argument("--gaze-y", type=float, required=True,
                        help="Gaze direction Y component")
    parser.add_argument("--gaze-z", type=float, required=True,
                        help="Gaze direction Z component")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"], help="Run device")
    parser.add_argument("--save-image", type=str, default=None,
                        help="Also render the scene to a PNG at this path")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open the interactive window "
                             "(requires --save-image)")
    parser.add_argument("--no-camera-flip", action="store_true",
                        help="Disable flipping the gaze to face the camera")
    parser.add_argument("--image-width", type=int, default=1280,
                        help="Offscreen render width (default: 1280)")
    parser.add_argument("--image-height", type=int, default=720,
                        help="Offscreen render height (default: 720)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.headless and not args.save_image:
        raise SystemExit("--headless requires --save-image <path>")

    config = config_from_args(args)
    config.batch_size = 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Using device:", device)

    flame = FLAME(config).to(device)

    shape_params = torch.zeros(1, config.shape_params, dtype=torch.float32, device=device)
    expression_params = torch.zeros(
        1, config.expression_params, dtype=torch.float32, device=device
    )
    pose_params = torch.zeros(1, config.pose_params, dtype=torch.float32, device=device)

    target_gaze = normalize(
        np.array([args.gaze_x, args.gaze_y, args.gaze_z], dtype=np.float32)
    )

    eye_axis_angle = forward_to_axis_angle(target_gaze)
    eye_pose_np = np.concatenate([eye_axis_angle, eye_axis_angle], axis=0).reshape(1, 6)
    eye_pose = torch.tensor(eye_pose_np, dtype=torch.float32, device=device)

    neck_pose = (
        torch.zeros(1, 3, dtype=torch.float32, device=device)
        if config.optimize_neckpose
        else None
    )

    vertices, landmarks, joints = flame(
        shape_params, expression_params, pose_params, neck_pose, eye_pose
    )

    vertices_np = vertices[0].detach().cpu().numpy()
    landmarks_np = landmarks[0].detach().cpu().numpy()
    joints_np = joints[0].detach().cpu().numpy()

    left_center, right_center, face_scale = estimate_eye_centers(
        landmarks_np, joints_np
    )

    # Keep the vector facing the camera for easier inspection in the default pose.
    camera_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if not args.no_camera_flip and np.dot(target_gaze, (camera_pos - left_center)) < 0:
        target_gaze = -target_gaze

    scene = build_scene(
        vertices_np, flame.faces, left_center, right_center, target_gaze, face_scale
    )

    print("Input gaze direction (normalized):", target_gaze)
    print("Eye axis-angle used:", eye_axis_angle)

    if args.save_image:
        from PIL import Image

        color = render_scene_to_image(scene, args.image_width, args.image_height)
        Image.fromarray(color).save(args.save_image)
        print("Saved render to:", args.save_image)

    if not args.headless:
        pyrender.Viewer(scene, use_raymond_lighting=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
FLAME demo: visualize 3D landmarks and eye-pose variations.

Loads the FLAME layer, creates a batch of neutral heads, and renders a few
frames with varying eyeball yaw into ``flame/outputs/frame_XX.png`` using an
offscreen renderer (no display required).

Examples:
    # Render frames to flame/outputs/
    python main.py

    # Also pop up an interactive window per frame (requires an X11 display)
    python main.py --interactive

Demo code adapted from FLAME_PyTorch
(https://github.com/soubhiksanyal/FLAME_PyTorch).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

import pyrender
import trimesh

from FLAME import FLAME
from config import get_config

RADIAN = np.pi / 180.0
OUT_DIR = "outputs"
FRAME_SIZE = (640, 480)


def estimate_eye_centers(
    landmarks: np.ndarray, joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Approximate left/right eye centers (median-x split + nearest joint)."""
    median_x = np.median(landmarks[:, 0])
    median_y = np.median(landmarks[:, 1])
    face_scale = max(np.ptp(landmarks[:, 0]), np.ptp(landmarks[:, 1]))
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

    left_joint = joints[np.argmin(np.linalg.norm(joints - left_centroid[None, :], axis=1))]
    right_joint = joints[
        np.argmin(np.linalg.norm(joints - right_centroid[None, :], axis=1))
    ]
    return left_joint, right_joint, face_scale


def axis_angle_to_dir(axis_angle: np.ndarray) -> np.ndarray:
    """Rotate the forward vector (0, 0, 1) by an axis-angle vector."""
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis_angle / theta
    forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cross = np.cross(axis, forward)
    dot = np.dot(axis, forward)
    return forward * np.cos(theta) + cross * np.sin(theta) + axis * dot * (1 - np.cos(theta))


def make_cylinder_between(
    p0: np.ndarray, p1: np.ndarray, radius: float = 0.002,
    sections: int = 16, color=(0.0, 0.0, 1.0, 1.0),
):
    """A cylinder mesh spanning from ``p0`` to ``p1``."""
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    cyl.visual.vertex_colors = np.tile(
        np.array(color, dtype=np.float32), (cyl.vertices.shape[0], 1)
    )
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = vec / length
    v = np.cross(z_axis, axis)
    c = np.dot(z_axis, axis)
    if np.linalg.norm(v) < 1e-8:
        rot = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx.dot(vx) * (1.0 / (1.0 + c))
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = p0 + vec * 0.5
    cyl.apply_transform(transform)
    return cyl


def build_frame_scene(vertices, faces, joints, eye_pose_np, landmarks):
    """Build a scene with the head, eye-center spheres and gaze cylinders."""
    vertices = vertices.detach().cpu().numpy().squeeze()
    joints = joints.detach().cpu().numpy().squeeze()
    landmarks = landmarks.detach().cpu().numpy().squeeze()
    eye_pose_np = np.asarray(eye_pose_np, dtype=np.float32)

    left_center, right_center, face_scale = estimate_eye_centers(landmarks, joints)

    left_dir = axis_angle_to_dir(eye_pose_np[0:3])
    right_dir = axis_angle_to_dir(eye_pose_np[3:6])
    camera_pos = np.array([0.0, 0.0, 1.0])
    if np.dot(left_dir, (camera_pos - left_center)) < 0:
        left_dir = -left_dir
    if np.dot(right_dir, (camera_pos - right_center)) < 0:
        right_dir = -right_dir

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
    tri_mesh = trimesh.Trimesh(
        vertices, faces,
        vertex_colors=np.ones([vertices.shape[0], 4]) * [0.8, 0.7, 0.6, 1.0],
    )
    scene.add(pyrender.Mesh.from_trimesh(tri_mesh, smooth=False))

    offset = face_scale * 0.02
    sphere_radius = max(face_scale * 0.03, 0.007)
    for center, direction in ((left_center, left_dir), (right_center, right_dir)):
        center_vis = center + direction * offset
        sphere = trimesh.creation.uv_sphere(radius=sphere_radius)
        sphere.apply_translation(center_vis)
        sphere.visual.vertex_colors = [0.0, 0.0, 1.0, 1.0]
        scene.add(pyrender.Mesh.from_trimesh(sphere))

    gaze_len = face_scale * 0.5
    cyl_radius = max(face_scale * 0.005, 0.003)
    l_cyl = make_cylinder_between(
        left_center + left_dir * offset,
        left_center + left_dir * (offset + gaze_len),
        radius=cyl_radius,
    )
    r_cyl = make_cylinder_between(
        right_center + right_dir * offset,
        right_center + right_dir * (offset + gaze_len),
        radius=cyl_radius,
    )
    if l_cyl is not None:
        scene.add(pyrender.Mesh.from_trimesh(l_cyl))
    if r_cyl is not None:
        scene.add(pyrender.Mesh.from_trimesh(r_cyl))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 1.0
    scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)
    return scene


def main(interactive: bool = False) -> int:
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    flamelayer = FLAME(config).to(device)
    faces = flamelayer.faces

    batch_size = 8
    shape_params = torch.zeros(batch_size, config.shape_params, device=device)
    pose_params = torch.zeros(batch_size, config.pose_params, dtype=torch.float32, device=device)
    expression_params = torch.zeros(batch_size, config.expression_params, dtype=torch.float32, device=device)

    vertice, landmark, joints = flamelayer(shape_params, expression_params, pose_params)
    print(vertice.size(), landmark.size(), joints.size())

    # Vary eyeball yaw for both eyes.
    eye_yaws = np.linspace(-20.0, 20.0, batch_size) * RADIAN
    eye_pose_np = np.zeros((batch_size, 6), dtype=np.float32)
    eye_pose_np[:, 0:3] = np.stack(
        [np.array([0.0, yaw, 0.0], dtype=np.float32) for yaw in eye_yaws]
    )
    eye_pose_np[:, 3:6] = eye_pose_np[:, 0:3]
    eye_pose = torch.tensor(eye_pose_np, dtype=torch.float32, device=device)

    vertice, landmark, joints = flamelayer(
        shape_params, expression_params, pose_params, None, eye_pose
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(*FRAME_SIZE)
    try:
        for i in range(batch_size):
            scene = build_frame_scene(
                vertice[i], faces, joints[i], eye_pose_np[i], landmark[i]
            )
            color, _ = renderer.render(scene)
            out_path = os.path.join(OUT_DIR, f"frame_{i:02d}.png")
            Image.fromarray(color).save(out_path)
            print("Saved:", out_path)
    finally:
        renderer.delete()

    if interactive:
        for i in range(batch_size):
            scene = build_frame_scene(
                vertice[i], faces, joints[i], eye_pose_np[i], landmark[i]
            )
            pyrender.Viewer(scene, use_raymond_lighting=True)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLAME demo (offscreen frames)")
    parser.add_argument("--interactive", action="store_true",
                        help="Open an interactive viewer per frame")
    raise SystemExit(main(interactive=parser.parse_args().interactive))

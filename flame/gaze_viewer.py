"""
Command-line FLAME gaze viewer.

Example:
python gaze_viewer.py --gaze-x 0 --gaze-y 0.2 --gaze-z 1
"""

import numpy as np
import torch
import trimesh
import pyrender

from FLAME import FLAME
from config import parser


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("Gaze direction norm is zero. Please provide a non-zero vector.")
    return v / n


def rotation_matrix_to_axis_angle(rot):
    trace = np.trace(rot)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)

    if np.pi - theta < 1e-5:
        # 180-degree case: pick axis from diagonal terms.
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


def forward_to_axis_angle(target_dir):
    src = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dst = normalize(target_dir.astype(np.float64))

    cross = np.cross(src, dst)
    dot = np.dot(src, dst)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0.0:
            return np.zeros(3, dtype=np.float32)
        # Opposite direction: rotate 180 degrees around X axis.
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


def make_cylinder_between(p0, p1, radius=0.003, sections=16, color=None):
    if color is None:
        color = [0.1, 0.3, 0.9, 1.0]

    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-8:
        return None

    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    cyl.visual.vertex_colors = np.tile(np.array(color, dtype=np.float32), (cyl.vertices.shape[0], 1))

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = vec / length
    v = np.cross(z_axis, axis)
    c = np.dot(z_axis, axis)

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
    transform[:3, 3] = p0 + 0.5 * vec
    cyl.apply_transform(transform)
    return cyl


def estimate_eye_centers(landmarks, joints):
    lm_xs = landmarks[:, 0]
    lm_ys = landmarks[:, 1]

    median_x = np.median(lm_xs)
    median_y = np.median(lm_ys)
    face_scale = max(np.ptp(lm_xs), np.ptp(lm_ys), 1e-4)
    y_tol = face_scale * 0.25

    left_idx = np.where((landmarks[:, 0] < median_x) & (np.abs(landmarks[:, 1] - median_y) < y_tol))[0]
    right_idx = np.where((landmarks[:, 0] >= median_x) & (np.abs(landmarks[:, 1] - median_y) < y_tol))[0]

    left_centroid = landmarks[left_idx].mean(axis=0) if left_idx.size > 0 else landmarks[landmarks[:, 0] < median_x].mean(axis=0)
    right_centroid = landmarks[right_idx].mean(axis=0) if right_idx.size > 0 else landmarks[landmarks[:, 0] >= median_x].mean(axis=0)

    left_joint_idx = np.argmin(np.linalg.norm(joints - left_centroid[None, :], axis=1))
    right_joint_idx = np.argmin(np.linalg.norm(joints - right_centroid[None, :], axis=1))

    return joints[left_joint_idx], joints[right_joint_idx], face_scale


def build_scene(vertices, faces, left_center, right_center, gaze_dir, face_scale):
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])

    head_colors = np.ones([vertices.shape[0], 4]) * [0.83, 0.72, 0.62, 1.0]
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=head_colors)
    scene.add(pyrender.Mesh.from_trimesh(tri_mesh, smooth=False))

    gaze_len = face_scale * 0.6
    sphere_radius = max(face_scale * 0.03, 0.006)
    line_radius = max(face_scale * 0.005, 0.0025)

    left_end = left_center + gaze_dir * gaze_len
    right_end = right_center + gaze_dir * gaze_len

    left_sphere = trimesh.creation.uv_sphere(radius=sphere_radius)
    left_sphere.apply_translation(left_center)
    left_sphere.visual.vertex_colors = [0.1, 0.3, 0.9, 1.0]

    right_sphere = trimesh.creation.uv_sphere(radius=sphere_radius)
    right_sphere.apply_translation(right_center)
    right_sphere.visual.vertex_colors = [0.1, 0.3, 0.9, 1.0]

    scene.add(pyrender.Mesh.from_trimesh(left_sphere))
    scene.add(pyrender.Mesh.from_trimesh(right_sphere))

    left_line = make_cylinder_between(left_center, left_end, radius=line_radius)
    right_line = make_cylinder_between(right_center, right_end, radius=line_radius)
    if left_line is not None:
        scene.add(pyrender.Mesh.from_trimesh(left_line))
    if right_line is not None:
        scene.add(pyrender.Mesh.from_trimesh(right_line))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    return scene


def main():
    parser.description = "FLAME gaze direction viewer"
    parser.add_argument("--gaze-x", type=float, required=True, help="Gaze direction X component")
    parser.add_argument("--gaze-y", type=float, required=True, help="Gaze direction Y component")
    parser.add_argument("--gaze-z", type=float, required=True, help="Gaze direction Z component")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Run device")

    config = parser.parse_args()
    config.batch_size = 1

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    print("Using device:", device)

    flame = FLAME(config).to(device)

    shape_params = torch.zeros(1, config.shape_params, dtype=torch.float32, device=device)
    expression_params = torch.zeros(1, config.expression_params, dtype=torch.float32, device=device)
    pose_params = torch.zeros(1, config.pose_params, dtype=torch.float32, device=device)

    target_gaze = np.array([config.gaze_x, config.gaze_y, config.gaze_z], dtype=np.float32)
    target_gaze = normalize(target_gaze)

    eye_axis_angle = forward_to_axis_angle(target_gaze)
    eye_pose_np = np.concatenate([eye_axis_angle, eye_axis_angle], axis=0).reshape(1, 6)
    eye_pose = torch.tensor(eye_pose_np, dtype=torch.float32, device=device)

    neck_pose = torch.zeros(1, 3, dtype=torch.float32, device=device) if config.optimize_neckpose else None

    vertices, landmarks, joints = flame(shape_params, expression_params, pose_params, neck_pose, eye_pose)

    vertices_np = vertices[0].detach().cpu().numpy()
    landmarks_np = landmarks[0].detach().cpu().numpy()
    joints_np = joints[0].detach().cpu().numpy()

    left_center, right_center, face_scale = estimate_eye_centers(landmarks_np, joints_np)

    # Keep vectors facing the camera for easier inspection in default pose.
    camera_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if np.dot(target_gaze, (camera_pos - left_center)) < 0:
        target_gaze = -target_gaze

    scene = build_scene(vertices_np, flame.faces, left_center, right_center, target_gaze, face_scale)

    print("Input gaze direction (normalized):", target_gaze)
    print("Eye axis-angle used:", eye_axis_angle)
    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    main()

"""
Demo code to load the FLAME Layer and visualise the 3D landmarks on the Face 

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""

import numpy as np
import torch
from FLAME import FLAME
import pyrender
import trimesh
import os
from PIL import Image
from config import get_config

config = get_config()
radian = np.pi/180.0
flamelayer = FLAME(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Creating a batch of mean shapes
shape_params = torch.zeros(8, 100, device=device)

# Batch size
batch_size = shape_params.shape[0]

# Create a batch of zero poses so the head remains unrotated
# pose_params[:, :3] are global rotation, pose_params[:, 3:] are jaw rotation
pose_params = torch.zeros(batch_size, config.pose_params, dtype=torch.float32, device=device)

# Cerating a batch of neutral expressions
expression_params = torch.zeros(8, 50, dtype=torch.float32, device=device)
flamelayer = flamelayer.to(device)

# Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework 
vertice, landmark, joints = flamelayer(shape_params, expression_params, pose_params) # For RingNet project
print(vertice.size(), landmark.size(), joints.size())

if config.optimize_eyeballpose and config.optimize_neckpose:
    neck_pose = torch.zeros(batch_size, 3, device=device)
    eye_pose = torch.zeros(batch_size, 6, device=device)
    vertice, landmark, joints = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

# Visualize Landmarks and render with varying eye poses
# We create different `eye_pose` values, pass them to FLAME, then
# use an offscreen renderer to save PNGs instead of popping GUI windows.
faces = flamelayer.faces
out_dir = 'outputs'
os.makedirs(out_dir, exist_ok=True)

# Create varying eye poses (axis-angle vectors). Here we vary yaw for both eyes.
batch_size = shape_params.shape[0]
eye_yaws = np.linspace(-20.0, 20.0, batch_size) * radian
eye_pose_np = np.zeros((batch_size, 6), dtype=np.float32)
for i in range(batch_size):
    yaw = eye_yaws[i]
    # axis-angle: [ax, ay, az] (rotate around Y for yaw)
    eye_pose_np[i, 0:3] = np.array([0.0, yaw, 0.0], dtype=np.float32)  # left eye
    eye_pose_np[i, 3:6] = np.array([0.0, yaw, 0.0], dtype=np.float32)  # right eye

eye_pose = torch.tensor(eye_pose_np, dtype=torch.float32, device=device)

# Forward pass with eye_pose to get mesh deformations including eyeball rotations
vertice, landmark, joints = flamelayer(shape_params, expression_params, pose_params, None, eye_pose)

# Offscreen renderer
renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
for i in range(batch_size):
    vertices = vertice[i].detach().cpu().numpy().squeeze()
    joints_arr = joints[i].detach().cpu().numpy().squeeze()
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.8, 0.7, 0.6, 1.0]

    tri_mesh = trimesh.Trimesh(vertices, faces,
                                vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
    scene.add(mesh)

    # Add landmark spheres
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(joints_arr), 1, 1))
    tfs[:, :3, 3] = joints_arr
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)

    # Add simple camera and light
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 1.0]])
    scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    color, depth = renderer.render(scene)
    img = Image.fromarray(color)
    img.save(os.path.join(out_dir, f'frame_{i:02d}.png'))

renderer.delete()
def axis_angle_to_dir(axis_angle):
    # axis_angle: (3,) axis-angle vector, returns rotated forward vector (0,0,1)
    v = axis_angle
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axis = v / theta
    forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Rodrigues rotation: v_rot = f*cos + (axis x f)*sin + axis*(axis·f)*(1-cos)
    cross = np.cross(axis, forward)
    dot = np.dot(axis, forward)
    v_rot = forward * cos_t + cross * sin_t + axis * dot * (1 - cos_t)
    return v_rot

def make_cylinder_between(p0, p1, radius=0.002, sections=16, color=[0.0, 0.0, 1.0, 1.0]):
    # Create a cylinder mesh between points p0 and p1
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    cyl.visual.vertex_colors = np.tile(np.array(color, dtype=np.float32), (cyl.vertices.shape[0], 1))
    # align cylinder along z, then compute transform
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = vec / length
    # compute rotation matrix that aligns z_axis to axis
    v = np.cross(z_axis, axis)
    c = np.dot(z_axis, axis)
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx.dot(vx) * ((1.0) / (1.0 + c))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p0 + vec * 0.5
    cyl.apply_transform(T)
    return cyl

# Show each frame interactively, only eye centers and gaze vectors
for i in range(batch_size):
    vertices = vertice[i].detach().cpu().numpy().squeeze()
    joints_positions = joints[i].detach().cpu().numpy().squeeze()

    # Estimate left/right eye centroids from landmarks, then find nearest joint as eye center
    lm = landmark[i].detach().cpu().numpy()
    jts = joints[i].detach().cpu().numpy()
    # split landmarks left/right by median x
    lm_xs = lm[:, 0]
    median_x = np.median(lm_xs)
    ys = lm[:, 1]
    median_y = np.median(ys)
    face_scale = max(np.ptp(lm_xs), np.ptp(ys))
    y_tol = face_scale * 0.25
    left_lm_idxs = np.where((lm[:, 0] < median_x) & (np.abs(lm[:, 1] - median_y) < y_tol))[0]
    right_lm_idxs = np.where((lm[:, 0] >= median_x) & (np.abs(lm[:, 1] - median_y) < y_tol))[0]
    if left_lm_idxs.size == 0:
        left_centroid = lm[lm[:, 0] < median_x].mean(axis=0)
    else:
        left_centroid = lm[left_lm_idxs].mean(axis=0)
    if right_lm_idxs.size == 0:
        right_centroid = lm[lm[:, 0] >= median_x].mean(axis=0)
    else:
        right_centroid = lm[right_lm_idxs].mean(axis=0)

    # find nearest joint to each centroid
    left_j_idx = np.argmin(np.linalg.norm(jts - left_centroid[None, :], axis=1))
    right_j_idx = np.argmin(np.linalg.norm(jts - right_centroid[None, :], axis=1))
    left_center = jts[left_j_idx]
    right_center = jts[right_j_idx]

    # compute gaze directions from eye_pose_np (axis-angle) used earlier
    left_axisang = eye_pose_np[i, 0:3]
    right_axisang = eye_pose_np[i, 3:6]
    left_dir = axis_angle_to_dir(left_axisang)
    right_dir = axis_angle_to_dir(right_axisang)

    # Ensure gaze points outward (not into the head) by checking against camera
    camera_pos = np.array([0.0, 0.0, 1.0])
    if np.dot(left_dir, (camera_pos - left_center)) < 0:
        left_dir = -left_dir
    if np.dot(right_dir, (camera_pos - right_center)) < 0:
        right_dir = -right_dir

    # build scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=np.ones([vertices.shape[0], 4]) * [0.8, 0.7, 0.6, 1.0])
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
    scene.add(mesh)

    # add eye center spheres (offset slightly along gaze so they sit on surface)
    offset = face_scale * 0.02
    left_center_vis = left_center + left_dir * offset
    right_center_vis = right_center + right_dir * offset
    sphere_radius = max(face_scale * 0.03, 0.007)
    l_s = trimesh.creation.uv_sphere(radius=sphere_radius)
    l_s.apply_translation(left_center_vis)
    l_s.visual.vertex_colors = [0.0, 0.0, 1.0, 1.0]
    r_s = trimesh.creation.uv_sphere(radius=sphere_radius)
    r_s.apply_translation(right_center_vis)
    r_s.visual.vertex_colors = [0.0, 0.0, 1.0, 1.0]
    scene.add(pyrender.Mesh.from_trimesh(l_s))
    scene.add(pyrender.Mesh.from_trimesh(r_s))

    # add gaze lines (cylinders)
    gaze_len = face_scale * 0.5
    l_end = left_center_vis + left_dir * gaze_len
    r_end = right_center_vis + right_dir * gaze_len
    l_cyl = make_cylinder_between(left_center_vis, l_end, radius=max(face_scale*0.005,0.003), color=[0.0, 0.0, 1.0, 1.0])
    r_cyl = make_cylinder_between(right_center_vis, r_end, radius=max(face_scale*0.005,0.003), color=[0.0, 0.0, 1.0, 1.0])
    if l_cyl is not None:
        scene.add(pyrender.Mesh.from_trimesh(l_cyl))
    if r_cyl is not None:
        scene.add(pyrender.Mesh.from_trimesh(r_cyl))

    # camera + light
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 1.0]])
    scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    # show interactive window per frame; close to proceed to next
    pyrender.Viewer(scene, use_raymond_lighting=True)
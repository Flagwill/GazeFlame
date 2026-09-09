"""Face detection and ETH-XGaze-style face normalization for GazeTR input.

GazeTR was trained on the ETH-XGaze *normalized* 224x224 face crops: the face
is frontalized (roll removed), the eye line is horizontal, and the scale is
fixed by a virtual camera.  Naively resizing an arbitrary photo to 224x224
therefore pushes the input far out of the training distribution.

This module replicates the official ETH-XGaze inference pipeline
(``demo.py``): detect the largest face with dlib, extract the same 6 facial
landmarks (4 eye corners + 2 nose alae), estimate the head pose with
``cv2.solvePnP`` against the ETH-XGaze generic 3D face model, and warp the
face into the canonical 224x224 crop with ``cv2.warpPerspective``.

The dlib shape predictor model is stored under ``common/models/`` (download it
with ``common/download_dlib_model.py`` if missing).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
PREDICTOR_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
PREDICTOR_BZ2_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat.bz2"
DLIB_MODEL_URL = (
    "https://github.com/davisking/dlib-models/raw/master/"
    "shape_predictor_68_face_landmarks.dat.bz2"
)

# ---------------------------------------------------------------------------
# ETH-XGaze 6-point landmark setup
# ---------------------------------------------------------------------------
# dlib 68-landmark indices selected by the official ETH-XGaze demo.py:
#   [right-eye-outer, right-eye-inner, left-eye-outer, left-eye-inner,
#    nose-left, nose-right]
ETH_LANDMARK_INDICES = np.array([36, 39, 42, 45, 31, 35], dtype=np.intp)

# 3D coordinates (mm) of the 6 landmarks above, i.e. rows
# [20, 23, 26, 29, 15, 19] of the generic ETH-XGaze 50-point face model.
FACE_MODEL_3D = np.array(
    [
        [-46.50949859619140625, -38.32709503173828125, 36.4160003662109375],
        [-17.76072120666503906, -32.58519744873046875, 29.0761566162109375],
        [18.04391098022460938, -30.95682334899902344, 29.06296730041503906],
        [44.73758697509765625, -34.10787200927734375, 36.7324371337890625],
        [-10.61166477203369141, 12.95834922790527344, 21.62276458740234375],
        [11.28962993621826172, 13.86424446105957031, 21.83790016174316406],
    ],
    dtype=np.float64,
)

# Normalized camera parameters used by the ETH-XGaze face normalization.
FOCAL_NORM = 960.0  # focal length of the virtual camera
DISTANCE_NORM = 600.0  # normalized distance between face and virtual camera
ROI_SIZE = 224  # cropped face patch size

# Assumed field of view for the unknown source camera (degrees).  Used to
# build a camera matrix for solvePnP when no calibration is available.
DEFAULT_FOV_DEG = 40.0

# dlib's HOG detector cost grows with image area; for large frames (webcam,
# big photos) we first downscale so the long edge is at most this many pixels.
# 384 px keeps small faces detectable while being ~13x faster than 640x480.
DETECT_MAX_DIM = 384


class NoFaceFoundError(RuntimeError):
    """Raised when dlib cannot find any face in the input image."""


# ---------------------------------------------------------------------------
# dlib singleton handles (loaded lazily so a missing model gives a clear error)
# ---------------------------------------------------------------------------
_detector = None
_predictor = None


def _ensure_dlib_ready() -> tuple:
    """Return ``(detector, predictor)``, loading dlib + the shape model once."""
    global _detector, _predictor
    if _detector is None:
        try:
            import dlib
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "dlib is required for face alignment. Install it with:\n"
                "  conda install -n gaze_flame -c conda-forge dlib --override-channels"
            ) from exc

        if not PREDICTOR_PATH.exists():
            raise FileNotFoundError(
                f"缺少 dlib 68点模型: {PREDICTOR_PATH}\n"
                f"请先运行 python -m common.download_dlib_model 下载。"
            )
        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    return _detector, _predictor


def detector_and_predictor():
    """Public accessor for tests / reuse; mirrors ``_ensure_dlib_ready``."""
    return _ensure_dlib_ready()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_largest_face(image_bgr: np.ndarray):
    """Return the dlib rectangle of the largest detected face (or None).

    Large images are downscaled before detection (see ``DETECT_MAX_DIM``) so
    the HOG scan stays cheap; the returned rectangle is mapped back to the
    original coordinates.  Small images are retried at 2x so small faces are
    more likely to be found.
    """
    detector, _ = _ensure_dlib_ready()
    h, w = image_bgr.shape[:2]
    long_edge = max(h, w)

    scale = 1.0
    detect_img = image_bgr
    if long_edge > DETECT_MAX_DIM:
        scale = DETECT_MAX_DIM / float(long_edge)
        detect_img = cv2.resize(
            image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    rects = list(detector(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB), 1))

    if not rects and scale == 1.0 and long_edge * 2 <= 4000:
        scale = 2.0
        detect_img = cv2.resize(
            image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        rects = list(detector(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB), 1))

    if not rects:
        return None

    def area(r):
        return (r.right() - r.left()) * (r.bottom() - r.top())

    best = max(rects, key=area)
    if scale != 1.0:
        best = _scale_rect(best, 1.0 / scale)
    return best


def _scale_rect(rect, scale: float):
    import dlib

    return dlib.rectangle(
        int(rect.left() * scale),
        int(rect.top() * scale),
        int(rect.right() * scale),
        int(rect.bottom() * scale),
    )


def landmarks_to_array(shape) -> np.ndarray:
    """Convert a dlib full_object_detection into an (68, 2) numpy array."""
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)


# ---------------------------------------------------------------------------
# Head pose estimation + ETH-XGaze normalization
# ---------------------------------------------------------------------------
def estimate_head_pose(
    landmarks6: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | None = None,
):
    """solvePnP of the 6 landmarks against the ETH-XGaze 3D face model.

    Mirrors the official ETH-XGaze ``estimateHeadPose``: EPNP first, then an
    iterative refinement initialized with that solution.
    """
    if distortion is None:
        distortion = np.zeros((4, 1), dtype=np.float64)
    face_pts = FACE_MODEL_3D.reshape(6, 1, 3).astype(np.float64)
    lm_pts = landmarks6.reshape(6, 1, 2).astype(np.float64)

    ret, rvec, tvec = cv2.solvePnP(
        face_pts, lm_pts, camera_matrix, distortion, flags=cv2.SOLVEPNP_EPNP
    )
    if not ret:
        raise RuntimeError("solvePnP(EPNP) failed for the given landmarks")
    ret, rvec, tvec = cv2.solvePnP(
        face_pts, lm_pts, camera_matrix, distortion, rvec, tvec, True
    )
    if not ret:
        raise RuntimeError("solvePnP(iterative) failed for the given landmarks")
    return rvec, tvec


def assumed_camera_matrix(image: np.ndarray, fov_deg: float = DEFAULT_FOV_DEG) -> np.ndarray:
    """Intrinsics for the (unknown) source camera, assuming ``fov_deg``.

    The normalization is fairly robust to the exact assumed FOV because the
    virtual-camera scaling re-normalizes the face size anyway.
    """
    h, w = image.shape[:2]
    focal = float(w) / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array(
        [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def normalize_face_image(
    image_bgr: np.ndarray,
    landmarks6: np.ndarray,
    camera_matrix: np.ndarray,
    roi_size: int = ROI_SIZE,
    focal_norm: float = FOCAL_NORM,
    distance_norm: float = DISTANCE_NORM,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a detected face into the ETH-XGaze normalized 224x224 crop.

    Direct port of the official ``normalizeData_face`` from the ETH-XGaze
    repo (``demo.py``).  Returns ``(warped_bgr, warped_landmarks6)``.
    """
    landmarks6 = landmarks6.astype(np.float64)
    hr, ht = estimate_head_pose(landmarks6, camera_matrix)
    hR = cv2.Rodrigues(hr)[0]
    ht = ht.reshape((3, 1))

    Fc = np.dot(hR, FACE_MODEL_3D.T) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(
        np.concatenate((two_eye_center, nose_center), axis=1), axis=1
    ).reshape((3, 1))

    distance = np.linalg.norm(face_center)
    if distance < 1e-6:
        raise RuntimeError("face center distance is (near) zero")

    z_scale = distance_norm / distance
    cam_norm = np.array(
        [
            [focal_norm, 0.0, roi_size / 2.0],
            [0.0, focal_norm, roi_size / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    S = np.diag([1.0, 1.0, z_scale])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down_norm = np.linalg.norm(down)
    if down_norm < 1e-8:
        raise RuntimeError("cannot build a stable frontalization basis")
    down = down / down_norm
    right = np.cross(down, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise RuntimeError("cannot build a stable frontalization basis")
    right = right / right_norm
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))

    img_warped = cv2.warpPerspective(image_bgr, W, (roi_size, roi_size))

    lm = landmarks6.reshape(6, 1, 2)
    warped_lm = cv2.perspectiveTransform(lm, W).reshape(6, 2)

    return img_warped, warped_lm


# ---------------------------------------------------------------------------
# Public high-level API
# ---------------------------------------------------------------------------
def align_face_to_gaze_crop(
    image_bgr: np.ndarray,
    fov_deg: float = DEFAULT_FOV_DEG,
):
    """Detect the largest face and return an ETH-XGaze normalized 224 crop.

    Returns ``(crop_bgr, info)`` where ``crop_bgr`` is the aligned 224x224
    BGR image and ``info`` is a dict with ``face_count`` (0 if none detected)
    and, when a face was found, the warped 6 landmarks used.

    If no face is found, ``crop_bgr`` is ``None`` and callers should fall back
    to the naive full-image resize.
    """
    rect = detect_largest_face(image_bgr)
    if rect is None:
        return None, {"face_count": 0}

    _, predictor = _ensure_dlib_ready()
    shape = predictor(image_bgr, rect)
    landmarks68 = landmarks_to_array(shape)
    landmarks6 = landmarks68[ETH_LANDMARK_INDICES]

    camera_matrix = assumed_camera_matrix(image_bgr, fov_deg)
    crop, warped_lm = normalize_face_image(image_bgr, landmarks6, camera_matrix)

    info = {
        "face_count": 1,
        "rect": [rect.left(), rect.top(), rect.right(), rect.bottom()],
        "landmarks6": landmarks6,
        "warped_landmarks6": warped_lm,
    }
    return crop, info


def draw_aligned_overlay(crop_bgr: np.ndarray, warped_landmarks6: np.ndarray) -> np.ndarray:
    """Annotate the aligned crop with the 6 warped landmarks (debug helper)."""
    out = crop_bgr.copy()
    for (x, y) in warped_landmarks6.astype(int):
        cv2.circle(out, (int(x), int(y)), 4, (0, 200, 0), -1)
    return out


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test ETH-XGaze face alignment")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--out", default="aligned_face.png", help="output crop path")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"无法读取图片: {args.image}", file=sys.stderr)
        sys.exit(1)

    crop, info = align_face_to_gaze_crop(img)
    if crop is None:
        print(f"未检测到人脸: {args.image}", file=sys.stderr)
        sys.exit(1)

    overlay = draw_aligned_overlay(crop, info["warped_landmarks6"])
    cv2.imwrite(args.out, overlay)
    print(f"已保存对齐结果 ({overlay.shape[1]}x{overlay.shape[0]}): {args.out}")

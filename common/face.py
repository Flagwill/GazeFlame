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


class FaceTracker:
    """Bridge brief detector misses while keeping the face crop current.

    dlib's HOG detector is reliable but a single frame with motion blur or a
    partial occlusion can return no rectangle.  A correlation tracker is much
    cheaper than another full-frame HOG scan and is sufficient to carry the
    landmark predictor across those short gaps.  Periodic detector refreshes
    correct tracker drift and allow a new face to be acquired after tracking
    is lost.
    """

    def __init__(
        self,
        detect_interval: int = 6,
        max_tracking_frames: int = 30,
        min_tracking_quality: float = 0.0,
    ):
        self.detect_interval = max(1, int(detect_interval))
        self.max_tracking_frames = max(1, int(max_tracking_frames))
        self.min_tracking_quality = float(min_tracking_quality)
        self._tracker = None
        self._frames_since_detection = self.detect_interval
        self._tracking_frames = 0

    def reset(self) -> None:
        """Discard the current track so the next frame forces detection."""
        self._tracker = None
        self._frames_since_detection = self.detect_interval
        self._tracking_frames = 0

    @staticmethod
    def _valid_rect(position, width: int, height: int):
        """Convert a dlib floating rectangle to a clipped integer rectangle."""
        import dlib

        left = max(0, min(width - 1, int(round(position.left()))))
        top = max(0, min(height - 1, int(round(position.top()))))
        right = max(0, min(width - 1, int(round(position.right()))))
        bottom = max(0, min(height - 1, int(round(position.bottom()))))
        if right - left < 20 or bottom - top < 20:
            return None
        return dlib.rectangle(left, top, right, bottom)

    def _start(self, image_rgb: np.ndarray, rect) -> bool:
        try:
            import dlib

            tracker = dlib.correlation_tracker()
            tracker.start_track(image_rgb, rect)
            self._tracker = tracker
            self._tracking_frames = 0
            return True
        except Exception:
            # Detection remains usable if a particular dlib build does not
            # support correlation tracking for the current image type.
            self._tracker = None
            self._tracking_frames = 0
            return False

    def locate(self, image_bgr: np.ndarray):
        """Return ``(rect, metadata)`` for the current frame.

        ``metadata["source"]`` is ``"detected"`` for a fresh HOG result and
        ``"tracked"`` when the result bridges a detector miss.
        """
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tracked_rect = None
        tracking_quality = None

        if self._tracker is not None:
            try:
                tracking_quality = float(self._tracker.update(image_rgb))
                tracked_rect = self._valid_rect(
                    self._tracker.get_position(), w, h
                )
                self._tracking_frames += 1
            except Exception:
                self.reset()

        detector_due = (
            self._tracker is None
            or self._frames_since_detection >= self.detect_interval
            or self._tracking_frames > self.max_tracking_frames
            or (
                tracking_quality is not None
                and tracking_quality < self.min_tracking_quality
            )
        )
        if detector_due:
            detected_rect = detect_largest_face(image_bgr)
            self._frames_since_detection = 0
            if detected_rect is not None:
                self._start(image_rgb, detected_rect)
                return detected_rect, {
                    "source": "detected",
                    "tracking_quality": tracking_quality,
                }

        if (
            tracked_rect is not None
            and self._tracking_frames <= self.max_tracking_frames
            and (
                tracking_quality is None
                or tracking_quality >= self.min_tracking_quality
            )
        ):
            self._frames_since_detection += 1
            return tracked_rect, {
                "source": "tracked",
                "tracking_quality": tracking_quality,
            }

        self.reset()
        return None, {"source": "none", "tracking_quality": tracking_quality}


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


def head_pose_to_flame_rotation(rvec: np.ndarray) -> np.ndarray:
    """Map an OpenCV face rotation into FLAME's y-up coordinate system.

    ``solvePnP`` uses an image-style y-down camera frame while FLAME uses y-up
    model coordinates.  Flipping y on both sides preserves the rotation's
    handedness and keeps the neutral pose equal to the identity matrix.
    """
    rotation_cv, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    axis_flip = np.diag([1.0, -1.0, 1.0])
    return axis_flip @ rotation_cv @ axis_flip


def head_rotation_to_euler(rotation: np.ndarray) -> np.ndarray:
    """Return XYZ Euler angles in radians for display/debugging."""
    rotation = np.asarray(rotation, dtype=np.float64)
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    singular = sy < 1e-6
    if not singular:
        angles = np.array(
            [
                np.arctan2(rotation[2, 1], rotation[2, 2]),
                np.arctan2(-rotation[2, 0], sy),
                np.arctan2(rotation[1, 0], rotation[0, 0]),
            ]
        )
    else:
        angles = np.array(
            [
                np.arctan2(-rotation[1, 2], rotation[1, 1]),
                np.arctan2(-rotation[2, 0], sy),
                0.0,
            ]
        )
    return angles


def eye_aspect_ratio(landmarks68: np.ndarray) -> float:
    """Return the mean eye aspect ratio from dlib's 68 landmarks.

    EAR is high when the eye is open and drops during a blink.  It is a
    deliberately model-free signal, useful for controlling the lightweight
    FLAME eye geometry without introducing another neural network.
    """
    points = np.asarray(landmarks68, dtype=np.float64)
    if points.shape[0] < 48:
        raise ValueError("68 facial landmarks are required for eye_aspect_ratio")

    def _single(indices: tuple[int, ...]) -> float:
        eye = points[list(indices)]
        width = np.linalg.norm(eye[0] - eye[3])
        if width < 1e-8:
            return 0.0
        return float(
            (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4]))
            / (2.0 * width)
        )

    return 0.5 * (_single((36, 37, 38, 39, 40, 41)) +
                  _single((42, 43, 44, 45, 46, 47)))


def _reprojection_error(
    landmarks6: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        FACE_MODEL_3D,
        rvec,
        tvec,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
    )
    return float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - landmarks6, axis=1)))


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
    head_pose: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a detected face into the ETH-XGaze normalized 224x224 crop.

    Direct port of the official ``normalizeData_face`` from the ETH-XGaze
    repo (``demo.py``).  Returns ``(warped_bgr, warped_landmarks6)``.
    """
    landmarks6 = landmarks6.astype(np.float64)
    hr, ht = head_pose if head_pose is not None else estimate_head_pose(
        landmarks6, camera_matrix
    )
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
    face_tracker: FaceTracker | None = None,
):
    """Detect the largest face and return an ETH-XGaze normalized 224 crop.

    Returns ``(crop_bgr, info)`` where ``crop_bgr`` is the aligned 224x224
    BGR image and ``info`` is a dict with ``face_count`` (0 if none detected)
    and, when a face was found, the warped 6 landmarks used.

    If no face is found, ``crop_bgr`` is ``None`` and callers should fall back
    to the naive full-image resize.
    """
    tracking_info = {"source": "detected", "tracking_quality": None}
    if face_tracker is None:
        rect = detect_largest_face(image_bgr)
    else:
        rect, tracking_info = face_tracker.locate(image_bgr)
    if rect is None:
        return None, {"face_count": 0, **tracking_info}

    _, predictor = _ensure_dlib_ready()
    shape = predictor(image_bgr, rect)
    landmarks68 = landmarks_to_array(shape)
    landmarks6 = landmarks68[ETH_LANDMARK_INDICES]

    camera_matrix = assumed_camera_matrix(image_bgr, fov_deg)
    rvec, tvec = estimate_head_pose(landmarks6, camera_matrix)
    crop, warped_lm = normalize_face_image(
        image_bgr,
        landmarks6,
        camera_matrix,
        head_pose=(rvec, tvec),
    )
    head_rotation = head_pose_to_flame_rotation(rvec)
    head_pose, _ = cv2.Rodrigues(head_rotation)

    info = {
        "face_count": 1,
        "rect": [rect.left(), rect.top(), rect.right(), rect.bottom()],
        **tracking_info,
        "landmarks68": landmarks68,
        "landmarks6": landmarks6,
        "warped_landmarks6": warped_lm,
        "rvec": rvec.reshape(3),
        "tvec": tvec.reshape(3),
        "camera_matrix": camera_matrix,
        "head_rotation": head_rotation,
        "head_pose": head_pose.reshape(3),
        "head_euler": head_rotation_to_euler(head_rotation),
        "reprojection_error": _reprojection_error(
            landmarks6, rvec, tvec, camera_matrix
        ),
        "eye_aspect_ratio": eye_aspect_ratio(landmarks68),
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

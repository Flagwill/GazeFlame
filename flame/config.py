"""FLAME model configuration.

This module defines the hyper-parameters consumed by :class:`FLAME` as a
plain dataclass (``FLAMEConfig``) and helpers to build it from argparse.

It deliberately avoids a module-level ``parser`` object: importing this
module must not have side effects, and callers (e.g. ``gaze_viewer.py``)
build their own ``argparse.ArgumentParser`` and merge these options into it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

FLAME_DEFAULT_MODEL_DIR = "./model"


@dataclass
class FLAMEConfig:
    """Hyper-parameters for the FLAME model and its forward pass."""

    flame_model_path: str = f"{FLAME_DEFAULT_MODEL_DIR}/generic_model.pkl"
    static_landmark_embedding_path: str = (
        f"{FLAME_DEFAULT_MODEL_DIR}/flame_static_embedding.pkl"
    )
    dynamic_landmark_embedding_path: str = (
        f"{FLAME_DEFAULT_MODEL_DIR}/flame_dynamic_embedding.npy"
    )

    # FLAME hyper-parameters
    shape_params: int = 100
    expression_params: int = 50
    pose_params: int = 6

    # Forward-pass flags
    use_face_contour: bool = True
    use_3D_translation: bool = True
    optimize_eyeballpose: bool = True
    optimize_neckpose: bool = True
    batch_size: int = 8

    # Training-only (kept for upstream compatibility)
    num_worker: int = 4
    ring_margin: float = 0.5
    ring_loss_weight: float = 1.0


def add_flame_args(parser: argparse.ArgumentParser) -> None:
    """Add FLAME model options to an existing argparse parser."""
    parser.add_argument("--flame_model_path", type=str, default=FLAMEConfig.flame_model_path,
                        help="FLAME model path")
    parser.add_argument("--static_landmark_embedding_path", type=str,
                        default=FLAMEConfig.static_landmark_embedding_path,
                        help="Static landmark embeddings path for FLAME")
    parser.add_argument("--dynamic_landmark_embedding_path", type=str,
                        default=FLAMEConfig.dynamic_landmark_embedding_path,
                        help="Dynamic contour embedding path for FLAME")

    parser.add_argument("--shape_params", type=int, default=FLAMEConfig.shape_params,
                        help="The number of shape parameters")
    parser.add_argument("--expression_params", type=int,
                        default=FLAMEConfig.expression_params,
                        help="The number of expression parameters")
    parser.add_argument("--pose_params", type=int, default=FLAMEConfig.pose_params,
                        help="The number of pose parameters")

    parser.add_argument("--use_face_contour", action="store_true",
                        default=FLAMEConfig.use_face_contour,
                        help="Apply the landmark loss on the face contour too")
    parser.add_argument("--use_3D_translation", action="store_true",
                        default=FLAMEConfig.use_3D_translation,
                        help="Apply landmark loss on the face contour too")
    parser.add_argument("--optimize_eyeballpose", action="store_true",
                        default=FLAMEConfig.optimize_eyeballpose,
                        help="If true optimize for the eyeball pose")
    parser.add_argument("--optimize_neckpose", action="store_true",
                        default=FLAMEConfig.optimize_neckpose,
                        help="If true optimize for the neck pose")
    parser.add_argument("--batch_size", type=int, default=FLAMEConfig.batch_size,
                        help="Batch size")


def config_from_args(args: argparse.Namespace) -> FLAMEConfig:
    """Build a :class:`FLAMEConfig` from parsed CLI arguments."""
    return FLAMEConfig(
        flame_model_path=args.flame_model_path,
        static_landmark_embedding_path=args.static_landmark_embedding_path,
        dynamic_landmark_embedding_path=args.dynamic_landmark_embedding_path,
        shape_params=args.shape_params,
        expression_params=args.expression_params,
        pose_params=args.pose_params,
        use_face_contour=args.use_face_contour,
        use_3D_translation=args.use_3D_translation,
        optimize_eyeballpose=args.optimize_eyeballpose,
        optimize_neckpose=args.optimize_neckpose,
        batch_size=args.batch_size,
    )


def get_config(argv: list[str] | None = None) -> FLAMEConfig:
    """Backwards-compatible entry point: parse FLAME args and return a config.

    Provided so legacy scripts keep working; prefer building your own parser
    with :func:`add_flame_args` and :func:`config_from_args`.
    """
    parser = argparse.ArgumentParser(description="FLAME model")
    add_flame_args(parser)
    return config_from_args(parser.parse_args(argv))

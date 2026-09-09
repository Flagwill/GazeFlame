"""
GazeTR inference script.

Run a pretrained GazeTR model on a single image or a folder of images and
report gaze pitch/yaw in radians (and degrees).

Examples:
    # Single image
    python inference.py --model pretrain.pt --image test.jpg

    # A whole folder of images
    python inference.py --model pretrain.pt --image_folder ./test_images

    # Write machine-readable results (JSON, one entry per image)
    python inference.py --model pretrain.pt --image_folder ./test_images --output results.json

    # Force CPU execution
    python inference.py --model pretrain.pt --image test.jpg --device cpu

Notes:
    - Input images must be RGB, 224x224. Anything else is resized
      automatically (which may reduce accuracy).
    - The 3D / popup visualizations require matplotlib / tkinter+pillow and
      are only imported when requested, keeping the default path fast.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import model
from gtools import gazeto3d

# Supported input image extensions (upper/lower case).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------
# Optional (heavy) dependencies are imported lazily.
# --------------------------------------------------------------------------
def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "matplotlib is required for --draw_gaze. Install it with: pip install matplotlib"
        ) from exc


def _import_tk():
    try:
        import tkinter as tk
        from PIL import Image, ImageTk

        return tk, Image, ImageTk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "tkinter and pillow are required for --popup. Install pillow and enable tkinter."
        ) from exc


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------
def read_image_bgr(image_path) -> np.ndarray:
    """Read an image with cv2 (BGR order). Raises ValueError if unreadable."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return img_bgr


def preprocess_image(img_bgr: np.ndarray, device) -> torch.Tensor:
    """Resize to 224x224, BGR->RGB, normalize to [0, 1] and move to device."""
    if img_bgr.shape[:2] != (224, 224):
        img = cv2.resize(img_bgr, (224, 224))
    else:
        img = img_bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def collect_image_paths(image_path: str | None, image_folder: str | None) -> list[str]:
    """Return the sorted list of image paths for the given CLI inputs."""
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        return [image_path]

    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {image_folder}")

    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(folder.glob(f"*{ext}"))
        paths.extend(folder.glob(f"*{ext.upper()}"))
    paths = sorted(str(p) for p in paths)
    if not paths:
        raise FileNotFoundError(f"文件夹中没有找到图片: {image_folder}")
    return paths


# --------------------------------------------------------------------------
# Visualization helpers
# --------------------------------------------------------------------------
def save_3d_gaze_visualization(
    image_bgr: np.ndarray,
    pitch: float,
    yaw: float,
    save_path,
    line_width: int = 2,
) -> None:
    """Save a two-panel figure: the input image and the 3D gaze vector."""
    plt = _import_matplotlib()
    gaze_vec = gazeto3d(np.array([pitch, yaw]))

    fig = plt.figure(figsize=(10, 4.5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.quiver(
        0, 0, 0, gaze_vec[0], gaze_vec[1], gaze_vec[2],
        color="red", linewidth=line_width, arrow_length_ratio=0.15,
    )
    axis_len = 1.0
    ax2.quiver(0, 0, 0, axis_len, 0, 0, color="green", linewidth=1)
    ax2.quiver(0, 0, 0, 0, axis_len, 0, color="blue", linewidth=1)
    ax2.quiver(0, 0, 0, 0, 0, axis_len, color="black", linewidth=1)
    ax2.text(1.05, 0, 0, "X", color="green")
    ax2.text(0, 1.05, 0, "Y", color="blue")
    ax2.text(0, 0, 1.05, "Z", color="black")
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_title("3D Gaze Vector")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=20, azim=-55)

    fig.suptitle(
        f"Pitch: {np.degrees(pitch):.2f} deg, Yaw: {np.degrees(yaw):.2f} deg"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def show_result_popup(image_bgr: np.ndarray, result: dict) -> None:
    """Block until the user closes a small tkinter window with the result."""
    tk, Image, ImageTk = _import_tk()

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    max_w, max_h = 800, 600
    w, h = pil_img.size
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        pil_img = pil_img.resize(new_size, Image.ANTIALIAS)

    root = tk.Tk()
    root.title(f"Result - {Path(result['image']).name}")

    photo = ImageTk.PhotoImage(pil_img)
    label_img = tk.Label(root, image=photo)
    label_img.image = photo
    label_img.pack()

    text = (
        f"Pitch: {result['pitch_deg']:.2f}° ({result['pitch']:.4f} rad)\n"
        f"Yaw:   {result['yaw_deg']:.2f}° ({result['yaw']:.4f} rad)"
    )
    lbl = tk.Label(root, text=text, font=("Arial", 12))
    lbl.pack(pady=6)

    btn = tk.Button(root, text="Close", command=root.destroy)
    btn.pack(pady=4)

    root.mainloop()


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def inference(
    net,
    image_paths: list[str],
    device: str = "cpu",
    output_file: str | None = None,
    batch_size: int = 16,
    draw_gaze: bool = False,
    vis_dir: str = "vis_results",
    line_thickness: int = 2,
    popup: bool = False,
) -> tuple[list[dict], int]:
    """Run GazeTR over ``image_paths``.

    Args:
        net: Loaded GazeTR model.
        image_paths: Image paths to process.
        device: ``cpu`` or ``cuda``.
        output_file: Optional results file. ``.json`` writes one JSON object,
            anything else writes a human-readable text table.
        batch_size: Number of images forwarded at once.
        draw_gaze: Whether to save 3D gaze visualizations.
        vis_dir: Directory for visualizations.
        line_thickness: Line width used by the 3D visualization.
        popup: Whether to show a blocking per-image popup window.

    Returns:
        ``(results, saved_vis_count)``.
    """
    net.eval()
    results: list[dict] = []
    saved_vis_count = 0

    print(f"\n开始推理，共 {len(image_paths)} 张图片 (batch_size={batch_size})...")
    print("-" * 70)

    if draw_gaze:
        vis_dir = os.path.abspath(vis_dir)
        os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        processed = 0
        for start in range(0, len(image_paths), batch_size):
            batch = image_paths[start : start + batch_size]
            loaded = []
            for image_path in batch:
                try:
                    loaded.append((image_path, read_image_bgr(image_path)))
                except Exception as exc:
                    print(f"[{image_path}] 加载失败: {exc}")

            if not loaded:
                continue

            inputs = torch.cat(
                [preprocess_image(bgr, device) for _, bgr in loaded], dim=0
            )
            gaze_pred = net({"face": inputs}).cpu().numpy()

            for (image_path, bgr), gaze in zip(loaded, gaze_pred):
                processed += 1
                pitch, yaw = float(gaze[0]), float(gaze[1])
                result = {
                    "image": image_path,
                    "pitch": pitch,
                    "yaw": yaw,
                    "pitch_deg": float(np.degrees(pitch)),
                    "yaw_deg": float(np.degrees(yaw)),
                }
                results.append(result)

                print(f"[{processed}/{len(image_paths)}] {Path(image_path).name}")
                print(
                    f"  Pitch: {result['pitch']:.4f} rad ({result['pitch_deg']:.2f}°)"
                )
                print(f"  Yaw:   {result['yaw']:.4f} rad ({result['yaw_deg']:.2f}°)")

                if draw_gaze:
                    try:
                        save_path = os.path.join(
                            vis_dir, Path(image_path).stem + "_gaze3d.png"
                        )
                        save_3d_gaze_visualization(
                            bgr, result["pitch"], result["yaw"],
                            save_path, line_width=line_thickness,
                        )
                        saved_vis_count += 1
                        print(f"  3D可视化: {save_path}")
                    except Exception as vis_error:
                        print(f"  3D可视化失败: {vis_error}")
                if popup:
                    try:
                        show_result_popup(bgr, result)
                    except Exception as popup_error:
                        print(f"  弹窗显示失败: {popup_error}")

    print("-" * 70)
    print(f"推理完成！成功处理 {len(results)}/{len(image_paths)} 张图片\n")

    if output_file:
        _write_results(output_file, results)

    if draw_gaze:
        print(f"可视化图片保存目录: {vis_dir}")

    return results, saved_vis_count


def _write_results(output_file: str, results: list[dict]) -> None:
    """Write results as JSON (``.json``) or a plain text table."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("image_name pitch(rad) yaw(rad) pitch(deg) yaw(deg)\n")
            for r in results:
                f.write(
                    f"{Path(r['image']).name} {r['pitch']:.6f} {r['yaw']:.6f} "
                    f"{r['pitch_deg']:.2f} {r['yaw_deg']:.2f}\n"
                )
    print(f"结果已保存到: {output_file}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GazeTR Inference - Gaze Estimation with Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python inference.py --model pretrain.pt --image test.jpg
  python inference.py --model pretrain.pt --image_folder ./test_images --output results.json
  python inference.py --model pretrain.pt --image test.jpg --device cpu
  python inference.py --model pretrain.pt --image test.jpg --draw_gaze --vis_dir vis_results
        """,
    )

    parser.add_argument(
        "--model", "-m", type=str, required=True, help="预训练模型路径 (.pt文件)"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", "-i", type=str, help="单张图片路径")
    input_group.add_argument("--image_folder", "-f", type=str, help="包含多张图片的文件夹路径")

    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出结果文件路径 (.json 输出 JSON; 其他后缀输出纯文本表格)",
    )
    parser.add_argument(
        "--draw_gaze", action="store_true",
        help="输出三维视线可视化图（左图原图，右图3D gaze向量）",
    )
    parser.add_argument(
        "--vis_dir", type=str, default="vis_results",
        help="可视化结果保存目录 (默认: vis_results)",
    )
    parser.add_argument(
        "--line_thickness", type=int, default=2, help="3D箭头线宽 (默认: 2)"
    )
    parser.add_argument(
        "--popup", action="store_true",
        help="每张图弹出 tkinter 窗口显示结果 (阻塞, 默认关闭)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="每次前向推理的图片数量 (默认: 16)",
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda", choices=["cuda", "cpu"],
        help="使用的设备 (默认: cuda; 无 GPU 时自动回退 cpu)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU模式")
        args.device = "cpu"
    device = torch.device(args.device)

    print("=" * 70)
    print("GazeTR Inference - Gaze Estimation with Transformer")
    print("=" * 70)

    print(f"\n加载模型: {args.model}")
    net = model.Model()
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(device)
    print(f"✓ 模型加载成功 (设备: {args.device})")

    try:
        image_paths = collect_image_paths(args.image, args.image_folder)
    except FileNotFoundError as exc:
        print(f"错误: {exc}")
        return 1

    results, saved_vis_count = inference(
        net,
        image_paths,
        device=args.device,
        output_file=args.output,
        batch_size=args.batch_size,
        draw_gaze=args.draw_gaze,
        vis_dir=args.vis_dir,
        line_thickness=args.line_thickness,
        popup=args.popup,
    )

    if args.draw_gaze and saved_vis_count == 0:
        print("错误: 未保存任何3D可视化图片，请检查上面的失败信息。")
        return 2

    print("\n推理任务完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
GazeTR Inference Script
使用预训练模型对单张或多张图片进行gaze estimation推理

使用方法:
    # 单张图片
    python inference.py --model pretrain.pt --image test.jpg
    
    # 文件夹中的所有图片
    python inference.py --model pretrain.pt --image_folder ./test_images
    
    # 指定输出文件
    python inference.py --model pretrain.pt --image test.jpg --output results.txt
    
    # 使用CPU
    python inference.py --model pretrain.pt --image test.jpg --device cpu

注意: 输入图片必须是224x224大小的RGB图像
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path

MATPLOTLIB_IMPORT_ERROR = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    MATPLOTLIB_IMPORT_ERROR = e

# tkinter / PIL (用于弹窗显示)
TK_IMPORT_ERROR = None
try:
    import tkinter as tk
    from PIL import Image, ImageTk
except Exception as e:
    tk = None
    Image = None
    ImageTk = None
    TK_IMPORT_ERROR = e

# 导入模型
import model


def load_image(image_path, device='cuda'):
    """
    加载并预处理图片
    
    Args:
        image_path: 图片路径
        device: 'cuda' 或 'cpu'
    
    Returns:
        处理后的tensor和原始BGR图片
    """
    # 读取图片
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 为模型推理准备输入副本
    img = img_bgr.copy()
    
    # 检查尺寸
    if img.shape[:2] != (224, 224):
        print(f"警告: 图片 {image_path} 尺寸为 {img.shape[:2]}, 将自动resize到 224x224")
        img = cv2.resize(img, (224, 224))
    
    # BGR转RGB (cv2读取的是BGR格式)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换为tensor: [H, W, C] -> [C, H, W], 归一化到[0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    
    # 添加batch维度: [C, H, W] -> [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0)
    
    # 移到指定设备
    if device == 'cuda':
        img_tensor = img_tensor.cuda()
    
    return img_tensor, img_bgr


def save_3d_gaze_visualization(image_bgr, pitch, yaw, save_path, line_width=2):
    """
    保存三维视线可视化：左侧原图，右侧3D坐标系中的视线向量。

    Args:
        image_bgr: 原始BGR图像
        pitch: 俯仰角（弧度）
        yaw: 偏航角（弧度）
        save_path: 保存路径
        line_width: 3D箭头线宽
    """
    if plt is None:
        raise RuntimeError("未安装matplotlib，无法生成3D可视化。请先执行: pip install matplotlib")

    gaze_vec = gazeto3d(np.array([pitch, yaw]))

    fig = plt.figure(figsize=(10, 4.5))

    # 左侧显示原图
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Input Image")
    ax1.axis("off")

    # 右侧显示3D视线向量
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.quiver(0, 0, 0, gaze_vec[0], gaze_vec[1], gaze_vec[2],
               color='red', linewidth=line_width, arrow_length_ratio=0.15)

    # 绘制标准坐标轴，便于观察方向
    axis_len = 1.0
    ax2.quiver(0, 0, 0, axis_len, 0, 0, color='green', linewidth=1)
    ax2.quiver(0, 0, 0, 0, axis_len, 0, color='blue', linewidth=1)
    ax2.quiver(0, 0, 0, 0, 0, axis_len, color='black', linewidth=1)

    ax2.text(1.05, 0, 0, 'X', color='green')
    ax2.text(0, 1.05, 0, 'Y', color='blue')
    ax2.text(0, 0, 1.05, 'Z', color='black')

    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_title("3D Gaze Vector")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=20, azim=-55)

    fig.suptitle(f"Pitch: {np.degrees(pitch):.2f} deg, Yaw: {np.degrees(yaw):.2f} deg")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def show_result_popup(image_bgr, result):
    """
    使用tkinter弹窗显示推理结果（含图片和角度信息）
    """
    if tk is None or Image is None or ImageTk is None:
        raise RuntimeError("未安装tkinter或Pillow，无法显示弹窗。请确保环境中有 tkinter 和 pillow。")

    # BGR -> RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 限制弹窗中图片最大尺寸
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

    text = (f"Pitch: {result['pitch_deg']:.2f}° ({result['pitch']:.4f} rad)\n"
            f"Yaw:   {result['yaw_deg']:.2f}° ({result['yaw']:.4f} rad)")
    lbl = tk.Label(root, text=text, font=("Arial", 12))
    lbl.pack(pady=6)

    btn = tk.Button(root, text="Close", command=root.destroy)
    btn.pack(pady=4)

    # 阻塞直到用户关闭弹窗
    root.mainloop()


def gazeto3d(gaze):
    """
    将2D gaze (pitch, yaw) 转换为3D向量
    
    Args:
        gaze: [pitch, yaw] in radians
    
    Returns:
        3D gaze vector [x, y, z]
    """
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def angular_error(gaze1, gaze2):
    """
    计算两个gaze向量之间的角度误差
    
    Args:
        gaze1, gaze2: 2D gaze [pitch, yaw] in radians
    
    Returns:
        角度误差（度）
    """
    vec1 = gazeto3d(gaze1)
    vec2 = gazeto3d(gaze2)
    
    # 计算夹角
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def inference(
    model,
    image_paths,
    device='cuda',
    output_file=None,
    draw_gaze=False,
    vis_dir='vis_results',
    line_thickness=2,
    popup=False,
):
    """
    对图片进行推理
    
    Args:
        model: 加载的GazeTR模型
        image_paths: 图片路径列表
        device: 'cuda' 或 'cpu'
        output_file: 输出文件路径（可选）
        draw_gaze: 是否绘制三维视线可视化
        vis_dir: 可视化结果目录
        line_thickness: 3D箭头线宽
    """
    model.eval()
    
    results = []
    
    print(f"\n开始推理，共 {len(image_paths)} 张图片...")
    print("-" * 70)
    
    saved_vis_count = 0
    failed_vis_count = 0

    if draw_gaze:
        vis_dir = os.path.abspath(vis_dir)
        os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            try:
                # 加载图片
                img_tensor, img_original = load_image(image_path, device)
                
                # 推理
                data = {'face': img_tensor}
                gaze_pred = model(data)
                
                # 转换为numpy
                gaze_pred = gaze_pred.cpu().numpy()[0]  # [pitch, yaw]
                
                # 存储结果
                result = {
                    'image': image_path,
                    'pitch': float(gaze_pred[0]),
                    'yaw': float(gaze_pred[1]),
                    'pitch_deg': float(np.degrees(gaze_pred[0])),
                    'yaw_deg': float(np.degrees(gaze_pred[1]))
                }
                results.append(result)
                
                # 打印结果
                print(f"[{i+1}/{len(image_paths)}] {Path(image_path).name}")
                print(f"  Pitch: {result['pitch']:.4f} rad ({result['pitch_deg']:.2f}°)")
                print(f"  Yaw:   {result['yaw']:.4f} rad ({result['yaw_deg']:.2f}°)")

                if draw_gaze:
                    try:
                        save_name = Path(image_path).stem + '_gaze3d.png'
                        save_path = os.path.join(vis_dir, save_name)
                        save_3d_gaze_visualization(
                            img_original,
                            result['pitch'],
                            result['yaw'],
                            save_path,
                            line_width=line_thickness,
                        )
                        saved_vis_count += 1
                        print(f"  3D可视化: {save_path}")
                    except Exception as vis_error:
                        failed_vis_count += 1
                        print(f"  3D可视化失败: {vis_error}")
                # 弹窗显示结果（如果启用）
                if popup:
                    try:
                        show_result_popup(img_original, result)
                    except Exception as popup_err:
                        print(f"  弹窗显示失败: {popup_err}")
                
            except Exception as e:
                print(f"[{i+1}/{len(image_paths)}] 错误: {image_path}")
                print(f"  {str(e)}")
                continue
    
    print("-" * 70)
    print(f"推理完成！成功处理 {len(results)}/{len(image_paths)} 张图片\n")
    
    # 保存结果
    if output_file:
        with open(output_file, 'w') as f:
            f.write("image_name pitch(rad) yaw(rad) pitch(deg) yaw(deg)\n")
            for r in results:
                f.write(f"{Path(r['image']).name} {r['pitch']:.6f} {r['yaw']:.6f} "
                       f"{r['pitch_deg']:.2f} {r['yaw_deg']:.2f}\n")
        print(f"结果已保存到: {output_file}")

    if draw_gaze:
        print(f"可视化图片保存目录: {vis_dir}")
        print(f"可视化保存统计: 成功 {saved_vis_count} / 失败 {failed_vis_count}")
    
    return results, saved_vis_count


def main():
    parser = argparse.ArgumentParser(
        description='GazeTR Inference - Gaze Estimation with Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python inference.py --model pretrain.pt --image test.jpg
  python inference.py --model pretrain.pt --image_folder ./test_images --output results.txt
  python inference.py --model pretrain.pt --image test.jpg --device cpu
    python inference.py --model pretrain.pt --image test.jpg --draw_gaze --vis_dir vis_results
        """
    )
    
    # 模型参数
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='预训练模型路径 (.pt文件)')
    
    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', type=str,
                            help='单张图片路径')
    input_group.add_argument('--image_folder', '-f', type=str,
                            help='包含多张图片的文件夹路径')
    
    # 输出参数
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出结果文件路径 (默认: 仅打印到屏幕)')

    parser.add_argument('--draw_gaze', action='store_true',
                       help='输出三维视线可视化图（左图原图，右图3D gaze向量）')

    parser.add_argument('--vis_dir', type=str, default='vis_results',
                       help='可视化结果保存目录 (默认: vis_results)')

    parser.add_argument('--line_thickness', type=int, default=2,
                       help='3D箭头线宽 (默认: 2)')
    
    # 弹窗显示：默认启用（不再需要命令行参数）

    # 设备参数
    parser.add_argument('--device', '-d', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='使用的设备 (默认: cuda)')
    
    args = parser.parse_args()

    if args.draw_gaze and plt is None:
        print("错误: 当前环境无法导入 matplotlib，无法进行3D可视化。")
        if MATPLOTLIB_IMPORT_ERROR is not None:
            print(f"详情: {MATPLOTLIB_IMPORT_ERROR}")
        print("请执行: pip install matplotlib")
        sys.exit(1)

    # 弹窗所需依赖检测（默认启用弹窗）
    if (tk is None or Image is None or ImageTk is None):
        print("错误: 当前环境不支持弹窗显示（缺少 tkinter 或 pillow）。")
        if TK_IMPORT_ERROR is not None:
            print(f"详情: {TK_IMPORT_ERROR}")
        print("请确保已安装 pillow 并启用 tkinter（Windows 通常自带 tkinter）。")
        sys.exit(1)
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU模式")
        args.device = 'cpu'
    
    print("=" * 70)
    print("GazeTR Inference - Gaze Estimation with Transformer")
    print("=" * 70)
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    try:
        net = model.Model()
        
        if args.device == 'cuda':
            net.cuda()
            state_dict = torch.load(args.model)
        else:
            # CPU模式下加载模型
            state_dict = torch.load(args.model, map_location='cpu')
            # 修改model.py中的硬编码.cuda()问题
            # 这里我们需要临时修改forward中的position创建
        
        net.load_state_dict(state_dict)
        print(f"✓ 模型加载成功 (设备: {args.device})")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 收集图片路径
    image_paths = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"错误: 图片不存在 - {args.image}")
            return
        image_paths = [args.image]
    else:
        folder = Path(args.image_folder)
        if not folder.exists():
            print(f"错误: 文件夹不存在 - {args.image_folder}")
            return
        
        # 支持的图片格式
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        for ext in extensions:
            image_paths.extend(folder.glob(f'*{ext}'))
            image_paths.extend(folder.glob(f'*{ext.upper()}'))
        
        image_paths = sorted([str(p) for p in image_paths])
        
        if len(image_paths) == 0:
            print(f"错误: 文件夹中没有找到图片 - {args.image_folder}")
            return
    
    # 执行推理
    results, saved_vis_count = inference(
        net,
        image_paths,
        args.device,
        args.output,
        draw_gaze=args.draw_gaze,
        vis_dir=args.vis_dir,
        line_thickness=args.line_thickness,
        popup=True,
    )

    if args.draw_gaze and saved_vis_count == 0:
        print("错误: 未保存任何3D可视化图片，请检查上面的失败信息。")
        sys.exit(2)
    
    print("\n推理任务完成！")


if __name__ == "__main__":
    main()

# GazeTR 推理使用指南

## 快速开始

### 1. 准备预训练模型

下载预训练模型（ETH-XGaze数据集训练）：
- [Google Drive](https://drive.google.com/file/d/1WEiKZ8Ga0foNmxM7xFabI4D5ajThWAWj/view?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1GEbjbNgXvVkisVWGtTJm7g) 提取码：`1234`

将下载的模型保存为 `pretrain.pt`

### 2. 准备测试图片

**重要要求：**
- 图片必须是 **224×224** 大小
- RGB 彩色图像（3通道）
- 建议使用 `.jpg` 或 `.png` 格式

如果图片不是224×224，脚本会自动resize，但可能影响精度。

### 3. 运行推理

#### 单张图片推理

```bash
python inference.py --model pretrain.pt --image test.jpg
```

#### 批量推理（文件夹）

```bash
python inference.py --model pretrain.pt --image_folder ./test_images
```

#### 保存结果到文件

```bash
python inference.py --model pretrain.pt --image_folder ./test_images --output results.txt
```

#### 使用CPU（无GPU环境）

```bash
python inference.py --model pretrain.pt --image test.jpg --device cpu
```

## 输出格式

### 屏幕输出示例

```
======================================================================
GazeTR Inference - Gaze Estimation with Transformer
======================================================================

加载模型: pretrain.pt
✓ 模型加载成功 (设备: cuda)

开始推理，共 3 张图片...
----------------------------------------------------------------------
[1/3] person1_001.jpg
  Pitch: -0.1234 rad (-7.07°)
  Yaw:   0.0567 rad (3.25°)
[2/3] person1_002.jpg
  Pitch: 0.2341 rad (13.42°)
  Yaw:   -0.1456 rad (-8.34°)
[3/3] person2_001.jpg
  Pitch: 0.0123 rad (0.70°)
  Yaw:   0.0987 rad (5.65°)
----------------------------------------------------------------------
推理完成！成功处理 3/3 张图片

推理任务完成！
```

### 结果文件格式 (results.txt)

```
image_name pitch(rad) yaw(rad) pitch(deg) yaw(deg)
person1_001.jpg -0.123400 0.056700 -7.07 3.25
person1_002.jpg 0.234100 -0.145600 13.42 -8.34
person2_001.jpg 0.012300 0.098700 0.70 5.65
```

## Gaze 参数说明

- **Pitch**（俯仰角）：
  - 正值：向上看
  - 负值：向下看
  
- **Yaw**（偏航角）：
  - 正值：向右看
  - 负值：向左看

- 单位：
  - `rad`：弧度制
  - `deg`：角度制

## 完整参数列表

```
python inference.py -h

参数说明:
  --model, -m       预训练模型路径 (.pt文件) [必需]
  --image, -i       单张图片路径 [与--image_folder二选一]
  --image_folder, -f 图片文件夹路径 [与--image二选一]
  --output, -o      输出结果文件路径 [可选]
  --device, -d      使用的设备: cuda/cpu [默认: cuda]
```

## 常见问题

### Q1: 图片尺寸不是224×224怎么办？

脚本会自动resize，但建议预先处理好图片以获得最佳效果：

```python
import cv2
img = cv2.imread('input.jpg')
img_resized = cv2.resize(img, (224, 224))
cv2.imwrite('input_224.jpg', img_resized)
```

### Q2: CUDA out of memory 错误

切换到CPU模式：
```bash
python inference.py --model pretrain.pt --image test.jpg --device cpu
```

### Q3: 推理结果不准确

可能原因：
1. 图片预处理不正确（未使用GazeHub工具）
2. 图片中头部/眼睛未对齐
3. 预训练模型的数据分布与测试数据差异较大

建议使用 [GazeHub](http://phi-ai.org/GazeHub/) 的数据处理流程预处理图片。

## 进阶使用：在Python代码中调用

```python
import torch
import cv2
from model import Model

# 加载模型
net = Model()
net.cuda()
net.load_state_dict(torch.load('pretrain.pt'))
net.eval()

# 加载图片 (224x224)
img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
img = img.unsqueeze(0).cuda()

# 推理
with torch.no_grad():
    data = {'face': img}
    gaze = net(data)
    
pitch, yaw = gaze[0].cpu().numpy()
print(f"Pitch: {pitch:.4f} rad, Yaw: {yaw:.4f} rad")
```

## 修改记录

- 修改了 [model.py](model.py) Line 114，将 `.cuda()` 改为 `.to(feature.device)`，支持CPU推理
- 创建了 [inference.py](inference.py) 独立推理脚本

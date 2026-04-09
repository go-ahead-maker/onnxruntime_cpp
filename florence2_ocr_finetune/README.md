# Florence-2 OCR 微调框架

一个灵活、可扩展的 Florence-2 OCR 微调框架，支持自定义 Vision Encoder 和细粒度的组件冻结控制。

## 主要特性

### 1. 支持修改 Vision Encoder ✓

框架支持多种 Vision Encoder，可以替换 Florence-2 默认的 DaViT：

- **DaViT** (默认): `davit_base`, `davit_tiny`, `davit_small`
- **ViT**: `vit_base`, `vit_large`, `vit_huge`, `vit_small`
- **Swin Transformer**: `swin_tiny`, `swin_small`, `swin_base`, `swin_large`
- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **自定义 Encoder**: 从 Python 文件、模块或 HuggingFace 加载

#### 预训练权重加载

所有 encoder 都支持两种预训练权重加载方式：
1. **自动下载**: 使用 timm 库自动下载预训练权重
2. **自定义路径**: 指定本地预训练权重文件路径

```yaml
model:
  vision_encoder:
    type: "vit"
    name: "vit_base"
    pretrained: true
    # 方式 1: 自动下载 (pretrained_path: null)
    pretrained_path: null
    # 方式 2: 使用本地权重文件
    # pretrained_path: "/path/to/vit_base_pretrained.pth"
    image_size: 384
```

#### 自定义 Vision Encoder 🆕

支持三种方式加载自定义 encoder：

1. **Python 文件**: `custom_path: "./my_encoder.py"`
2. **模块路径**: `custom_path: "my_package.encoders.MyEncoder"`
3. **HuggingFace**: `custom_path: "facebook/dino-vitb16"`

详见：[自定义 Encoder 使用指南](docs/custom_encoder_guide.md)

```yaml
model:
  vision_encoder:
    type: "custom"
    custom_path: "./examples/custom_vision_encoder.py"
    pretrained: true
    pretrained_path: "./weights/my_encoder.pth"  # 可选
    image_size: 384
```

### 2. 细粒度组件冻结控制 ✓

支持独立冻结模型的三个主要部分：

```yaml
model:
  freeze_components:
    vision_encoder: false      # 冻结 Vision Encoder
    image_projection: false    # 冻结 Image Projection 层
    language_model: false      # 冻结 Language Model (Text Decoder)
```

**冻结策略示例：**

| 场景 | vision_encoder | image_projection | language_model | 适用情况 |
|------|---------------|------------------|----------------|----------|
| 全量微调 | false | false | false | 充足数据，追求最佳效果 |
| 仅训练投影层 | true | false | true | 小数据集，快速适配 |
| 仅训练 LM | true | true | false | 视觉特征已足够好 |
| 仅训练 VE | false | true | true | 特定领域视觉特征 |
| 冻结全部 | true | true | true | 仅推理/测试 |

### 3. 清晰的数据集构建 ✓

支持三种 OCR 数据格式：

```python
# 1. Simple 格式 (JSON)
{
  "annotations": [
    {"image_path": "img1.jpg", "text": "Hello World"},
    {"image_path": "img2.jpg", "text": "OCR Text"}
  ]
}

# 2. COCO 格式
{
  "images": [...],
  "annotations": [...]
}

# 3. ICDAR 格式 (XML)
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础训练

```bash
python train.py --config config/training_config.yaml
```

### 使用不同 Vision Encoder

```bash
# 使用 ViT 作为 encoder
python train.py --config config/training_config.yaml \
  --override model.vision_encoder.type=vit \
  --override model.vision_encoder.name=vit_base

# 使用 Swin Transformer
python train.py --config config/training_config.yaml \
  --override model.vision_encoder.type=swin \
  --override model.vision_encoder.name=swin_base
```

### 冻结组件训练

```bash
# 仅训练 projection 层 (few-shot learning)
python train.py --config config/training_config.yaml \
  --override model.freeze_components.vision_encoder=true \
  --override model.freeze_components.language_model=true

# 冻结 vision encoder，微调其他部分
python train.py --config config/training_config.yaml \
  --override model.freeze_components.vision_encoder=true
```

### 使用自定义预训练权重

```bash
python train.py --config config/training_config.yaml \
  --override model.vision_encoder.pretrained_path=/path/to/weights.pth
```

## 配置文件说明

### 完整配置示例

```yaml
model:
  base_model: "microsoft/Florence-2-base"
  
  vision_encoder:
    type: "davit"           # encoder 类型
    name: "davit_base"      # 具体模型名称
    pretrained: true        # 是否加载预训练权重
    pretrained_path: null   # 自定义权重路径 (可选)
    image_size: 384         # 输入图像尺寸
    custom_path: null       # 自定义 encoder 模块路径
  
  freeze_components:
    vision_encoder: false
    image_projection: false
    language_model: false
  
  text_decoder:
    pretrained: true
    max_length: 256

data:
  train_data:
    image_dir: "./data/images"
    annotation_file: "./data/annotations_train.json"
  val_data:
    image_dir: "./data/images"
    annotation_file: "./data/annotations_val.json"
  
  transforms:
    image_size: 384
    augmentations:
      random_resize_crop: true
      color_jitter: false
      random_flip: false
  
  dataloader:
    batch_size: 8
    num_workers: 4
    shuffle: true

training:
  optimizer:
    type: "AdamW"
    lr: 1e-5
    weight_decay: 0.01
  scheduler:
    type: "cosine"
    warmup_ratio: 0.1
  epochs: 20
  grad_clip: 1.0
  fp16: true
  eval_steps: 500
  save_steps: 1000

output:
  output_dir: "./outputs"
  experiment_name: "florence2_ocr_finetune"
```

## 项目结构

```
florence2_ocr_finetune/
├── config/
│   └── training_config.yaml    # 训练配置
├── models/
│   ├── __init__.py
│   ├── vision_encoders.py      # Vision Encoder 工厂
│   └── florence2_wrapper.py    # Florence-2 包装器
├── data/
│   ├── __init__.py
│   ├── dataset.py              # 数据集构建
│   └── transforms.py           # 数据增强
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── inference.py                # 推理脚本
├── requirements.txt            # 依赖列表
└── README.md                   # 本文档
```

## 评估指标

框架内置 OCR 专用评估指标：
- **精确匹配率 (Exact Match)**: 预测文本与真实文本完全匹配的比例
- **编辑距离 (Edit Distance)**: Levenshtein 距离
- **字符准确率 (Character Accuracy)**: 正确字符比例

```bash
python evaluate.py --config config/training_config.yaml \
  --checkpoint ./outputs/experiment/checkpoint_best.pt
```

## 推理示例

```python
from models.florence2_wrapper import create_florence2_model
from PIL import Image
import torch

# 加载模型
model = create_florence2_model(
    base_model="microsoft/Florence-2-base",
    vision_encoder_config={"type": "davit", "name": "davit_base"},
    freeze_vision_encoder=False
)
model.eval()
model.cuda()

# 加载图像
image = Image.open("test_image.jpg").convert("RGB")

# OCR 推理
with torch.no_grad():
    result = model.generate(
        pixel_values=image,
        task_prompt="<OCR>",
        max_length=256
    )

print(f"OCR Result: {result[0]}")
```

## 常见问题

### Q: 如何选择合适的 Vision Encoder？

- **精度优先**: 使用 `davit_base` 或 `vit_large`
- **速度优先**: 使用 `resnet50` 或 `vit_small`
- **平衡**: 使用 `swin_base` 或 `davit_small`

### Q: 小数据集应该如何设置冻结策略？

建议冻结 vision encoder，只训练 projection 和 language model：
```yaml
freeze_components:
  vision_encoder: true
  image_projection: false
  language_model: false
```

### Q: 如何使用自己的 Vision Encoder？

1. 创建自定义 encoder 模块（参考 `models/vision_encoders.py`）
2. 在配置中指定：
```yaml
vision_encoder:
  type: "custom"
  custom_path: "/path/to/custom_encoder.py"
```

## License

MIT License

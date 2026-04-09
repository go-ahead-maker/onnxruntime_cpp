# Florence-2 OCR Fine-tuning Framework

这是一个用于微调 Florence-2 模型进行 OCR 任务的框架，支持自定义 vision encoder。

## 项目结构

```
florence2_ocr_finetune/
├── config/
│   └── training_config.yaml      # 训练配置文件
├── data/
│   ├── dataset.py                # 数据集构建模块
│   └── transforms.py             # 数据预处理和增强
├── models/
│   ├── vision_encoders.py        # 自定义 vision encoder
│   └── florence2_wrapper.py      # Florence-2 模型包装器
├── train.py                      # 训练脚本
├── evaluate.py                   # 评估脚本
├── inference.py                  # 推理脚本
├── requirements.txt              # 依赖包
└── README.md                     # 使用说明
```

## 主要特性

1. **支持多种 Vision Encoder**: 可以替换默认的 DaViT encoder 为其他编码器（如 ViT, Swin Transformer, ResNet 等）
2. **清晰的 dataset 构建**: 模块化设计，易于扩展和定制
3. **专注于 OCR 任务**: 针对文本识别任务优化的训练流程
4. **灵活的配置**: 通过 YAML 配置文件管理所有超参数

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备数据

将你的 OCR 数据集组织成以下格式：
```
data/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── annotations.json
```

annotations.json 格式：
```json
[
    {
        "image_path": "images/img1.jpg",
        "text": "Hello World"
    },
    {
        "image_path": "images/img2.jpg",
        "text": "OCR Test"
    }
]
```

### 修改配置

编辑 `config/training_config.yaml` 文件，设置：
- 数据集路径
- vision encoder 类型
- 训练超参数
- 输出目录

### 开始训练

```bash
python train.py --config config/training_config.yaml
```

### 推理测试

```bash
python inference.py --checkpoint path/to/checkpoint --image path/to/image.jpg
```

## 自定义 Vision Encoder

在 `config/training_config.yaml` 中设置 `vision_encoder` 参数：

```yaml
model:
  vision_encoder: "vit"  # 可选: davit(默认), vit, swin, resnet
  encoder_name: "vit_base_patch16_224"  # 具体编码器名称
```

或在代码中使用自定义 encoder：

```python
from models.vision_encoders import CustomVisionEncoder

encoder = CustomVisionEncoder(
    encoder_type="vit",
    pretrained=True,
    image_size=384
)
```

## 支持的 OCR 任务

- 文本检测与识别
- 场景文本识别
- 文档 OCR
- 手写文本识别

## 许可证

MIT License

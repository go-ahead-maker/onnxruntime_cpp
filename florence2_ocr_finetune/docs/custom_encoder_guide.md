# Florence-2 OCR 微调框架 - 自定义 Vision Encoder 使用指南

本指南详细介绍如何加载和使用自定义 Vision Encoder。

## 📋 目录

1. [支持的加载方式](#支持的加载方式)
2. [从 Python 文件加载](#从-python-文件加载)
3. [从模块路径加载](#从模块路径加载)
4. [从 HuggingFace 加载](#从-huggingface-加载)
5. [加载预训练权重](#加载预训练权重)
6. [配置示例](#配置示例)

---

## 支持的加载方式

框架支持三种方式加载自定义 Vision Encoder：

| 方式 | 描述 | 适用场景 |
|------|------|----------|
| **Python 文件** | 直接导入 `.py` 文件 | 本地自定义模型 |
| **模块路径** | 从已安装的包中导入类 | 第三方库或项目内模块 |
| **HuggingFace** | 从 HF Hub 加载模型 | 使用现有预训练模型 |

---

## 从 Python 文件加载

### 步骤 1: 创建自定义 Encoder 文件

创建 `my_custom_encoder.py`：

```python
import torch
import torch.nn as nn
from typing import Optional

def create_encoder(
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    image_size: int = 384,
    **kwargs
) -> nn.Module:
    """
    创建自定义 encoder
    
    框架会自动查找以下函数之一：
    - create_encoder (推荐)
    - build_encoder
    - get_encoder
    - create_model
    """
    # 你的自定义模型
    model = MyCustomModel(image_size=image_size, **kwargs)
    
    # 可选：加载预训练权重
    if pretrained and pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    return model


class MyCustomModel(nn.Module):
    def __init__(self, image_size=384, embed_dim=768, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        # ... 定义你的模型层
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """必须实现此方法，返回 [B, L, D] 格式的特征"""
        # ... 你的前向传播逻辑
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
```

### 步骤 2: 修改配置文件

在 `training_config.yaml` 中：

```yaml
model:
  vision_encoder:
    type: "custom"
    custom_path: "./my_custom_encoder.py"  # 文件路径
    pretrained: true
    pretrained_path: "./weights/my_encoder.pth"  # 可选
    image_size: 384
```

---

## 从模块路径加载

如果你的 encoder 在一个 Python 包中：

### 步骤 1: 确保模块可导入

```bash
# 项目结构
my_project/
├── my_encoders/
│   ├── __init__.py
│   └── vit_encoder.py  # 包含 MyViTEncoder 类
```

### 步骤 2: 配置

```yaml
model:
  vision_encoder:
    type: "custom"
    custom_path: "my_encoders.vit_encoder.MyViTEncoder"  # 模块路径。类名
    image_size: 384
    embed_dim: 768  # 传递给类的参数
```

框架会自动实例化 `MyViTEncoder(image_size=384, embed_dim=768)`

---

## 从 HuggingFace 加载

直接使用 HF Hub 上的模型作为 encoder：

```yaml
model:
  vision_encoder:
    type: "custom"
    custom_path: "facebook/dino-vitb16"  # HF model_id
    pretrained: true
    image_size: 384
```

支持的模型示例：
- `facebook/dino-vitb16`
- `google/vit-base-patch16-224`
- `microsoft/beit-base-patch16-224`
- 任何返回图像特征的 HF 模型

---

## 加载预训练权重

### 方式 1: 自动处理（推荐）

在 `create_encoder` 函数中处理：

```python
def create_encoder(pretrained=True, pretrained_path=None, **kwargs):
    model = MyModel(**kwargs)
    
    if pretrained and pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 处理不同的 checkpoint 格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 过滤不需要的键
        filtered = {k.replace('module.', ''): v 
                   for k, v in state_dict.items() 
                   if 'head.' not in k}
        
        model.load_state_dict(filtered, strict=False)
    
    return model
```

### 方式 2: 框架自动处理

如果 `create_encoder` 不处理权重，框架会自动尝试加载：

```yaml
model:
  vision_encoder:
    type: "custom"
    custom_path: "./my_encoder.py"
    pretrained: true
    pretrained_path: "./weights/checkpoint.pth"  # 框架会自动加载
```

### 支持的权重格式

- `.pth` - PyTorch 格式
- `.pt` - PyTorch 格式
- `.bin` - PyTorch bin 格式
- `.safetensors` - SafeTensors 格式（需要安装 `safetensors`）

---

## 配置示例

### 示例 1: 使用本地自定义 CNN

```yaml
model:
  base_model: "microsoft/Florence-2-base"
  
  vision_encoder:
    type: "custom"
    custom_path: "./encoders/my_cnn.py"
    pretrained: true
    pretrained_path: "./weights/cnn_imagenet.pth"
    image_size: 384
  
  freeze_components:
    vision_encoder: false  # 微调 encoder
    image_projection: false
    language_model: true   # 冻结 LM
```

### 示例 2: 使用 HuggingFace DINOv2

```yaml
model:
  vision_encoder:
    type: "custom"
    custom_path: "facebook/dinov2-base"
    pretrained: true
    image_size: 518  # DINOv2 推荐尺寸
  
  freeze_components:
    vision_encoder: true  # 冻结预训练 encoder
    image_projection: false
    language_model: false
```

### 示例 3: Few-shot 学习设置

```yaml
model:
  vision_encoder:
    type: "custom"
    custom_path: "./my_encoder.py"
    pretrained: true
    pretrained_path: "./pretrained/encoder.pth"
  
  freeze_components:
    vision_encoder: true       # 冻结
    image_projection: true     # 冻结
    language_model: false      # 仅微调 LM
```

---

## 常见问题

### Q: 我的 encoder 需要什么接口？

A: 最少需要：
1. `forward_features(x)` 方法，返回 `[B, L, D]` 格式
2. `embed_dim` 属性（或通过框架自动推断）
3. `image_size` 属性

### Q: 如何处理多尺度特征？

A: 在 `forward_features` 中选择合适的特征层：

```python
def forward_features(self, x):
    features = self.model(x)
    
    # 如果是多尺度，选择最后一层
    if isinstance(features, (list, tuple)):
        features = features[-1]
    
    # 如果是 [B, C, H, W]，转换为 [B, L, D]
    if len(features.shape) == 4:
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)
    
    return features
```

### Q: 权重加载失败怎么办？

A: 检查：
1. 权重文件格式是否正确
2. 键名是否匹配（框架会自动移除 `module.` 等前缀）
3. 是否跳过了分类头（框架会自动跳过 `head.`, `classifier.` 等）

---

## 完整测试

运行示例代码测试你的 encoder：

```bash
cd /workspace/florence2_ocr_finetune
python examples/custom_vision_encoder.py
```

如果看到 `✓ Custom encoder test passed!` 说明 encoder 可以正常工作。

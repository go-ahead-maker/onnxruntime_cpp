"""
自定义 Vision Encoder 示例

本文件展示了如何创建自定义 Vision Encoder 以用于 Florence-2 OCR 微调框架。
提供三种实现方式：
1. 函数式：create_encoder() 函数
2. 类式：CustomVisionEncoder 类
3. 基于现有模型包装

使用方法：
    在 training_config.yaml 中设置:
    ```yaml
    vision_encoder:
      type: "custom"
      custom_path: "./my_custom_encoder.py"
      pretrained: true
      pretrained_path: "./weights/my_encoder.pth"  # 可选
      image_size: 384
    ```
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple


# ============================================================================
# 方式 1: 使用 create_encoder 函数 (推荐)
# ============================================================================

def create_encoder(
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    image_size: int = 384,
    **kwargs
) -> nn.Module:
    """
    创建自定义 Vision Encoder
    
    Args:
        pretrained: 是否加载预训练权重
        pretrained_path: 预训练权重路径
        image_size: 图像输入尺寸
        **kwargs: 其他参数
        
    Returns:
        nn.Module: 自定义 encoder
    """
    # 示例：创建一个简单的 CNN + Transformer 混合 encoder
    # 从 kwargs 中提取参数，避免重复
    encoder_kwargs = {
        'image_size': image_size,
        'embed_dim': kwargs.pop('embed_dim', 768),
        'depth': kwargs.pop('depth', 12),
        'num_heads': kwargs.pop('num_heads', 12),
    }
    encoder_kwargs.update(kwargs)
    
    encoder = CustomVisionEncoder(**encoder_kwargs)
    
    # 如果需要加载预训练权重
    if pretrained and pretrained_path is not None:
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 处理 checkpoint 格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 过滤和重命名键
        filtered_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith('module.'):
                new_k = new_k[7:]
            if new_k.startswith('encoder.'):
                new_k = new_k[8:]
            # 跳过分类头
            if 'head.' in new_k or 'classifier.' in new_k:
                continue
            filtered_state_dict[new_k] = v
        
        # 加载权重
        missing_keys, unexpected_keys = encoder.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
    
    return encoder


class CustomVisionEncoder(nn.Module):
    """
    自定义 Vision Encoder 示例
    
    这是一个简单的 CNN + Transformer 混合架构，
    你可以根据需要替换为任何自定义架构。
    """
    
    def __init__(
        self,
        image_size: int = 384,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
        **kwargs
    ):
        super().__init__()
        
        self.image_size = image_size
        self.embed_dim = embed_dim
        
        # Patch embedding (将图像分割为 patches)
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        print(f"CustomVisionEncoder created: embed_dim={embed_dim}, depth={depth}")
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            图像特征 [B, L, D]
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, L, D]
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Normalization
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.forward_features(x)


# ============================================================================
# 方式 2: 直接从模块导入类 (alternative)
# ============================================================================

class AlternativeCustomEncoder(nn.Module):
    """
    另一种自定义 Encoder 实现
    
    如果配置文件中指定了类名，框架会尝试直接实例化这个类。
    类名需要包含 'encoder' 或 'vision' 关键字。
    """
    
    def __init__(
        self,
        image_size: int = 384,
        embed_dim: int = 512,
        **kwargs
    ):
        super().__init__()
        
        self.image_size = image_size
        self.embed_dim = embed_dim
        
        # 简单的 ResNet-like 结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, embed_dim, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


# ============================================================================
# 方式 3: 包装现有的 HuggingFace 模型
# ============================================================================

def build_encoder_from_hf(
    model_name: str = "facebook/dino-vitb16",
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    image_size: int = 384,
    **kwargs
) -> nn.Module:
    """
    从 HuggingFace 构建 encoder
    
    Args:
        model_name: HF 模型名称
        pretrained: 是否加载预训练权重
        pretrained_path: 额外的权重路径
        image_size: 图像尺寸
        **kwargs: 其他参数
        
    Returns:
        nn.Module: 包装后的 encoder
    """
    from transformers import AutoModel
    
    # 加载 HF 模型
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    # 如果有额外的权重路径，加载它
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=False)
    
    # 包装为标准格式
    class HFEncoderWrapper(nn.Module):
        def __init__(self, hf_model, img_size):
            super().__init__()
            self.model = hf_model
            self.image_size = img_size
            
            # 获取 embed_dim
            if hasattr(hf_model.config, 'hidden_size'):
                self.embed_dim = hf_model.config.hidden_size
            elif hasattr(hf_model.config, 'embed_dim'):
                self.embed_dim = hf_model.config.embed_dim
            else:
                self.embed_dim = 768
        
        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            outputs = self.model(x, output_hidden_states=True)
            # 取最后一层的 hidden states
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                return outputs.hidden_states[-1]
            else:
                return outputs
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_features(x)
    
    return HFEncoderWrapper(model, image_size)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("Testing Custom Vision Encoder...")
    
    # 测试 create_encoder 函数
    encoder = create_encoder(
        pretrained=False,
        image_size=224,
        depth=6
    )
    
    print(f"Encoder created: {type(encoder).__name__}")
    print(f"Embed dim: {encoder.embed_dim}")
    print(f"Image size: {encoder.image_size}")
    
    # 测试 forward
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = encoder(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dims: [{output.shape[0]}, {output.shape[1]}, {output.shape[2]}]")
    
    print("\n✓ Custom encoder test passed!")

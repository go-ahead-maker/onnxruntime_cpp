"""
Vision Encoder 模块

支持多种 vision encoder，可以替换 Florence-2 默认的 DaViT encoder
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from transformers import AutoConfig


class VisionEncoderFactory:
    """Vision Encoder 工厂类，用于创建不同类型的 encoder"""
    
    SUPPORTED_ENCODERS = {
        'davit', 'vit', 'swin', 'resnet', 'custom'
    }
    
    @staticmethod
    def create_encoder(
        encoder_type: str = 'davit',
        encoder_name: str = 'davit_base',
        pretrained: bool = True,
        image_size: int = 384,
        custom_path: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """
        创建 vision encoder
        
        Args:
            encoder_type: encoder 类型 (davit, vit, swin, resnet, custom)
            encoder_name: 具体编码器名称
            pretrained: 是否加载预训练权重
            image_size: 图像输入尺寸
            custom_path: 自定义编码器路径
            **kwargs: 其他参数
            
        Returns:
            nn.Module: vision encoder
        """
        if encoder_type not in VisionEncoderFactory.SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type}. "
                f"Supported types: {VisionEncoderFactory.SUPPORTED_ENCODERS}"
            )
        
        if encoder_type == 'custom':
            if custom_path is None:
                raise ValueError("custom_path must be provided for custom encoder")
            return VisionEncoderFactory._load_custom_encoder(custom_path, **kwargs)
        
        elif encoder_type == 'davit':
            return VisionEncoderFactory._create_davit_encoder(
                encoder_name, pretrained, image_size, **kwargs
            )
        
        elif encoder_type == 'vit':
            return VisionEncoderFactory._create_vit_encoder(
                encoder_name, pretrained, image_size, **kwargs
            )
        
        elif encoder_type == 'swin':
            return VisionEncoderFactory._create_swin_encoder(
                encoder_name, pretrained, image_size, **kwargs
            )
        
        elif encoder_type == 'resnet':
            return VisionEncoderFactory._create_resnet_encoder(
                encoder_name, pretrained, image_size, **kwargs
            )
        
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    @staticmethod
    def _create_davit_encoder(
        name: str, 
        pretrained: bool, 
        image_size: int,
        pretrained_path: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """创建 DaViT encoder (Florence-2 默认)
        
        Args:
            name: 模型名称
            pretrained: 是否加载预训练权重
            image_size: 图像尺寸
            pretrained_path: 自定义预训练权重路径（优先于自动下载）
            **kwargs: 其他参数
        """
        try:
            from timm import create_model
            model = create_model(
                name,
                pretrained=False,  # 先不加载，手动处理
                img_size=image_size,
                num_classes=0,
                global_pool='',
                **kwargs
            )
            
            # 加载预训练权重
            if pretrained:
                if pretrained_path is not None:
                    # 从自定义路径加载
                    print(f"Loading DaViT pretrained weights from: {pretrained_path}")
                    checkpoint = torch.load(pretrained_path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        checkpoint = checkpoint['state_dict']
                    # 过滤不匹配的键
                    filtered_checkpoint = {}
                    for k, v in checkpoint.items():
                        if k.startswith('model.'):
                            filtered_checkpoint[k[6:]] = v
                        else:
                            filtered_checkpoint[k] = v
                    
                    # 移除不需要的分类头
                    filtered_checkpoint = {
                        k: v for k, v in filtered_checkpoint.items() 
                        if not k.startswith('head.') and not k.startswith('classifier.')
                    }
                    
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=False)
                    if missing_keys:
                        print(f"Warning: Missing keys in checkpoint: {missing_keys[:10]}...")
                    if unexpected_keys:
                        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:10]}...")
                else:
                    # 使用 timm 自动下载预训练权重
                    print(f"Loading DaViT pretrained weights from timm (model: {name})")
                    temp_model = create_model(name, pretrained=True, num_classes=0)
                    temp_state = temp_model.state_dict()
                    # 过滤掉分类头
                    filtered_state = {
                        k: v for k, v in temp_state.items()
                        if not k.startswith('head.') and not k.startswith('classifier.')
                    }
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
                    del temp_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return DaViTEncoderWrapper(model, image_size)
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        except Exception as e:
            raise RuntimeError(f"Failed to create DaViT encoder: {e}")
    
    @staticmethod
    def _create_vit_encoder(
        name: str, 
        pretrained: bool, 
        image_size: int,
        pretrained_path: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """创建 ViT encoder
        
        Args:
            name: 模型名称
            pretrained: 是否加载预训练权重
            image_size: 图像尺寸
            pretrained_path: 自定义预训练权重路径
            **kwargs: 其他参数
        """
        try:
            from timm import create_model
            # 常见的 ViT 模型名称映射
            vit_names = {
                'vit_base': 'vit_base_patch16_224',
                'vit_large': 'vit_large_patch16_224',
                'vit_huge': 'vit_huge_patch14_224',
                'vit_small': 'vit_small_patch16_224',
            }
            model_name = vit_names.get(name, name)
            
            model = create_model(
                model_name,
                pretrained=False,
                img_size=image_size,
                num_classes=0,
                global_pool='',
                **kwargs
            )
            
            # 加载预训练权重
            if pretrained:
                if pretrained_path is not None:
                    print(f"Loading ViT pretrained weights from: {pretrained_path}")
                    checkpoint = torch.load(pretrained_path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        checkpoint = checkpoint['state_dict']
                    
                    filtered_checkpoint = {
                        k: v for k, v in checkpoint.items()
                        if not k.startswith('head.') and not k.startswith('classifier.')
                    }
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=False)
                else:
                    print(f"Loading ViT pretrained weights from timm (model: {model_name})")
                    temp_model = create_model(model_name, pretrained=True, num_classes=0)
                    temp_state = temp_model.state_dict()
                    filtered_state = {
                        k: v for k, v in temp_state.items()
                        if not k.startswith('head.') and not k.startswith('classifier.')
                    }
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
                    del temp_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return ViTEncoderWrapper(model, image_size)
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        except Exception as e:
            raise RuntimeError(f"Failed to create ViT encoder: {e}")
    
    @staticmethod
    def _create_swin_encoder(
        name: str, 
        pretrained: bool, 
        image_size: int,
        pretrained_path: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """创建 Swin Transformer encoder
        
        Args:
            name: 模型名称
            pretrained: 是否加载预训练权重
            image_size: 图像尺寸
            pretrained_path: 自定义预训练权重路径
            **kwargs: 其他参数
        """
        try:
            from timm import create_model
            # 常见的 Swin 模型名称映射
            swin_names = {
                'swin_tiny': 'swin_tiny_patch4_window7_224',
                'swin_small': 'swin_small_patch4_window7_224',
                'swin_base': 'swin_base_patch4_window7_224',
                'swin_large': 'swin_large_patch4_window7_224',
            }
            model_name = swin_names.get(name, name)
            
            model = create_model(
                model_name,
                pretrained=False,
                img_size=image_size,
                num_classes=0,
                **kwargs
            )
            
            # 加载预训练权重
            if pretrained:
                if pretrained_path is not None:
                    print(f"Loading Swin pretrained weights from: {pretrained_path}")
                    checkpoint = torch.load(pretrained_path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        checkpoint = checkpoint['state_dict']
                    
                    filtered_checkpoint = {
                        k: v for k, v in checkpoint.items()
                        if not k.startswith('head.') and not k.startswith('classifier.')
                    }
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=False)
                else:
                    print(f"Loading Swin pretrained weights from timm (model: {model_name})")
                    temp_model = create_model(model_name, pretrained=True, num_classes=0)
                    temp_state = temp_model.state_dict()
                    filtered_state = {
                        k: v for k, v in temp_state.items()
                        if not k.startswith('head.') and not k.startswith('classifier.')
                    }
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
                    del temp_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return SwinEncoderWrapper(model, image_size)
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        except Exception as e:
            raise RuntimeError(f"Failed to create Swin encoder: {e}")
    
    @staticmethod
    def _create_resnet_encoder(
        name: str, 
        pretrained: bool, 
        image_size: int,
        pretrained_path: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """创建 ResNet encoder
        
        Args:
            name: 模型名称
            pretrained: 是否加载预训练权重
            image_size: 图像尺寸
            pretrained_path: 自定义预训练权重路径
            **kwargs: 其他参数
        """
        try:
            from timm import create_model
            # 常见的 ResNet 模型名称映射
            resnet_names = {
                'resnet18': 'resnet18',
                'resnet34': 'resnet34',
                'resnet50': 'resnet50',
                'resnet101': 'resnet101',
                'resnet152': 'resnet152',
            }
            model_name = resnet_names.get(name, name)
            
            model = create_model(
                model_name,
                pretrained=False,
                img_size=image_size,
                num_classes=0,
                **kwargs
            )
            
            # 加载预训练权重
            if pretrained:
                if pretrained_path is not None:
                    print(f"Loading ResNet pretrained weights from: {pretrained_path}")
                    checkpoint = torch.load(pretrained_path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        checkpoint = checkpoint['state_dict']
                    
                    filtered_checkpoint = {
                        k: v for k, v in checkpoint.items()
                        if not k.startswith('head.') and not k.startswith('classifier.')
                        and not k.startswith('fc.')
                    }
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=False)
                else:
                    print(f"Loading ResNet pretrained weights from timm (model: {model_name})")
                    temp_model = create_model(model_name, pretrained=True, num_classes=0)
                    temp_state = temp_model.state_dict()
                    filtered_state = {
                        k: v for k, v in temp_state.items()
                        if not k.startswith('head.') and not k.startswith('classifier.')
                        and not k.startswith('fc.')
                    }
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
                    del temp_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return ResNetEncoderWrapper(model, image_size)
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        except Exception as e:
            raise RuntimeError(f"Failed to create ResNet encoder: {e}")
    
    @staticmethod
    def _load_custom_encoder(custom_path: str, **kwargs) -> nn.Module:
        """加载自定义 encoder"""
        import importlib.util
        import sys
        
        # 加载自定义模块
        spec = importlib.util.spec_from_file_location("custom_encoder", custom_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load custom encoder from {custom_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_encoder"] = module
        spec.loader.exec_module(module)
        
        # 假设自定义模块中有 create_encoder 函数
        if not hasattr(module, 'create_encoder'):
            raise AttributeError(
                "Custom encoder module must have a 'create_encoder' function"
            )
        
        return module.create_encoder(**kwargs)


class BaseEncoderWrapper(nn.Module):
    """Encoder 基础包装类"""
    
    def __init__(self, model: nn.Module, image_size: int):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.embed_dim = self._get_embed_dim()
    
    def _get_embed_dim(self) -> int:
        """获取 embedding 维度"""
        if hasattr(self.model, 'embed_dim'):
            return self.model.embed_dim
        elif hasattr(self.model, 'num_features'):
            return self.model.num_features
        else:
            # 尝试通过 forward 获取
            return self._infer_embed_dim()
    
    def _infer_embed_dim(self) -> int:
        """推断 embedding 维度"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.image_size, self.image_size)
            output = self.forward_features(dummy_input)
            if isinstance(output, (list, tuple)):
                return output[0].shape[-1]
            return output.shape[-1]
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取图像特征"""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.forward_features(x)
    
    def freeze(self):
        """冻结 encoder"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """解冻 encoder"""
        for param in self.parameters():
            param.requires_grad = True


class DaViTEncoderWrapper(BaseEncoderWrapper):
    """DaViT Encoder 包装器"""
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            图像特征 [B, L, D]
        """
        # DaViT 的 forward 返回多尺度特征，取最后一层
        features = self.model(x)
        
        # 如果是多尺度特征，选择适合的特征层
        if isinstance(features, (list, tuple)):
            # 通常最后一个特征是最高级语义特征
            features = features[-1]
        
        # 如果是 [B, C, H, W] 格式，转换为 [B, L, D]
        if len(features.shape) == 4:
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        return features


class ViTEncoderWrapper(BaseEncoderWrapper):
    """ViT Encoder 包装器"""
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            图像特征 [B, L, D]
        """
        features = self.model(x)
        
        # ViT 通常返回 [B, L, D] 格式
        if len(features.shape) == 3:
            return features
        
        # 如果是 [B, C, H, W] 格式，转换
        if len(features.shape) == 4:
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)
        
        return features


class SwinEncoderWrapper(BaseEncoderWrapper):
    """Swin Transformer Encoder 包装器"""
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            图像特征 [B, L, D]
        """
        features = self.model(x)
        
        # Swin 可能返回多尺度特征或单个特征
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # 如果是 [B, C, H, W] 格式，转换
        if len(features.shape) == 4:
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)
        
        return features


class ResNetEncoderWrapper(BaseEncoderWrapper):
    """ResNet Encoder 包装器"""
    
    def __init__(self, model: nn.Module, image_size: int):
        super().__init__(model, image_size)
        # 添加全局平均池化后的投影层（如果需要）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            图像特征 [B, L, D]
        """
        # ResNet 的 forward 通常返回 logits，需要修改
        # 这里我们手动提取特征
        features = self.model.forward_features(x) if hasattr(self.model, 'forward_features') else self.model(x)
        
        # 如果是 [B, C, H, W] 格式，转换
        if len(features.shape) == 4:
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)
        
        return features


def get_encoder_config(encoder_type: str, encoder_name: str) -> Dict[str, Any]:
    """
    获取 encoder 配置信息
    
    Args:
        encoder_type: encoder 类型
        encoder_name: encoder 名称
        
    Returns:
        配置字典
    """
    configs = {
        'davit': {
            'default_name': 'davit_base',
            'image_sizes': [224, 384, 512],
            'description': 'DaViT (Dual-Attention Vision Transformer)'
        },
        'vit': {
            'default_name': 'vit_base_patch16_224',
            'image_sizes': [224, 384],
            'description': 'Vision Transformer'
        },
        'swin': {
            'default_name': 'swin_base_patch4_window7_224',
            'image_sizes': [224, 384],
            'description': 'Swin Transformer'
        },
        'resnet': {
            'default_name': 'resnet50',
            'image_sizes': [224, 384],
            'description': 'Residual Network'
        }
    }
    
    return configs.get(encoder_type, {})


if __name__ == "__main__":
    # 测试 encoder 创建
    print("Testing Vision Encoder Factory...")
    
    # 测试 DaViT
    try:
        davit_encoder = VisionEncoderFactory.create_encoder(
            encoder_type='davit',
            encoder_name='davit_tiny',
            pretrained=False,
            image_size=224
        )
        print(f"✓ DaViT encoder created: {davit_encoder.embed_dim} dims")
    except Exception as e:
        print(f"✗ Failed to create DaViT encoder: {e}")
    
    # 测试 ViT
    try:
        vit_encoder = VisionEncoderFactory.create_encoder(
            encoder_type='vit',
            encoder_name='vit_tiny_patch16_224',
            pretrained=False,
            image_size=224
        )
        print(f"✓ ViT encoder created: {vit_encoder.embed_dim} dims")
    except Exception as e:
        print(f"✗ Failed to create ViT encoder: {e}")
    
    # 测试 Swin
    try:
        swin_encoder = VisionEncoderFactory.create_encoder(
            encoder_type='swin',
            encoder_name='swin_tiny_patch4_window7_224',
            pretrained=False,
            image_size=224
        )
        print(f"✓ Swin encoder created: {swin_encoder.embed_dim} dims")
    except Exception as e:
        print(f"✗ Failed to create Swin encoder: {e}")
    
    print("\nAll tests completed!")

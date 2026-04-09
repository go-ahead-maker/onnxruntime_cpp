"""
Florence-2 模型包装器

支持自定义 vision encoder 的 Florence-2 模型封装
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from transformers import AutoProcessor, AutoModelForCausalLM


class Florence2Wrapper(nn.Module):
    """
    Florence-2 模型包装器
    
    支持替换 vision encoder，保持其他组件不变
    """
    
    def __init__(
        self,
        base_model_name: str = "microsoft/Florence-2-base",
        vision_encoder: Optional[nn.Module] = None,
        freeze_vision_encoder: bool = False,
        text_decoder_pretrained: bool = True,
        max_length: int = 256,
        **kwargs
    ):
        """
        初始化 Florence-2 包装器
        
        Args:
            base_model_name: Florence-2 基础模型名称
            vision_encoder: 自定义 vision encoder（可选）
            freeze_vision_encoder: 是否冻结 vision encoder
            text_decoder_pretrained: 是否使用预训练文本解码器
            max_length: 最大生成长度
            **kwargs: 其他参数
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.max_length = max_length
        
        # 加载原始 Florence-2 模型
        print(f"Loading Florence-2 model: {base_model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load {base_model_name}, trying alternative...")
            # 如果加载失败，尝试其他方法
            raise RuntimeError(f"Failed to load Florence-2 model: {e}")
        
        # 获取原始 vision encoder 的配置信息
        self.original_vision_config = self._get_vision_config()
        
        # 如果提供了自定义 vision encoder，进行替换
        if vision_encoder is not None:
            self._replace_vision_encoder(vision_encoder, freeze_vision_encoder)
        
        # 加载 processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # 配置信息
        self.config = {
            'base_model': base_model_name,
            'vision_encoder_type': type(vision_encoder).__name__ if vision_encoder else 'default',
            'freeze_vision_encoder': freeze_vision_encoder,
            'max_length': max_length
        }
        
        print(f"Florence-2 wrapper initialized with config: {self.config}")
    
    def _get_vision_config(self) -> Dict[str, Any]:
        """获取 vision encoder 配置"""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'vision_config'):
            return {
                'embed_dim': getattr(self.model.config.vision_config, 'embed_dim', None),
                'image_size': getattr(self.model.config.vision_config, 'image_size', 384),
                'num_channels': getattr(self.model.config.vision_config, 'num_channels', 3)
            }
        return {
            'embed_dim': None,
            'image_size': 384,
            'num_channels': 3
        }
    
    def _replace_vision_encoder(
        self,
        vision_encoder: nn.Module,
        freeze: bool = False
    ):
        """
        替换 vision encoder
        
        Args:
            vision_encoder: 新的 vision encoder
            freeze: 是否冻结 encoder
        """
        print(f"Replacing vision encoder with {type(vision_encoder).__name__}")
        
        # 获取原始 vision encoder 的位置
        # Florence-2 的架构可能因版本而异，需要适配
        if hasattr(self.model, 'vision_tower'):
            # 某些版本的 Florence-2 使用 vision_tower
            original_encoder = self.model.vision_tower
            self.model.vision_tower = vision_encoder
        elif hasattr(self.model, 'vision_model'):
            # 某些版本使用 vision_model
            original_encoder = self.model.vision_model
            self.model.vision_model = vision_encoder
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'vision'):
            # 某些版本使用 encoder.vision
            original_encoder = self.model.encoder.vision
            self.model.encoder.vision = vision_encoder
        else:
            # 尝试找到 vision 相关的模块
            vision_module = self._find_vision_module()
            if vision_module is not None:
                original_encoder = vision_module
            else:
                print("Warning: Could not find vision encoder module")
                return
        
        # 如果需要，冻结 encoder
        if freeze:
            print("Freezing vision encoder...")
            vision_encoder.freeze()
        
        # 打印替换信息
        print(f"Replaced {type(original_encoder).__name__} with {type(vision_encoder).__name__}")
        print(f"New encoder embed_dim: {vision_encoder.embed_dim}")
    
    def _find_vision_module(self) -> Optional[nn.Module]:
        """查找 vision 相关模块"""
        # 递归查找包含 'vision' 或 'image' 的模块
        for name, module in self.model.named_modules():
            if 'vision' in name.lower() or 'image' in name.lower():
                if isinstance(module, nn.Module) and name.split('.')[-1] in ['encoder', 'model', 'tower']:
                    return module
        return None
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            pixel_values: 图像像素值 [B, C, H, W]
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            labels: 标签（用于训练）
            task_prompt: 任务提示符
            **kwargs: 其他参数
            
        Returns:
            模型输出
        """
        # 构建输入
        inputs = {
            'pixel_values': pixel_values
        }
        
        if input_ids is not None:
            inputs['input_ids'] = input_ids
        
        if attention_mask is not None:
            inputs['attention_mask'] = attention_mask
        
        if labels is not None:
            inputs['labels'] = labels
        
        # 添加任务提示符（如果需要）
        if task_prompt is not None:
            # Florence-2 期望特定的输入格式
            pass
        
        # 前向传播
        outputs = self.model(**inputs, **kwargs)
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        task_prompt: str = "<OCR>",
        text_input: Optional[str] = None,
        max_length: Optional[int] = None,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs
    ) -> List[str]:
        """
        生成文本
        
        Args:
            pixel_values: 图像像素值
            task_prompt: 任务提示符（如 <OCR>, <CAPTION> 等）
            text_input: 额外的文本输入
            max_length: 最大生成长度
            num_beams: beam search 的 beam 数
            do_sample: 是否采样
            **kwargs: 其他参数
            
        Returns:
            生成的文本列表
        """
        if max_length is None:
            max_length = self.max_length
        
        # 构建完整的 prompt
        prompt = task_prompt
        if text_input is not None:
            prompt = f"{task_prompt}{text_input}"
        
        # 使用 processor 处理输入
        inputs = self.processor(
            text=prompt,
            images=pixel_values,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成
        generated_ids = self.model.generate(
            pixel_values=inputs.get('pixel_values', pixel_values),
            input_ids=inputs.get('input_ids', None),
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            **kwargs
        )
        
        # 解码生成的文本
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return generated_texts
    
    @property
    def device(self) -> torch.device:
        """获取模型所在设备"""
        return next(self.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        """获取模型数据类型"""
        return next(self.parameters()).dtype
    
    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def print_parameter_summary(self):
        """打印参数摘要"""
        total = self.get_total_parameters()
        trainable = self.get_trainable_parameters()
        
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {total - trainable:,}")
        print(f"Trainable ratio: {trainable / total * 100:.2f}%")


def create_florence2_model(
    base_model: str = "microsoft/Florence-2-base",
    vision_encoder_config: Optional[Dict[str, Any]] = None,
    freeze_vision_encoder: bool = False,
    **kwargs
) -> Florence2Wrapper:
    """
    创建 Florence-2 模型的工厂函数
    
    Args:
        base_model: 基础模型名称
        vision_encoder_config: vision encoder 配置
        freeze_vision_encoder: 是否冻结 vision encoder
        **kwargs: 其他参数
        
    Returns:
        Florence2Wrapper
    """
    from models.vision_encoders import VisionEncoderFactory
    
    vision_encoder = None
    
    if vision_encoder_config is not None:
        encoder_type = vision_encoder_config.get('type', 'davit')
        encoder_name = vision_encoder_config.get('name', 'davit_base')
        pretrained = vision_encoder_config.get('pretrained', True)
        image_size = vision_encoder_config.get('image_size', 384)
        custom_path = vision_encoder_config.get('custom_path', None)
        
        print(f"Creating vision encoder: {encoder_type} - {encoder_name}")
        vision_encoder = VisionEncoderFactory.create_encoder(
            encoder_type=encoder_type,
            encoder_name=encoder_name,
            pretrained=pretrained,
            image_size=image_size,
            custom_path=custom_path
        )
    
    model = Florence2Wrapper(
        base_model_name=base_model,
        vision_encoder=vision_encoder,
        freeze_vision_encoder=freeze_vision_encoder,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # 测试模型创建
    print("Testing Florence-2 Wrapper...")
    
    # 测试基本模型加载（不实际加载大模型）
    print("\nNote: Skipping actual model loading for testing purposes.")
    print("To test with real model, ensure you have:")
    print("1. Sufficient GPU memory")
    print("2. Internet connection to download model")
    print("3. HuggingFace transformers installed")
    
    # 测试 vision encoder 创建
    try:
        from models.vision_encoders import VisionEncoderFactory
        
        encoder = VisionEncoderFactory.create_encoder(
            encoder_type='vit',
            encoder_name='vit_tiny_patch16_224',
            pretrained=False,
            image_size=224
        )
        print(f"✓ Vision encoder created: {encoder.embed_dim} dims")
    except Exception as e:
        print(f"✗ Failed to create vision encoder: {e}")
    
    print("\nWrapper test completed!")

"""
OCR 数据集构建模块

提供清晰的 dataset 构建流程，支持多种 OCR 数据格式
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np


class OCRDataset(Dataset):
    """
    OCR 数据集类
    
    支持以下标注格式：
    1. 简单格式：{"image_path": str, "text": str}
    2. COCO 格式：包含 images 和 annotations
    3. ICDAR 格式：包含多边形标注和文本
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[Any] = None,
        task_prompt: str = "<OCR>",
        max_length: int = 256,
        format_type: str = "simple"
    ):
        """
        初始化 OCR 数据集
        
        Args:
            image_dir: 图像目录路径
            annotation_file: 标注文件路径 (JSON)
            transform: 图像预处理/增强变换
            task_prompt: Florence-2 任务提示符
            max_length: 最大文本长度
            format_type: 标注格式类型 (simple, coco, icdar)
        """
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.transform = transform
        self.task_prompt = task_prompt
        self.max_length = max_length
        self.format_type = format_type
        
        # 加载标注数据
        self.annotations = self._load_annotations()
        
        # 统计信息
        self.total_samples = len(self.annotations)
        
        print(f"Loaded {self.total_samples} samples from {annotation_file}")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """加载标注文件"""
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if self.format_type == "simple":
            return self._parse_simple_format(data)
        elif self.format_type == "coco":
            return self._parse_coco_format(data)
        elif self.format_type == "icdar":
            return self._parse_icdar_format(data)
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")
    
    def _parse_simple_format(self, data: List[Dict]) -> List[Dict]:
        """
        解析简单格式
        
        格式示例：
        [
            {"image_path": "img1.jpg", "text": "Hello World"},
            {"image_path": "img2.jpg", "text": "OCR Test"}
        ]
        """
        annotations = []
        for item in data:
            if 'image_path' not in item or 'text' not in item:
                continue
            
            annotations.append({
                'image_path': str(self.image_dir / item['image_path']),
                'text': item['text'],
                'bbox': item.get('bbox', None),  # 可选的边界框
                'language': item.get('language', 'en')  # 可选的语言
            })
        
        return annotations
    
    def _parse_coco_format(self, data: Dict) -> List[Dict]:
        """
        解析 COCO 格式
        
        COCO 格式包含 images 和 annotations 两个主要部分
        """
        # 构建图像 ID 到路径的映射
        image_map = {}
        for img_info in data.get('images', []):
            image_map[img_info['id']] = img_info['file_name']
        
        # 构建类别 ID 到名称的映射（如果有）
        category_map = {}
        for cat_info in data.get('categories', []):
            category_map[cat_info['id']] = cat_info.get('name', '')
        
        annotations = []
        for ann in data.get('annotations', []):
            image_id = ann.get('image_id')
            if image_id not in image_map:
                continue
            
            image_path = str(self.image_dir / image_map[image_id])
            
            # COCO 中的 text 可能在不同的字段中
            text = ann.get('text', ann.get('caption', ''))
            if not text:
                # 尝试从 category 获取
                cat_id = ann.get('category_id')
                text = category_map.get(cat_id, '')
            
            annotations.append({
                'image_path': image_path,
                'text': text,
                'bbox': ann.get('bbox', None),
                'segmentation': ann.get('segmentation', None)
            })
        
        return annotations
    
    def _parse_icdar_format(self, data: Any) -> List[Dict]:
        """
        解析 ICDAR 格式
        
        ICDAR 格式通常包含多边形标注和文本内容
        """
        annotations = []
        
        # ICDAR 可能是列表或字典
        if isinstance(data, dict):
            data = data.get('annotations', data.get('results', []))
        
        for item in data:
            if 'image_path' not in item and 'filename' not in item:
                continue
            
            image_path = item.get('image_path', item.get('filename', ''))
            image_path = str(self.image_dir / image_path)
            
            # 提取文本
            text = item.get('text', item.get('transcription', ''))
            
            # 提取多边形或边界框
            polygon = item.get('polygon', item.get('points', None))
            bbox = item.get('bbox', None)
            
            annotations.append({
                'image_path': image_path,
                'text': text,
                'polygon': polygon,
                'bbox': bbox
            })
        
        return annotations
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Returns:
            包含以下键的字典：
            - image: PIL Image
            - text: 文本标注
            - task_prompt: 任务提示符
            - image_path: 图像路径
            - metadata: 其他元数据
        """
        ann = self.annotations[idx]
        
        # 加载图像
        image_path = ann['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # 如果图像加载失败，返回一个空白图像
            print(f"Warning: Failed to load image {image_path}: {e}")
            image = Image.new('RGB', (384, 384), color=(128, 128, 128))
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        # 准备输出
        item = {
            'image': image,
            'text': ann['text'],
            'task_prompt': self.task_prompt,
            'image_path': image_path,
            'metadata': {
                'bbox': ann.get('bbox', None),
                'language': ann.get('language', 'en'),
                'polygon': ann.get('polygon', None)
            }
        }
        
        return item
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        text_lengths = [len(ann['text']) for ann in self.annotations]
        
        stats = {
            'total_samples': self.total_samples,
            'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'min_text_length': min(text_lengths) if text_lengths else 0,
            'max_text_length': max(text_lengths) if text_lengths else 0,
            'format_type': self.format_type
        }
        
        return stats


class OCRDataCollator:
    """
    OCR 数据整理器
    
    用于将 batch 中的样本整理成模型可接受的格式
    """
    
    def __init__(
        self,
        tokenizer: Any,
        processor: Any,
        max_length: int = 256,
        padding: str = "max_length"
    ):
        """
        初始化数据整理器
        
        Args:
            tokenizer: 文本分词器
            processor: Florence-2 处理器
            max_length: 最大序列长度
            padding: 填充策略
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.padding = padding
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理 batch 数据
        
        Args:
            batch: 样本列表
            
        Returns:
            整理后的 batch 数据
        """
        images = []
        texts = []
        prompts = []
        image_paths = []
        
        for sample in batch:
            images.append(sample['image'])
            texts.append(sample['text'])
            prompts.append(sample['task_prompt'])
            image_paths.append(sample['image_path'])
        
        # 处理图像和文本
        # Florence-2 期望的输入格式
        inputs = []
        for image, prompt, text in zip(images, prompts, texts):
            # 构建 Florence-2 的输入格式
            # 格式：<task_prompt><text>
            if isinstance(image, torch.Tensor):
                # 如果已经是 tensor，转换为 PIL Image
                from torchvision import transforms
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            
            input_dict = {
                'image': image,
                'task_prompt': prompt,
                'text': text
            }
            inputs.append(input_dict)
        
        # 使用 processor 处理
        # 注意：这里需要根据实际的 Florence-2 processor API 调整
        try:
            processed = self.processor(
                text=[f"{prompt} {text}" for prompt, text in zip(prompts, texts)],
                images=images,
                return_tensors="pt",
                padding=self.padding,
                max_length=self.max_length
            )
        except Exception as e:
            # 如果 processor 处理失败，手动构建
            processed = self._manual_process(images, prompts, texts)
        
        # 添加元数据
        processed['image_paths'] = image_paths
        processed['ground_truth_texts'] = texts
        
        return processed
    
    def _manual_process(
        self,
        images: List,
        prompts: List[str],
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """手动处理数据（备用方案）"""
        # 这里实现一个简单的处理逻辑
        # 实际使用时应该根据 Florence-2 的具体 API 调整
        
        pixel_values = []
        for image in images:
            if isinstance(image, torch.Tensor):
                pixel_values.append(image)
            else:
                # 转换为 tensor
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((384, 384)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values.append(transform(image))
        
        pixel_values = torch.stack(pixel_values)
        
        # 编码文本
        text_inputs = self.tokenizer(
            [f"{p} {t}" for p, t in zip(prompts, texts)],
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': pixel_values,
            **text_inputs
        }


def create_ocr_dataloader(
    image_dir: str,
    annotation_file: str,
    tokenizer: Any,
    processor: Any,
    transform: Optional[Any] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    task_prompt: str = "<OCR>",
    max_length: int = 256,
    format_type: str = "simple",
    pin_memory: bool = True
) -> DataLoader:
    """
    创建 OCR DataLoader
    
    Args:
        image_dir: 图像目录
        annotation_file: 标注文件
        tokenizer: 文本分词器
        processor: Florence-2 处理器
        transform: 图像变换
        batch_size: 批次大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        task_prompt: 任务提示符
        max_length: 最大长度
        format_type: 标注格式
        pin_memory: 是否固定内存
        
    Returns:
        DataLoader
    """
    dataset = OCRDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=transform,
        task_prompt=task_prompt,
        max_length=max_length,
        format_type=format_type
    )
    
    # 打印数据集统计信息
    stats = dataset.get_statistics()
    print(f"Dataset statistics: {stats}")
    
    collator = OCRDataCollator(
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collator,
        pin_memory=pin_memory
    )
    
    return dataloader


def create_sample_annotation_file(
    output_path: str,
    num_samples: int = 10,
    format_type: str = "simple"
):
    """
    创建示例标注文件（用于测试）
    
    Args:
        output_path: 输出路径
        num_samples: 样本数量
        format_type: 格式类型
    """
    sample_texts = [
        "Hello World",
        "OCR Test Sample",
        "Florence-2 Fine-tuning",
        "Text Recognition",
        "Deep Learning",
        "Computer Vision",
        "Natural Language Processing",
        "Machine Learning",
        "Artificial Intelligence",
        "Neural Networks"
    ]
    
    if format_type == "simple":
        data = []
        for i in range(num_samples):
            data.append({
                "image_path": f"img_{i:04d}.jpg",
                "text": sample_texts[i % len(sample_texts)]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    elif format_type == "coco":
        data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "text"}]
        }
        
        for i in range(num_samples):
            data["images"].append({
                "id": i,
                "file_name": f"img_{i:04d}.jpg",
                "width": 384,
                "height": 384
            })
            
            data["annotations"].append({
                "id": i,
                "image_id": i,
                "category_id": 1,
                "text": sample_texts[i % len(sample_texts)],
                "bbox": [0, 0, 384, 384]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample annotation file: {output_path}")


if __name__ == "__main__":
    # 测试数据集创建
    print("Testing OCR Dataset...")
    
    # 创建示例标注文件
    create_sample_annotation_file(
        "./sample_annotations.json",
        num_samples=5,
        format_type="simple"
    )
    
    # 测试简单格式
    try:
        dataset = OCRDataset(
            image_dir="./",
            annotation_file="./sample_annotations.json",
            format_type="simple"
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # 获取统计信息
        stats = dataset.get_statistics()
        print(f"Statistics: {stats}")
        
        # 测试获取单个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Text: {sample['text']}")
    except Exception as e:
        print(f"✗ Failed to test dataset: {e}")
    
    print("\nDataset test completed!")

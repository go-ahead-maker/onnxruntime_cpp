"""
OCR 推理脚本

使用微调后的 Florence-2 模型进行 OCR 推理
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from PIL import Image
import yaml


def load_model(checkpoint_path: str, config_path: Optional[str] = None):
    """加载模型"""
    from models.florence2_wrapper import Florence2Wrapper
    
    # 加载检查点
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 如果提供了配置文件，使用配置文件创建模型
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        from models.florence2_wrapper import create_florence2_model
        model = create_florence2_model(
            base_model=config['model']['base_model'],
            vision_encoder_config=config['model'].get('vision_encoder'),
            freeze_vision_encoder=config['model'].get('freeze_vision_encoder', False),
            max_length=config['model']['text_decoder']['max_length']
        )
    else:
        # 从检查点中获取配置
        config = checkpoint.get('config', {})
        base_model = config.get('model', {}).get('base_model', 'microsoft/Florence-2-base')
        
        model = Florence2Wrapper(
            base_model_name=base_model,
            max_length=config.get('model', {}).get('text_decoder', {}).get('max_length', 256)
        )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    
    return model, device


def preprocess_image(image_path: str, image_size: int = 384):
    """预处理图像"""
    from data.transforms import OCRTransform
    
    transform = OCRTransform(
        image_size=image_size,
        augmentations={'normalize': True}
    )
    
    image = Image.open(image_path).convert('RGB')
    pixel_values = transform(image).unsqueeze(0)  # 添加 batch 维度
    
    return pixel_values, image


@torch.no_grad()
def run_ocr(
    model,
    device: torch.device,
    image_path: str,
    task_prompt: str = "<OCR>",
    max_length: int = 256,
    num_beams: int = 1
) -> Dict[str, Any]:
    """
    运行 OCR 推理
    
    Args:
        model: 模型
        device: 设备
        image_path: 图像路径
        task_prompt: 任务提示符
        max_length: 最大生成长度
        num_beams: beam search 数量
        
    Returns:
        包含识别结果的字典
    """
    # 预处理图像
    pixel_values, original_image = preprocess_image(image_path)
    pixel_values = pixel_values.to(device)
    
    # 生成文本
    generated_texts = model.generate(
        pixel_values=pixel_values,
        task_prompt=task_prompt,
        max_length=max_length,
        num_beams=num_beams
    )
    
    result = {
        'image_path': image_path,
        'text': generated_texts[0] if generated_texts else '',
        'task_prompt': task_prompt
    }
    
    return result


@torch.no_grad()
def run_batch_ocr(
    model,
    device: torch.device,
    image_paths: List[str],
    task_prompt: str = "<OCR>",
    max_length: int = 256,
    num_beams: int = 1,
    batch_size: int = 4
) -> List[Dict[str, Any]]:
    """批量 OCR 推理"""
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # 预处理批次图像
        images = []
        for path in batch_paths:
            pixel_values, _ = preprocess_image(path)
            images.append(pixel_values)
        
        batch_pixel_values = torch.cat(images, dim=0).to(device)
        
        # 生成文本
        generated_texts = model.generate(
            pixel_values=batch_pixel_values,
            task_prompt=task_prompt,
            max_length=max_length,
            num_beams=num_beams
        )
        
        # 收集结果
        for path, text in zip(batch_paths, generated_texts):
            results.append({
                'image_path': path,
                'text': text,
                'task_prompt': task_prompt
            })
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存结果"""
    import json
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Florence-2 OCR Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--image-dir', type=str, default=None, help='Path to directory of images')
    parser.add_argument('--output', type=str, default='ocr_results.json', help='Output file path')
    parser.add_argument('--task-prompt', type=str, default='<OCR>', help='Task prompt')
    parser.add_argument('--max-length', type=int, default=256, help='Max generation length')
    parser.add_argument('--num-beams', type=int, default=1, help='Number of beams for beam search')
    args = parser.parse_args()
    
    # 加载模型
    model, device = load_model(args.checkpoint, args.config)
    
    # 收集图像路径
    image_paths = []
    
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_paths = [
            str(p) for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        image_paths.sort()
    else:
        print("Error: Please provide either --image or --image-dir")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images to process")
    
    # 运行 OCR
    if len(image_paths) == 1:
        result = run_ocr(
            model=model,
            device=device,
            image_path=image_paths[0],
            task_prompt=args.task_prompt,
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        results = [result]
        
        # 打印结果
        print("\n" + "="*50)
        print(f"Image: {result['image_path']}")
        print(f"Recognized Text: {result['text']}")
        print("="*50)
    else:
        results = run_batch_ocr(
            model=model,
            device=device,
            image_paths=image_paths,
            task_prompt=args.task_prompt,
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        
        # 打印结果
        print("\n" + "="*50)
        print("OCR Results:")
        print("="*50)
        for i, result in enumerate(results, 1):
            print(f"\n[{i}/{len(results)}] {result['image_path']}")
            print(f"Text: {result['text']}")
    
    # 保存结果
    save_results(results, args.output)
    
    print(f"\nProcessed {len(results)} images")


if __name__ == '__main__':
    main()

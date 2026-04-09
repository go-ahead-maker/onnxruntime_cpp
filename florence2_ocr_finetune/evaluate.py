"""
OCR 评估脚本

评估微调后的 Florence-2 模型的 OCR 性能
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict

import torch
import yaml
from tqdm import tqdm


def load_model(checkpoint_path: str, config_path: Optional[str] = None):
    """加载模型"""
    from models.florence2_wrapper import Florence2Wrapper, create_florence2_model
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model = create_florence2_model(
            base_model=config['model']['base_model'],
            vision_encoder_config=config['model'].get('vision_encoder'),
            freeze_vision_encoder=config['model'].get('freeze_vision_encoder', False),
            max_length=config['model']['text_decoder']['max_length']
        )
    else:
        config = checkpoint.get('config', {})
        base_model = config.get('model', {}).get('base_model', 'microsoft/Florence-2-base')
        
        model = Florence2Wrapper(
            base_model_name=base_model,
            max_length=config.get('model', {}).get('text_decoder', {}).get('max_length', 256)
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    return model, device


def compute_ocr_metrics(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, float]:
    """计算 OCR 评估指标"""
    try:
        import editdistance
    except ImportError:
        print("Warning: editdistance not installed. Installing...")
        os.system("pip install editdistance")
        import editdistance
    
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    n = len(predictions)
    if n == 0:
        return {
            'exact_match': 0.0,
            'avg_edit_distance': 0.0,
            'normalized_edit_distance': 0.0,
            'case_insensitive_exact_match': 0.0
        }
    
    # 精确匹配
    exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
    exact_match = exact_matches / n
    
    # 不区分大小写的精确匹配
    ci_exact_matches = sum(
        1 for p, g in zip(predictions, ground_truths) 
        if p.strip().lower() == g.strip().lower()
    )
    ci_exact_match = ci_exact_matches / n
    
    # 编辑距离
    edit_distances = [editdistance.eval(p.lower(), g.lower()) for p, g in zip(predictions, ground_truths)]
    avg_edit_distance = sum(edit_distances) / n
    
    # 归一化编辑距离
    total_gt_length = sum(len(g) for g in ground_truths)
    normalized_edit_distance = sum(edit_distances) / total_gt_length if total_gt_length > 0 else 0
    
    # 字符级准确率
    correct_chars = sum(
        sum(1 for a, b in zip(p.lower(), g.lower()) if a == b)
        for p, g in zip(predictions, ground_truths)
    )
    char_accuracy = correct_chars / total_gt_length if total_gt_length > 0 else 0
    
    return {
        'exact_match': exact_match,
        'case_insensitive_exact_match': ci_exact_match,
        'avg_edit_distance': avg_edit_distance,
        'normalized_edit_distance': normalized_edit_distance,
        'character_accuracy': char_accuracy
    }


def evaluate_on_dataset(
    model,
    processor,
    device: torch.device,
    image_dir: str,
    annotation_file: str,
    task_prompt: str = "<OCR>",
    batch_size: int = 4,
    max_length: int = 256
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """在数据集上评估模型"""
    from PIL import Image
    from data.transforms import OCRTransform
    import torchvision.transforms as T
    
    # 加载标注
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 准备图像路径和真实文本
    image_paths = []
    ground_truths = []
    
    for ann in annotations:
        if 'image_path' in ann and 'text' in ann:
            image_path = str(Path(image_dir) / ann['image_path'])
            if os.path.exists(image_path):
                image_paths.append(image_path)
                ground_truths.append(ann['text'])
    
    print(f"Found {len(image_paths)} valid images")
    
    # 预处理
    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 批量推理
    predictions = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Evaluating"):
        batch_paths = image_paths[i:i + batch_size]
        batch_gts = ground_truths[i:i + batch_size]
        
        # 加载和处理图像
        images = []
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            pixel_values = transform(image)
            images.append(pixel_values)
        
        batch_pixel_values = torch.stack(images).to(device)
        
        # 生成预测
        with torch.no_grad():
            generated_texts = model.generate(
                pixel_values=batch_pixel_values,
                task_prompt=task_prompt,
                max_length=max_length,
                num_beams=1
            )
        
        predictions.extend(generated_texts)
    
    # 计算指标
    metrics = compute_ocr_metrics(predictions, ground_truths)
    
    return predictions, ground_truths, metrics


def save_evaluation_results(
    predictions: List[str],
    ground_truths: List[str],
    metrics: Dict[str, float],
    output_path: str
):
    """保存评估结果"""
    results = {
        'metrics': metrics,
        'samples': [
            {
                'prediction': p,
                'ground_truth': g,
                'match': p.strip().lower() == g.strip().lower()
            }
            for p, g in zip(predictions, ground_truths)
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Florence-2 OCR Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--image-dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--annotation-file', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file path')
    parser.add_argument('--task-prompt', type=str, default='<OCR>', help='Task prompt')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--max-length', type=int, default=256, help='Max generation length')
    args = parser.parse_args()
    
    # 加载模型
    model, device = load_model(args.checkpoint, args.config)
    
    # 评估
    predictions, ground_truths, metrics = evaluate_on_dataset(
        model=model,
        processor=model.processor,
        device=device,
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        task_prompt=args.task_prompt,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total samples: {len(predictions)}")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("="*60)
    
    # 保存结果
    save_evaluation_results(predictions, ground_truths, metrics, args.output)
    
    # 显示一些示例
    print("\nSample Predictions:")
    print("-"*60)
    for i in range(min(5, len(predictions))):
        print(f"\n[{i+1}] Ground Truth: {ground_truths[i]}")
        print(f"    Prediction:   {predictions[i]}")
        print(f"    Match:        {predictions[i].strip().lower() == ground_truths[i].strip().lower()}")


if __name__ == '__main__':
    main()

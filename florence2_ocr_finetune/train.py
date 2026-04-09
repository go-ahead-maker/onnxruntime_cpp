"""
OCR 微调训练脚本

用于微调 Florence-2 模型进行 OCR 任务
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.florence2_wrapper import create_florence2_model
from data.dataset import create_ocr_dataloader, OCRDataset
from data.transforms import create_transform


class OCRTrainer:
    """OCR 微调训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        
        # 设置随机种子
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 输出目录
        self.output_dir = Path(config['output']['output_dir']) / config['output']['experiment_name']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_path = self.output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Config saved to {config_path}")
        
        # 初始化模型
        print("\nInitializing model...")
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 打印参数摘要
        self.model.print_parameter_summary()
        
        # 初始化数据加载器
        print("\nInitializing data loaders...")
        self.train_loader = self._create_train_loader()
        self.val_loader = self._create_val_loader()
        
        # 初始化优化器和调度器
        print("\nInitializing optimizer and scheduler...")
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 训练状态
        self.global_step = 0
        self.best_metric = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
    
    def _create_model(self):
        """创建模型"""
        model_config = self.config['model']
        
        vision_encoder_config = model_config.get('vision_encoder', None)
        freeze_vision = model_config.get('freeze_vision_encoder', False)
        
        model = create_florence2_model(
            base_model=model_config['base_model'],
            vision_encoder_config=vision_encoder_config,
            freeze_vision_encoder=freeze_vision,
            max_length=model_config.get('text_decoder', {}).get('max_length', 256)
        )
        
        return model
    
    def _create_train_loader(self):
        """创建训练数据加载器"""
        train_config = self.config['data']
        transform_config = train_config.get('transforms', {})
        
        transform = create_transform(
            image_size=transform_config.get('image_size', 384),
            is_training=True,
            use_augmentation=True,
            config=transform_config
        )
        
        loader = create_ocr_dataloader(
            image_dir=train_config['train_data']['image_dir'],
            annotation_file=train_config['train_data']['annotation_file'],
            tokenizer=self.model.processor.tokenizer,
            processor=self.model.processor,
            transform=transform,
            batch_size=train_config['dataloader']['batch_size'],
            num_workers=train_config['dataloader']['num_workers'],
            shuffle=train_config['dataloader']['shuffle'],
            task_prompt=self.config['ocr']['task_prompt'],
            max_length=self.config['model']['text_decoder']['max_length'],
            pin_memory=train_config['dataloader']['pin_memory']
        )
        
        return loader
    
    def _create_val_loader(self):
        """创建验证数据加载器"""
        val_config = self.config['data']
        transform_config = val_config.get('transforms', {})
        
        transform = create_transform(
            image_size=transform_config.get('image_size', 384),
            is_training=False,
            use_augmentation=False,
            config=transform_config
        )
        
        loader = create_ocr_dataloader(
            image_dir=val_config['val_data']['image_dir'],
            annotation_file=val_config['val_data']['annotation_file'],
            tokenizer=self.model.processor.tokenizer,
            processor=self.model.processor,
            transform=transform,
            batch_size=val_config['dataloader']['batch_size'],
            num_workers=val_config['dataloader']['num_workers'],
            shuffle=False,
            task_prompt=self.config['ocr']['task_prompt'],
            max_length=self.config['model']['text_decoder']['max_length'],
            pin_memory=val_config['dataloader']['pin_memory']
        )
        
        return loader
    
    def _create_optimizer(self):
        """创建优化器"""
        opt_config = self.config['training']['optimizer']
        
        # 获取可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if opt_config['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=tuple(opt_config.get('betas', [0.9, 0.999]))
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        sched_config = self.config['training']['scheduler']
        total_steps = len(self.train_loader) * self.config['training']['epochs']
        
        if sched_config['type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=sched_config.get('min_lr_ratio', 0.1) * self.config['training']['optimizer']['lr']
            )
        elif sched_config['type'] == 'linear':
            warmup_steps = int(sched_config.get('warmup_ratio', 0.1) * total_steps)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch.get('input_ids', None)
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get('labels', None)
            if labels is not None:
                labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 计算损失
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # 梯度累积
            loss = loss / self.config['training']['gradient_accumulation_steps']
            loss.backward()
            
            # 更新权重
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                # 梯度裁剪
                if self.config['training'].get('grad_clip', None):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.config['training']['gradient_accumulation_steps']
            num_batches += 1
            
            # 更新进度条
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # 日志记录
            if self.global_step % self.config['training']['log_steps'] == 0:
                self._log_training_info(epoch, batch_idx, avg_loss)
            
            # 评估
            if self.global_step % self.config['training']['eval_steps'] == 0:
                self.evaluate()
            
            # 保存检查点
            if self.global_step % self.config['training']['save_steps'] == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}')
        
        avg_train_loss = total_loss / max(num_batches, 1)
        return avg_train_loss
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """验证评估"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        predictions = []
        ground_truths = []
        
        progress_bar = tqdm(self.val_loader, desc="[Eval]")
        
        for batch in progress_bar:
            # 移动数据到设备
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch.get('input_ids', None)
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get('labels', None)
            if labels is not None:
                labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            total_loss += loss.item()
            num_batches += 1
            
            # 收集预测和真实值（用于计算指标）
            ground_truths.extend(batch.get('ground_truth_texts', []))
        
        avg_val_loss = total_loss / max(num_batches, 1)
        
        # 计算 OCR 指标
        metrics = self._compute_ocr_metrics(predictions, ground_truths)
        metrics['val_loss'] = avg_val_loss
        
        # 更新最佳指标
        if self.best_metric is None or metrics.get('accuracy', 0) > self.best_metric:
            self.best_metric = metrics.get('accuracy', 0)
            self.save_checkpoint('best_model')
        
        self.training_history['val_loss'].append(avg_val_loss)
        self.training_history['val_metrics'].append(metrics)
        
        print(f"\nValidation Results:")
        print(f"  Loss: {avg_val_loss:.4f}")
        for key, value in metrics.items():
            if key != 'val_loss':
                print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def _compute_ocr_metrics(
        self,
        predictions: list,
        ground_truths: list
    ) -> Dict[str, float]:
        """计算 OCR 评估指标"""
        try:
            import editdistance
        except ImportError:
            return {'accuracy': 0.0}
        
        if len(predictions) == 0 or len(ground_truths) == 0:
            return {'accuracy': 0.0}
        
        # 精确匹配准确率
        exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
        accuracy = exact_matches / len(ground_truths)
        
        # 编辑距离
        total_ed = sum(editdistance.eval(p.lower(), g.lower()) for p, g in zip(predictions, ground_truths))
        avg_ed = total_ed / len(ground_truths)
        
        # 归一化编辑距离
        total_len = sum(len(g) for g in ground_truths)
        ned = total_ed / total_len if total_len > 0 else 0
        
        return {
            'accuracy': accuracy,
            'avg_edit_distance': avg_ed,
            'normalized_edit_distance': ned
        }
    
    def _log_training_info(self, epoch: int, batch_idx: int, loss: float):
        """记录训练信息"""
        lr = self.optimizer.param_groups[0]['lr']
        
        log_info = {
            'epoch': epoch + 1,
            'step': self.global_step,
            'loss': loss,
            'lr': lr
        }
        
        self.training_history['train_loss'].append(loss)
        
        # 打印日志
        print(f"\nStep {self.global_step}:")
        print(f"  Epoch: {epoch + 1}")
        print(f"  Loss: {loss:.4f}")
        print(f"  LR: {lr:.6f}")
    
    def save_checkpoint(self, name: str = 'checkpoint'):
        """保存检查点"""
        checkpoint_path = self.output_dir / f'{name}.pt'
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """开始训练"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        epochs = self.config['training']['epochs']
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch + 1} Training Loss: {train_loss:.4f}")
            
            # 保存每个 epoch 的检查点
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}')
        
        # 保存最终模型
        self.save_checkpoint('final_model')
        
        # 保存训练历史
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\nTraining completed! Best metric: {self.best_metric}")
        print(f"Model saved to {self.output_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Florence-2 OCR Fine-tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器
    trainer = OCRTrainer(config)
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()

"""
数据预处理和增强模块

提供 OCR 任务专用的图像变换和增强功能
"""

from typing import Optional, List, Tuple, Any
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import random
import cv2
import numpy as np


class OCRTransform:
    """
    OCR 图像变换类
    
    提供针对 OCR 任务的图像预处理和增强功能
    """
    
    def __init__(
        self,
        image_size: int = 384,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augmentations: Optional[dict] = None
    ):
        """
        初始化 OCR 变换
        
        Args:
            image_size: 图像尺寸
            mean: 归一化均值
            std: 归一化标准差
            augmentations: 增强配置
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augmentations = augmentations or {}
        
        # 构建基础变换
        self.transform = self._build_transform()
    
    def _build_transform(self) -> T.Compose:
        """构建变换流水线"""
        transforms_list = []
        
        # 调整大小
        transforms_list.append(T.Resize((self.image_size, self.image_size)))
        
        # 数据增强（训练时使用）
        if self.augmentations.get('random_resize_crop', False):
            transforms_list.append(
                T.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                )
            )
        
        if self.augmentations.get('color_jitter', False):
            transforms_list.append(
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                )
            )
        
        if self.augmentations.get('random_flip', False):
            # OCR 任务通常不使用水平翻转，因为会改变文字顺序
            # 这里仅作为选项提供
            pass
        
        # 转换为 tensor
        transforms_list.append(T.ToTensor())
        
        # 归一化
        if self.augmentations.get('normalize', True):
            transforms_list.append(T.Normalize(mean=self.mean, std=self.std))
        
        return T.Compose(transforms_list)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """应用变换"""
        return self.transform(image)
    
    def __repr__(self) -> str:
        return f"OCRTransform(image_size={self.image_size}, augmentations={self.augmentations})"


class OCRAugmentation:
    """
    OCR 专用数据增强
    
    包含针对文本识别的特殊增强技术
    """
    
    @staticmethod
    def random_erode(
        image: np.ndarray,
        prob: float = 0.1,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        随机腐蚀操作，模拟低质量文本
        
        Args:
            image: 输入图像 (H, W, C)
            prob: 应用概率
            kernel_size: 腐蚀核大小
            
        Returns:
            处理后的图像
        """
        if random.random() > prob:
            return image
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(image, kernel, iterations=1)
        
        return eroded
    
    @staticmethod
    def random_dilate(
        image: np.ndarray,
        prob: float = 0.1,
        kernel_size: int = 2
    ) -> np.ndarray:
        """
        随机膨胀操作，模拟粗体文本
        
        Args:
            image: 输入图像 (H, W, C)
            prob: 应用概率
            kernel_size: 膨胀核大小
            
        Returns:
            处理后的图像
        """
        if random.random() > prob:
            return image
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)
        
        return dilated
    
    @staticmethod
    def random_motion_blur(
        image: np.ndarray,
        prob: float = 0.1,
        max_kernel_size: int = 10
    ) -> np.ndarray:
        """
        随机运动模糊，模拟相机抖动
        
        Args:
            image: 输入图像 (H, W, C)
            prob: 应用概率
            max_kernel_size: 最大模糊核大小
            
        Returns:
            处理后的图像
        """
        if random.random() > prob:
            return image
        
        kernel_size = random.randint(3, max_kernel_size)
        angle = random.uniform(0, 360)
        
        # 创建运动模糊核
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            y = int(center + (i - center) * np.sin(np.radians(angle)))
            x = int(center + (i - center) * np.cos(np.radians(angle)))
            if 0 <= y < kernel_size and 0 <= x < kernel_size:
                kernel[y, x] = 1
        
        kernel /= kernel.sum()
        
        # 应用模糊
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    @staticmethod
    def random_gaussian_noise(
        image: np.ndarray,
        prob: float = 0.1,
        sigma: float = 25
    ) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            image: 输入图像 (H, W, C)
            prob: 应用概率
            sigma: 噪声标准差
            
        Returns:
            处理后的图像
        """
        if random.random() > prob:
            return image
        
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    @staticmethod
    def random_contrast(
        image: np.ndarray,
        prob: float = 0.2,
        alpha_range: Tuple[float, float] = (0.5, 1.5)
    ) -> np.ndarray:
        """
        随机对比度调整
        
        Args:
            image: 输入图像 (H, W, C)
            prob: 应用概率
            alpha_range: 对比度因子范围
            
        Returns:
            处理后的图像
        """
        if random.random() > prob:
            return image
        
        alpha = random.uniform(*alpha_range)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        return adjusted
    
    @staticmethod
    def random_rotation(
        image: np.ndarray,
        prob: float = 0.1,
        max_angle: float = 10
    ) -> np.ndarray:
        """
        随机小角度旋转，模拟拍摄角度偏差
        
        Args:
            image: 输入图像 (H, W, C)
            prob: 应用概率
            max_angle: 最大旋转角度
            
        Returns:
            处理后的图像
        """
        if random.random() > prob:
            return image
        
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated


class AugmentedOCRTransform:
    """
    带增强的 OCR 变换类
    
    结合基础变换和 OCR 专用增强
    """
    
    def __init__(
        self,
        image_size: int = 384,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augmentation_prob: float = 0.3,
        is_training: bool = True
    ):
        """
        初始化带增强的 OCR 变换
        
        Args:
            image_size: 图像尺寸
            mean: 归一化均值
            std: 归一化标准差
            augmentation_prob: 增强应用概率
            is_training: 是否为训练模式
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augmentation_prob = augmentation_prob
        self.is_training = is_training
        
        # 基础变换
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    
    def apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """应用各种增强"""
        if not self.is_training:
            return image
        
        # 随机应用增强
        if random.random() < self.augmentation_prob:
            aug_type = random.choice([
                'contrast', 'rotation', 'noise', 'erode', 'dilate'
            ])
            
            if aug_type == 'contrast':
                image = OCRAugmentation.random_contrast(
                    image, prob=1.0, alpha_range=(0.7, 1.3)
                )
            elif aug_type == 'rotation':
                image = OCRAugmentation.random_rotation(
                    image, prob=1.0, max_angle=15
                )
            elif aug_type == 'noise':
                image = OCRAugmentation.random_gaussian_noise(
                    image, prob=1.0, sigma=20
                )
            elif aug_type == 'erode':
                image = OCRAugmentation.random_erode(
                    image, prob=1.0, kernel_size=2
                )
            elif aug_type == 'dilate':
                image = OCRAugmentation.random_dilate(
                    image, prob=1.0, kernel_size=2
                )
        
        return image
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """应用变换"""
        # 转换为 numpy 数组
        image_np = np.array(image)
        
        # 应用增强
        if self.is_training:
            image_np = self.apply_augmentations(image_np)
        
        # 转回 PIL Image
        image_pil = Image.fromarray(image_np)
        
        # 应用基础变换
        return self.base_transform(image_pil)
    
    def __repr__(self) -> str:
        mode = "training" if self.is_training else "evaluation"
        return f"AugmentedOCRTransform(image_size={self.image_size}, mode={mode})"


def create_transform(
    image_size: int = 384,
    is_training: bool = True,
    use_augmentation: bool = True,
    config: Optional[dict] = None
) -> Any:
    """
    创建变换对象
    
    Args:
        image_size: 图像尺寸
        is_training: 是否为训练模式
        use_augmentation: 是否使用增强
        config: 配置字典
        
    Returns:
        变换对象
    """
    if config is None:
        config = {}
    
    if use_augmentation and is_training:
        return AugmentedOCRTransform(
            image_size=image_size,
            augmentation_prob=config.get('augmentation_prob', 0.3),
            is_training=True
        )
    else:
        return OCRTransform(
            image_size=image_size,
            augmentations=config.get('augmentations', {})
        )


def get_default_transform_config() -> dict:
    """获取默认变换配置"""
    return {
        'image_size': 384,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'augmentations': {
            'random_resize_crop': True,
            'color_jitter': False,
            'random_flip': False,
            'normalize': True
        },
        'augmentation_prob': 0.3
    }


if __name__ == "__main__":
    # 测试变换
    print("Testing OCR Transforms...")
    
    # 创建测试图像
    test_image = Image.new('RGB', (512, 512), color=(255, 255, 255))
    
    # 测试基础变换
    try:
        transform = OCRTransform(
            image_size=384,
            augmentations={'normalize': True}
        )
        result = transform(test_image)
        print(f"✓ Base transform: {result.shape}")
    except Exception as e:
        print(f"✗ Base transform failed: {e}")
    
    # 测试增强变换
    try:
        aug_transform = AugmentedOCRTransform(
            image_size=384,
            is_training=True,
            augmentation_prob=0.5
        )
        result = aug_transform(test_image)
        print(f"✓ Augmented transform: {result.shape}")
    except Exception as e:
        print(f"✗ Augmented transform failed: {e}")
    
    # 测试工厂函数
    try:
        transform = create_transform(
            image_size=384,
            is_training=True,
            use_augmentation=True
        )
        result = transform(test_image)
        print(f"✓ Factory transform: {result.shape}")
    except Exception as e:
        print(f"✗ Factory transform failed: {e}")
    
    print("\nTransform tests completed!")

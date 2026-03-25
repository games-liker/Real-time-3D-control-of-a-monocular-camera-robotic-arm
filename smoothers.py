"""
平滑器类

包含用于实时数据平滑的类：
- RealtimeSmoother3D: 3D位置平滑器
- AreaDepthSmoother: 面积深度平滑器
"""
import numpy as np
from typing import Optional


class RealtimeSmoother3D:
    """实时平滑器（3D版本）"""
    
    def __init__(
        self,
        alpha: float = 0.3,
        outlier_threshold: float = 0.2,
        depth_outlier_threshold: float = 0.1,
    ):
        """
        Args:
            alpha: 平滑系数，越小越平滑
            outlier_threshold: 2D位置异常值阈值
            depth_outlier_threshold: 深度异常值阈值
        """
        self.alpha = alpha
        self.outlier_threshold = outlier_threshold
        self.depth_outlier_threshold = depth_outlier_threshold
        self.smoothed_last: Optional[np.ndarray] = None
        
    def smooth(self, current_pos: np.ndarray) -> np.ndarray:
        """
        平滑3D位置
        
        Args:
            current_pos: 当前3D位置
            
        Returns:
            平滑后的3D位置
        """
        current_pos = np.array(current_pos)
        
        if self.smoothed_last is not None:
            distance_2d = np.linalg.norm(current_pos[:2] - self.smoothed_last[:2])
            depth_diff = abs(current_pos[2] - self.smoothed_last[2])
            
            if distance_2d > self.outlier_threshold or depth_diff > self.depth_outlier_threshold:
                current_pos = self.smoothed_last * 0.7 + current_pos * 0.3
        
        if self.smoothed_last is None:
            smoothed = current_pos
        else:
            smoothed = self.alpha * current_pos + (1 - self.alpha) * self.smoothed_last
        
        self.smoothed_last = smoothed.copy()
        return smoothed
    
    def reset(self):
        """重置平滑器状态"""
        self.smoothed_last = None


class AreaDepthSmoother:
    """面积深度平滑器 - 专门用于平滑面积估计的深度值"""
    
    def __init__(self, alpha: float = 0.2, outlier_ratio: float = 2.0):
        """
        Args:
            alpha: 平滑系数，越小越平滑
            outlier_ratio: 异常值阈值比率，超过此比率的变化视为异常
        """
        self.alpha = alpha
        self.outlier_ratio = outlier_ratio
        self.last_area_ratio: Optional[float] = None
        self.smoothed_area_ratio: Optional[float] = None
        
    def smooth(self, area_ratio: float) -> float:
        """
        平滑面积占比
        
        Args:
            area_ratio: 当前面积占比
            
        Returns:
            平滑后的面积占比
        """
        if self.smoothed_area_ratio is None:
            self.smoothed_area_ratio = area_ratio
            self.last_area_ratio = area_ratio
            return area_ratio
        
        # 检测异常值
        if self.last_area_ratio > 0:
            change_ratio = area_ratio / self.last_area_ratio
            if change_ratio > self.outlier_ratio or change_ratio < 1 / self.outlier_ratio:
                # 异常值，使用更小的权重
                area_ratio = self.smoothed_area_ratio * 0.8 + area_ratio * 0.2
        
        # 指数平滑
        self.smoothed_area_ratio = self.alpha * area_ratio + (1 - self.alpha) * self.smoothed_area_ratio
        self.last_area_ratio = area_ratio
        
        return self.smoothed_area_ratio
    
    def reset(self):
        """重置平滑器状态"""
        self.last_area_ratio = None
        self.smoothed_area_ratio = None


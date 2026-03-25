"""
数据结构定义

包含所有用于进程间通信的数据结构
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """检测结果数据结构（不包含图像）"""
    joint_pos: Optional[np.ndarray]
    wrist_2d_pos: Optional[np.ndarray]
    wrist_rot: Optional[np.ndarray]
    palm_landmarks_2d: Optional[np.ndarray]  # 手掌多边形的2D像素坐标
    frame_id: int
    timestamp: float = 0.0


@dataclass
class DepthResult:
    """基于面积的深度估计结果"""
    wrist_depth: Optional[float]  # 映射后的深度值
    palm_area_ratio: Optional[float]  # 手掌面积占比 (0-1)
    palm_area_pixels: Optional[float]  # 手掌面积（像素平方）
    frame_id: int
    timestamp: float = 0.0


@dataclass
class IKResult:
    """IK求解结果数据结构"""
    full_qpos: np.ndarray
    joint_angles: dict
    ee_position: Optional[np.ndarray]
    target_position: Optional[np.ndarray]
    frame_id: int
    timestamp: float = 0.0


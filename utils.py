"""
工具函数

包含：
- 手掌面积计算
- 面积到深度的映射
- 机械臂组合体信息获取
"""
import numpy as np
from pathlib import Path
# 处理导入（支持直接运行和模块导入）
try:
    from .config import ARM_HAND_ASSEMBLY_MAP
except ImportError:
    from config import ARM_HAND_ASSEMBLY_MAP


def compute_polygon_area(vertices: np.ndarray) -> float:
    """
    使用鞋带公式计算多边形面积
    
    Args:
        vertices: 多边形顶点坐标，形状为 (N, 2)
        
    Returns:
        多边形面积
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    
    # 鞋带公式
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    
    return abs(area) / 2.0


def area_ratio_to_depth(area_ratio: float, 
                        min_area_ratio: float = 0.005,
                        max_area_ratio: float = 0.15,
                        min_depth: float = 0.05,
                        max_depth: float = 0.4) -> float:
    """
    将手掌面积占比映射到深度值
    
    面积越大 -> 手越近相机 -> 深度值越小
    面积越小 -> 手越远相机 -> 深度值越大
    
    Args:
        area_ratio: 手掌面积占整个图像的比例 (0-1)
        min_area_ratio: 预期的最小面积占比（手最远时）
        max_area_ratio: 预期的最大面积占比（手最近时）
        min_depth: 最小深度值（手最近时）
        max_depth: 最大深度值（手最远时）
        
    Returns:
        估计的深度值
    """
    # 限制面积占比范围
    area_ratio_clamped = np.clip(area_ratio, min_area_ratio, max_area_ratio)
    
    # 归一化到 0-1
    normalized = (area_ratio_clamped - min_area_ratio) / (max_area_ratio - min_area_ratio)
    
    # 面积越大，深度越小（反比关系）
    depth = max_depth - normalized * (max_depth - min_depth)
    
    return depth


def get_assembly_info(robot_name: str, hand_type: str):
    """
    根据机器人名称和手型获取组合体信息
    
    Args:
        robot_name: 机器人名称
        hand_type: 手型 ("Left" 或 "Right")
        
    Returns:
        (assembly_dir, urdf_file, scale) 元组
    """
    for key, info in ARM_HAND_ASSEMBLY_MAP.items():
        if key in robot_name.lower():
            urdf_file = info["left_urdf"] if hand_type.lower() == "left" else info["right_urdf"]
            return info["assembly_dir"], urdf_file, info["scale"]
    return None, None, 1.0


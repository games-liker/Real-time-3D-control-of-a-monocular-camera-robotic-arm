"""
配置和常量定义

包含：
- 手掌多边形关节点索引
- 多视角相机配置
- 视角布局预设
- 机械臂组合映射表
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict


# ==================== 手掌多边形关节点索引 ====================
# MediaPipe手部关键点索引
# 手掌多边形由以下关节点循环连接形成：
# - 0: Wrist（手腕）
# - 1: Thumb_CMC（拇指掌腕关节）
# - 5: Index_Finger_MCP（食指掌指关节）
# - 9: Middle_Finger_MCP（中指掌指关节）
# - 13: Ring_Finger_MCP（无名指掌指关节）
# - 17: Pinky_MCP（小指掌指关节）
PALM_POLYGON_INDICES = [0, 1, 5, 9, 13, 17]


# ==================== 多视角相机配置 ====================

@dataclass
class CameraConfig:
    """相机配置"""
    name: str
    position: np.ndarray
    quaternion: np.ndarray  # [w, x, y, z] -> SAPIEN uses [x, y, z, w]
    width: int = 400
    height: int = 400
    fovy: float = 1.0


# 预定义的多视角相机配置
MULTI_VIEW_CAMERAS: Dict[str, CameraConfig] = {
    "front": CameraConfig(
        name="front_camera",
        position=np.array([1.5, 0.0, 0.4]),
        quaternion=np.array([0, 0, 0, 1]),  # 正对机器人
        width=800,
        height=800,
    ),
    "side_right": CameraConfig(
        name="side_right_camera",
        position=np.array([0.0, -0.6, 0.3]),
        quaternion=np.array([-0.5, -0.5, -0.5, 0.5]),  # 从右侧看
        width=800,
        height=800,
    ),
    "side_left": CameraConfig(
        name="side_left_camera",
        position=np.array([0.0, 0.6, 0.3]),
        quaternion=np.array([0.5, -0.5, 0.5, 0.5]),  # 从左侧看
        width=800,
        height=800,
    ),
    "top": CameraConfig(
        name="top_camera",
        position=np.array([0.4, 0.0, 1.5]),
        quaternion=np.array([0.7071, 0, 0.7071, 0]),  # 从上方看
        width=800,
        height=800,
    ),
    "oblique": CameraConfig(
        name="oblique_camera",
        position=np.array([0.5, 0.5, 0.5]),
        quaternion=np.array([-0.3536, -0.3536, -0.6124, 0.6124]),  # 斜视角
        width=800,
        height=800,
    ),
}

# 视角布局预设
VIEW_LAYOUTS: Dict[str, list] = {
    "2x2": ["front", "side_right", "top", "oblique"],  # 2行2列
    "1x4": ["front", "side_right", "side_left", "top"],  # 1行4列
    "2x1": ["front", "top"],  # 2列1行
    "1x3": ["front", "side_right", "top"],  # 1行3列
    "single": ["front"],  # 单视角
}


# ==================== 机械臂组合映射表 ====================

ARM_HAND_ASSEMBLY_MAP: Dict[str, Dict] = {
    "ability": {
        "assembly_dir": "xarm7_ability",
        "left_urdf": "xarm7_ability_left_hand_glb.urdf",
        "right_urdf": "xarm7_ability_right_hand_glb.urdf",
        "scale": 1.0,
    },
    "shadow": {
        "assembly_dir": "ur5e_shadow",
        "left_urdf": "ur5e_shadow_left_hand_glb.urdf",
        "right_urdf": "ur5e_shadow_right_hand_glb.urdf",
        "scale": 1.0,
    },
    "inspire": {
        "assembly_dir": "rm75_inspire",
        "left_urdf": "rm75_inspire_left_hand.urdf",
        "right_urdf": "rm75_inspire_right_hand.urdf",
        "scale": 1.0,
    },
}


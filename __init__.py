"""
3D实时重定向演示模块

模块化设计的3D实时重定向系统，基于手掌面积估计深度。

主要模块：
- config: 配置和常量
- data_structures: 数据结构定义
- utils: 工具函数
- smoothers: 平滑器类
- camera_manager: 多视角相机管理器
- producer: 图像采集进程
- detection: 手部检测进程
- depth_estimation: 深度估计进程
- ik_solver: IK求解进程
- renderer: 渲染进程
- main: 主入口
"""

__version__ = "1.0.0"

from .config import (
    PALM_POLYGON_INDICES,
    CameraConfig,
    MULTI_VIEW_CAMERAS,
    VIEW_LAYOUTS,
    ARM_HAND_ASSEMBLY_MAP,
)

from .data_structures import (
    DetectionResult,
    DepthResult,
    IKResult,
)

from .utils import (
    compute_polygon_area,
    area_ratio_to_depth,
    get_assembly_info,
)

from .smoothers import (
    RealtimeSmoother3D,
    AreaDepthSmoother,
)

from .camera_manager import (
    MultiViewCameraManager,
)

__all__ = [
    # 配置
    "PALM_POLYGON_INDICES",
    "CameraConfig",
    "MULTI_VIEW_CAMERAS",
    "VIEW_LAYOUTS",
    "ARM_HAND_ASSEMBLY_MAP",
    # 数据结构
    "DetectionResult",
    "DepthResult",
    "IKResult",
    # 工具函数
    "compute_polygon_area",
    "area_ratio_to_depth",
    "get_assembly_info",
    # 平滑器
    "RealtimeSmoother3D",
    "AreaDepthSmoother",
    # 相机管理器
    "MultiViewCameraManager",
]


"""
3D实时重定向演示（基于手掌面积估计深度）- 主入口

与 benchmark_trajectory_3d_area_multiview.py 的主要区别：
- 纯操控模式，不做benchmark
- 队列不传输骨架图片，只传输必要数据
- 支持多视角渲染（可选）

手掌多边形由以下关节点循环连接形成：
- 0: Wrist（手腕）
- 1: Thumb_CMC（拇指掌腕关节）
- 5: Index_Finger_MCP（食指掌指关节）
- 9: Middle_Finger_MCP（中指掌指关节）
- 13: Ring_Finger_MCP（无名指掌指关节）
- 17: Pinky_MCP（小指掌指关节）
"""
import multiprocessing
import sys
from pathlib import Path
from typing import Optional

import tyro
from loguru import logger

# 处理直接运行时的导入问题
if __name__ == "__main__":
    # 直接运行时，添加当前目录到sys.path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    # 添加项目根目录到sys.path
    project_root = current_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)

# 导入本地模块（直接导入，不使用processes.py）
from producer import producer_process
from detection import detection_process
from depth_estimation import depth_estimation_process
from ik_solver import ik_process_3d
from renderer import render_process_multiview
from config import VIEW_LAYOUTS


def main(
    robot_name: RobotName = RobotName.ability,
    retargeting_type: RetargetingType = RetargetingType.dexpilot,
    hand_type: HandType = HandType.right,
    camera_path: Optional[str] = None,
    # 面积深度估计参数
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.15,
    min_depth: float = 0.01,
    max_depth: float = 0.4,
    # 平滑参数
    enable_smoothing: bool = True,
    smoothing_alpha: float = 0.3,
    area_smoothing_alpha: float = 0.2,
    # 多视角参数
    view_layout: str = "2x1",
    show_multiview: bool = True,
    # 其他参数
    image_scale_factor: float = 1,
    queue_size: int = 100,
):
    """
    3D实时重定向演示（基于手掌面积估计深度）
    
    Args:
        robot_name: 机器人标识符
        retargeting_type: 重定向类型
        hand_type: 手型（左手/右手）
        camera_path: 摄像头路径
        min_area_ratio: 最小面积占比（手最远时）
        max_area_ratio: 最大面积占比（手最近时）
        min_depth: 最小深度值（手最近时）
        max_depth: 最大深度值（手最远时）
        enable_smoothing: 是否启用平滑
        smoothing_alpha: 3D位置平滑系数
        area_smoothing_alpha: 面积平滑系数
        view_layout: 视角布局 ("2x2", "1x4", "2x1", "1x3", "single")
        show_multiview: 是否显示多视角窗口
        image_scale_factor: 图像缩放因子
        queue_size: 队列大小
        
    使用示例：
        # 直接运行（推荐）
        cd example/3d_retargeting
        python main.py
        
        # 启用多视角显示
        python main.py --show-multiview --view-layout 2x2
        
        # 自定义深度范围
        python main.py --min-depth 0.05 --max-depth 0.5
    """
    logger.info("=" * 60)
    logger.info("3D实时重定向演示（基于手掌面积估计深度）")
    logger.info("=" * 60)
    
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    
    # 创建队列
    queue_img = multiprocessing.Queue(maxsize=queue_size)
    queue_detection = multiprocessing.Queue(maxsize=queue_size)
    queue_depth = multiprocessing.Queue(maxsize=queue_size)
    queue_ik = multiprocessing.Queue(maxsize=queue_size)
    
    stop_event = multiprocessing.Event()
    
    hand_type_str = "Right" if hand_type == HandType.right else "Left"
    
    processes = [
        multiprocessing.Process(
            target=producer_process,
            args=(queue_img, stop_event, camera_path, 30.0),
            name="Producer"
        ),
        multiprocessing.Process(
            target=detection_process,
            args=(queue_img, queue_detection, stop_event, hand_type_str),
            name="Detection"
        ),
        multiprocessing.Process(
            target=depth_estimation_process,
            args=(
                queue_detection, queue_depth, stop_event,
                640, 480,  # 默认图像尺寸
                min_area_ratio, max_area_ratio,
                min_depth, max_depth,
                enable_smoothing, area_smoothing_alpha,
            ),
            name="DepthEstimation"
        ),
        multiprocessing.Process(
            target=ik_process_3d,
            args=(
                queue_depth, queue_ik, stop_event,
                str(robot_dir), str(config_path),
                True,  # use_arm
                True,  # enable_arm_ik
                enable_smoothing,
                smoothing_alpha,
                0.1,  # fallback_y_position
                image_scale_factor,
                True,  # enable_orientation_control
                0.1,   # orientation_weight
            ),
            name="IK"
        ),
        multiprocessing.Process(
            target=render_process_multiview,
            args=(
                queue_ik, stop_event,
                str(robot_dir), str(config_path),
                True, image_scale_factor,
                view_layout, show_multiview,
            ),
            name="Render"
        ),
    ]
    
    logger.info("启动5个进程...")
    for p in processes:
        p.start()
        logger.info(f"  {p.name} 进程已启动 (PID: {p.pid})")
    
    logger.info("=" * 60)
    logger.info(f"深度估计方法: 手掌面积占比")
    if show_multiview:
        logger.info(f"视角布局: {view_layout}")
        logger.info(f"视角列表: {VIEW_LAYOUTS.get(view_layout, VIEW_LAYOUTS['single'])}")
    logger.info(f"面积范围: [{min_area_ratio:.4f}, {max_area_ratio:.4f}]")
    logger.info(f"深度范围: [{min_depth:.2f}, {max_depth:.2f}]")
    logger.info("=" * 60)
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止所有进程...")
        stop_event.set()
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    
    logger.info("所有进程已结束")


if __name__ == "__main__":
    tyro.cli(main)


"""
深度估计进程

基于手掌面积计算深度值
"""
import multiprocessing
from queue import Empty

from loguru import logger

# 处理导入（支持直接运行和模块导入）
try:
    from .data_structures import DetectionResult, DepthResult
    from .utils import compute_polygon_area, area_ratio_to_depth
    from .smoothers import AreaDepthSmoother
except ImportError:
    from data_structures import DetectionResult, DepthResult
    from utils import compute_polygon_area, area_ratio_to_depth
    from smoothers import AreaDepthSmoother


def depth_estimation_process(
    queue_in: multiprocessing.Queue,
    queue_out: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    image_width: int = 640,
    image_height: int = 480,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.15,
    min_depth: float = 0.05,
    max_depth: float = 0.4,
    enable_smoothing: bool = True,
    smoothing_alpha: float = 0.2,
):
    """
    深度估计进程：基于手掌面积计算深度
    
    Args:
        queue_in: 输入队列，接收 DetectionResult 对象
        queue_out: 输出队列，发送 (DetectionResult, DepthResult) 元组
        stop_event: 停止事件
        image_width: 图像宽度
        image_height: 图像高度
        min_area_ratio: 最小面积占比（手最远时）
        max_area_ratio: 最大面积占比（手最近时）
        min_depth: 最小深度值（手最近时）
        max_depth: 最大深度值（手最远时）
        enable_smoothing: 是否启用平滑
        smoothing_alpha: 面积平滑系数
    """
    logger.info("[深度估计进程] 进程启动")
    logger.info(f"  - 图像尺寸: {image_width}x{image_height}")
    logger.info(f"  - 面积范围: [{min_area_ratio:.4f}, {max_area_ratio:.4f}]")
    logger.info(f"  - 深度范围: [{min_depth:.2f}, {max_depth:.2f}]")
    
    total_image_area = image_width * image_height
    
    # 面积平滑器
    area_smoother = AreaDepthSmoother(alpha=smoothing_alpha) if enable_smoothing else None
    
    while not stop_event.is_set():
        try:
            detection_result = queue_in.get(timeout=0.1)
        except Empty:
            continue
        
        palm_area_ratio = None
        palm_area_pixels = None
        wrist_depth = None
        
        if detection_result.palm_landmarks_2d is not None:
            # 计算手掌多边形面积
            palm_area_pixels = compute_polygon_area(detection_result.palm_landmarks_2d)
            palm_area_ratio = palm_area_pixels / total_image_area
            
            # 平滑处理
            if area_smoother is not None:
                palm_area_ratio_smoothed = area_smoother.smooth(palm_area_ratio)
            else:
                palm_area_ratio_smoothed = palm_area_ratio
            
            # 面积占比映射到深度
            wrist_depth = area_ratio_to_depth(
                palm_area_ratio_smoothed,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                min_depth=min_depth,
                max_depth=max_depth,
            )
        
        result = DepthResult(
            wrist_depth=wrist_depth,
            palm_area_ratio=palm_area_ratio,
            palm_area_pixels=palm_area_pixels,
            frame_id=detection_result.frame_id,
            timestamp=detection_result.timestamp,
        )
        
        try:
            if queue_out.full():
                try:
                    queue_out.get_nowait()
                except Empty:
                    pass
            queue_out.put((detection_result, result), timeout=0.1)
        except:
            pass
    
    logger.info("[深度估计进程] 进程结束")


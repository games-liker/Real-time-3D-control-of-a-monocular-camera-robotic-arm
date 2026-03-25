"""
手部检测进程

执行MediaPipe手部检测，提取手掌多边形关键点
"""
import multiprocessing
import sys
import cv2
import numpy as np
from pathlib import Path
from queue import Empty

from loguru import logger

# 处理导入（支持直接运行和模块导入）
try:
    from .data_structures import DetectionResult
    from .config import PALM_POLYGON_INDICES
    from .single_hand_detector import SingleHandDetector
except ImportError:
    from data_structures import DetectionResult
    from config import PALM_POLYGON_INDICES
    from single_hand_detector import SingleHandDetector


def detection_process(
    queue_in: multiprocessing.Queue,
    queue_out: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    hand_type: str = "Right",
):
    """
    检测进程：执行MediaPipe手部检测，提取手掌多边形关键点（不传输图像）
    
    Args:
        queue_in: 输入队列，接收 (bgr_image, frame_id, timestamp) 元组
        queue_out: 输出队列，发送 DetectionResult 对象
        stop_event: 停止事件
        hand_type: 手型 ("Right" 或 "Left")
    """
    logger.info("[检测进程] 进程启动")
    
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)
    
    while not stop_event.is_set():
        try:
            bgr, frame_id, timestamp = queue_in.get(timeout=1.0)
        except Empty:
            continue
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = bgr.shape[:2]
        
        _, joint_pos, keypoint_2d, wrist_rot = detector.detect(rgb)
        
        wrist_2d_pos = None
        palm_landmarks_2d = None
        
        if keypoint_2d is not None:
            # 提取手腕位置（归一化坐标）
            wrist_normalized = keypoint_2d.landmark[0]
            wrist_2d_pos = np.array([
                (wrist_normalized.x - 0.5) * 2,
                (wrist_normalized.y - 0.5) * 2,
            ])
            
            # 提取手掌多边形关键点的像素坐标
            palm_landmarks_2d = np.zeros((len(PALM_POLYGON_INDICES), 2))
            for i, idx in enumerate(PALM_POLYGON_INDICES):
                landmark = keypoint_2d.landmark[idx]
                # 转换为像素坐标
                palm_landmarks_2d[i, 0] = landmark.x * img_w
                palm_landmarks_2d[i, 1] = landmark.y * img_h
        
        # 不传输图像，只传输检测结果
        result = DetectionResult(
            joint_pos=joint_pos,
            wrist_2d_pos=wrist_2d_pos,
            wrist_rot=wrist_rot,
            palm_landmarks_2d=palm_landmarks_2d,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        
        try:
            if queue_out.full():
                try:
                    queue_out.get_nowait()
                except Empty:
                    pass
            queue_out.put(result, timeout=0.1)
        except:
            pass
    
    logger.info("[检测进程] 进程结束")


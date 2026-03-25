"""
图像采集进程

从摄像头读取图像帧并发送到队列
"""
import multiprocessing
import time
import cv2
from queue import Empty
from typing import Optional

from loguru import logger


def producer_process(
    queue_out: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    camera_path: Optional[str] = None,
    target_fps: float = 30.0,
):
    """
    生产者进程：从摄像头读取图像帧
    
    Args:
        queue_out: 输出队列，发送 (image, frame_id, timestamp) 元组
        stop_event: 停止事件
        camera_path: 摄像头路径，None表示使用默认摄像头
        target_fps: 目标帧率
    """
    logger.info("[生产者] 进程启动")
    
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)
    
    if not cap.isOpened():
        logger.error("[生产者] 无法打开摄像头")
        return
    
    frame_id = 0
    frame_interval = 1.0 / target_fps
    start_time = time.time()
    
    while not stop_event.is_set():
        loop_start = time.time()
        
        success, image = cap.read()
        if not success:
            continue
        
        timestamp = time.time() - start_time
        frame_data = (image, frame_id, timestamp)
        
        try:
            if queue_out.full():
                try:
                    queue_out.get_nowait()
                except Empty:
                    pass
            queue_out.put(frame_data, timeout=0.1)
        except:
            pass
        
        frame_id += 1
        
        elapsed = time.time() - loop_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
    
    cap.release()
    logger.info("[生产者] 进程结束")


"""
多视角相机管理器

用于管理多个相机视角的创建、捕获和图像合成
"""
import cv2
import numpy as np
import sapien
from typing import Dict, Optional
from loguru import logger

# 处理导入（支持直接运行和模块导入）
try:
    from .config import MULTI_VIEW_CAMERAS, VIEW_LAYOUTS
except ImportError:
    from config import MULTI_VIEW_CAMERAS, VIEW_LAYOUTS


class MultiViewCameraManager:
    """多视角相机管理器"""
    
    def __init__(self, scene: sapien.Scene, view_layout: str = "single"):
        """
        Args:
            scene: SAPIEN场景
            view_layout: 视角布局，可选 "2x2", "1x4", "2x1", "1x3", "single"
        """
        self.scene = scene
        self.view_layout = view_layout
        self.cameras: Dict[str, sapien.Entity] = {}
        self.view_names = VIEW_LAYOUTS.get(view_layout, VIEW_LAYOUTS["single"])
        
        # 计算布局
        self._calculate_layout()
        
        # 创建相机
        self._create_cameras()
    
    def _calculate_layout(self):
        """计算布局参数"""
        n_views = len(self.view_names)
        
        if self.view_layout == "2x2":
            self.rows, self.cols = 2, 2
        elif self.view_layout == "1x4":
            self.rows, self.cols = 1, 4
        elif self.view_layout == "2x1":
            self.rows, self.cols = 1, 2
        elif self.view_layout == "1x3":
            self.rows, self.cols = 1, 3
        elif self.view_layout == "single":
            self.rows, self.cols = 1, 1
        else:
            # 自动计算
            self.cols = int(np.ceil(np.sqrt(n_views)))
            self.rows = int(np.ceil(n_views / self.cols))
    
    def _create_cameras(self):
        """创建多个相机"""
        for view_name in self.view_names:
            if view_name not in MULTI_VIEW_CAMERAS:
                logger.warning(f"未知视角: {view_name}, 跳过")
                continue
            
            config = MULTI_VIEW_CAMERAS[view_name]
            cam = self.scene.add_camera(
                name=config.name,
                width=config.width,
                height=config.height,
                fovy=config.fovy,
                near=0.1,
                far=10
            )
            
            # 设置相机位姿
            pose = sapien.Pose(config.position, config.quaternion)
            cam.set_local_pose(pose)
            
            self.cameras[view_name] = cam
            logger.info(f"创建相机: {view_name} @ {config.position}")
    
    def capture_all_views(self) -> Dict[str, np.ndarray]:
        """
        从所有相机捕获图像
        
        Returns:
            字典，键为视角名称，值为BGR图像
        """
        images = {}
        
        for view_name, cam in self.cameras.items():
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            images[view_name] = bgr
        
        return images
    
    def create_composite_image(self, images: Dict[str, np.ndarray], 
                                add_labels: bool = True) -> np.ndarray:
        """
        将多视角图像拼接成一张大图
        
        Args:
            images: 视角名称到图像的字典
            add_labels: 是否添加视角标签
            
        Returns:
            合成的图像
        """
        if not images:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # 单视角模式直接返回
        if len(images) == 1:
            img = list(images.values())[0]
            if add_labels:
                view_name = list(images.keys())[0]
                label = view_name.replace("_", " ").title()
                cv2.putText(img, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1)
            return img
        
        # 获取单个图像尺寸
        first_img = list(images.values())[0]
        h, w = first_img.shape[:2]
        
        # 创建大图
        composite = np.zeros((h * self.rows, w * self.cols, 3), dtype=np.uint8)
        
        for idx, view_name in enumerate(self.view_names):
            if view_name not in images:
                continue
            
            row = idx // self.cols
            col = idx % self.cols
            
            img = images[view_name].copy()
            
            # 添加视角标签
            if add_labels:
                label = view_name.replace("_", " ").title()
                cv2.putText(img, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1)
            
            # 放入对应位置
            composite[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        return composite
    
    def get_primary_camera(self) -> Optional[sapien.Entity]:
        """
        获取主相机（用于Viewer）
        
        Returns:
            主相机实体，如果不存在则返回None
        """
        if self.view_names and self.view_names[0] in self.cameras:
            return self.cameras[self.view_names[0]]
        return list(self.cameras.values())[0] if self.cameras else None


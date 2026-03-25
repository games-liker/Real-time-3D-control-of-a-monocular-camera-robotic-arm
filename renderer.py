"""
渲染进程

负责3D场景渲染和机器人可视化
"""
import multiprocessing
import cv2
import numpy as np
import sapien
from pathlib import Path
from queue import Empty

from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.retargeting_config import RetargetingConfig

# 处理导入（支持直接运行和模块导入）
try:
    from .data_structures import IKResult, DepthResult
    from .utils import get_assembly_info
    from .camera_manager import MultiViewCameraManager
except ImportError:
    from data_structures import IKResult, DepthResult
    from utils import get_assembly_info
    from camera_manager import MultiViewCameraManager


def render_process_multiview(
    queue_in: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    robot_dir: str,
    config_path: str,
    use_arm: bool = True,
    image_scale_factor: float = 0.5,
    view_layout: str = "single",
    show_multiview: bool = False,
):
    """
    渲染进程（多视角版本，纯操控模式）
    
    Args:
        queue_in: 输入队列，接收 (IKResult, DepthResult) 元组
        stop_event: 停止事件
        robot_dir: 机器人URDF目录
        config_path: 重定向配置文件路径
        use_arm: 是否使用机械臂
        image_scale_factor: 图像缩放因子
        view_layout: 视角布局
        show_multiview: 是否显示多视角窗口
    """
    logger.info("[渲染进程] 进程启动（多视角模式）")
    logger.info(f"  - 视角布局: {view_layout}")
    
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])
    
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([-2, 0, 2]), np.array([1.5, 1.5, 1.5]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    
    # 创建多视角相机管理器
    multi_view_manager = None
    if show_multiview:
        multi_view_manager = MultiViewCameraManager(scene, view_layout=view_layout)
    
    # 主相机用于Viewer交互
    main_cam = scene.add_camera(name="main_camera", width=800, height=800, fovy=1, near=0.1, far=10)
    main_cam.set_local_pose(sapien.Pose([0, 0.8, 0.4], [-0.7071, 0, 0, 0.7071]))
    
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = True
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(main_cam.get_local_pose())
    
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    config = RetargetingConfig.load_from_file(config_path)
    retargeting = config.build()
    
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    
    hand_type = "Right" if "right" in config_path.lower() else "Left"
    
    if use_arm:
        assembly_dir, assembly_urdf, scale = get_assembly_info(robot_name, hand_type)
        if assembly_dir and assembly_urdf:
            assembly_path = Path(robot_dir).parent / "assembly" / assembly_dir / assembly_urdf
            if assembly_path.exists():
                filepath = str(assembly_path)
            else:
                use_arm = False
        else:
            use_arm = False
    
    if use_arm:
        if "ability" in robot_name:
            loader.scale = 1.0
        elif "shadow" in robot_name:
            loader.scale = 0.8
    
    robot = loader.load(str(filepath))
    robot.set_pose(sapien.Pose([0, 0, 0]))
    
    all_joints = robot.get_active_joints()
    sapien_joint_names = [joint.get_name() for joint in all_joints]
    retargeting_joint_names = retargeting.joint_names
    
    arm_joint_indices_sapien = []
    for i, joint_name in enumerate(sapien_joint_names):
        if joint_name not in retargeting_joint_names:
            arm_joint_indices_sapien.append(i)
    
    init_qpos_sapien = np.zeros(len(all_joints))
    if use_arm and len(arm_joint_indices_sapien) > 0:
        arm_init_pose = [0, -0.5, 0, 1.0, 0, 0.8, 0]
        for idx, arm_idx in enumerate(arm_joint_indices_sapien[:min(len(arm_init_pose), len(arm_joint_indices_sapien))]):
            init_qpos_sapien[arm_idx] = arm_init_pose[idx]
    
    robot.set_qpos(init_qpos_sapien)
    
    logger.info("[渲染进程] 机器人加载完成")
    logger.info("=" * 60)
    logger.info("3D实时重定向演示（基于手掌面积估计深度）")
    logger.info("请用手在空中移动，机器人会跟随你的手部动作")
    if show_multiview:
        logger.info(f"视角布局: {view_layout} ({len(multi_view_manager.view_names)}个视角)")
    logger.info("按 'q' 键退出")
    logger.info("=" * 60)
    
    while not stop_event.is_set():
        try:
            ik_result, depth_result = queue_in.get(timeout=0.1)
            
            if ik_result.joint_angles is not None:
                sapien_qpos = np.zeros(len(all_joints))
                for i, joint_name in enumerate(sapien_joint_names):
                    if joint_name in ik_result.joint_angles:
                        sapien_qpos[i] = ik_result.joint_angles[joint_name]
                
                robot.set_qpos(sapien_qpos)
            
            # 更新场景渲染
            scene.update_render()
            
            # 多视角显示（可选）
            if show_multiview and multi_view_manager is not None:
                multi_view_images = multi_view_manager.capture_all_views()
                composite_image = multi_view_manager.create_composite_image(multi_view_images, add_labels=True)
                
                # 在合成图像上添加深度信息
                if depth_result is not None:
                    info_y = 60
                    if depth_result.palm_area_ratio is not None:
                        area_text = f"Area: {depth_result.palm_area_ratio*100:.2f}%"
                        cv2.putText(composite_image, area_text, (10, info_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        info_y += 25
                    if depth_result.wrist_depth is not None:
                        depth_text = f"Depth: {depth_result.wrist_depth:.3f}"
                        cv2.putText(composite_image, depth_text, (10, info_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("Multi-View 3D Retargeting", composite_image)
            
        except Empty:
            pass
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            logger.info("[渲染进程] 用户请求退出")
            stop_event.set()
            break
        
        for _ in range(2):
            viewer.render()
    
    cv2.destroyAllWindows()
    logger.info("[渲染进程] 进程结束")


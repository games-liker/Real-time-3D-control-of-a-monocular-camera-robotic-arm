"""
IK求解进程

执行逆运动学求解，计算机器人关节角度
"""
import multiprocessing
import sys
import numpy as np
from pathlib import Path
from queue import Empty

from loguru import logger

from dex_retargeting.retargeting_config import RetargetingConfig

# 处理导入（支持直接运行和模块导入）
try:
    from .data_structures import DetectionResult, DepthResult, IKResult
    from .utils import get_assembly_info
    from .smoothers import RealtimeSmoother3D
    from .single_hand_detector import OPERATOR2MANO_RIGHT, OPERATOR2MANO_LEFT
except ImportError:
    from data_structures import DetectionResult, DepthResult, IKResult
    from utils import get_assembly_info
    from smoothers import RealtimeSmoother3D
    from single_hand_detector import OPERATOR2MANO_RIGHT, OPERATOR2MANO_LEFT


def ik_process_3d(
    queue_in: multiprocessing.Queue,
    queue_out: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    robot_dir: str,
    config_path: str,
    use_arm: bool = True,
    enable_arm_ik: bool = True,
    enable_smoothing: bool = True,
    smoothing_alpha: float = 0.3,
    fallback_y_position: float = 0.1,
    image_scale_factor: float = 1,
    enable_orientation_control: bool = True,
    orientation_weight: float = 0.1,
):
    """
    IK求解进程（3D版本）
    
    Args:
        queue_in: 输入队列，接收 (DetectionResult, DepthResult) 元组
        queue_out: 输出队列，发送 (IKResult, DepthResult) 元组
        stop_event: 停止事件
        robot_dir: 机器人URDF目录
        config_path: 重定向配置文件路径
        use_arm: 是否使用机械臂
        enable_arm_ik: 是否启用机械臂IK
        enable_smoothing: 是否启用平滑
        smoothing_alpha: 3D位置平滑系数
        fallback_y_position: 默认Y位置（当深度不可用时）
        image_scale_factor: 图像缩放因子
        enable_orientation_control: 是否启用姿态控制
        orientation_weight: 姿态权重
    """
    logger.info("[IK进程] 进程启动")
    
    # 延迟导入arm_ik_solver（从当前目录）
    try:
        from .arm_ik_solver import ArmIKSolver, get_end_effector_frame
    except ImportError:
        from arm_ik_solver import ArmIKSolver, get_end_effector_frame
    
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    config = RetargetingConfig.load_from_file(config_path)
    
    hand_type = "Right" if "right" in config_path.lower() else "Left"
    
    operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
    logger.info(f"[IK进程] 手型: {hand_type}, 姿态控制: {enable_orientation_control}")
    
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    
    assembly_dir = None
    if use_arm:
        assembly_dir, assembly_urdf, scale = get_assembly_info(robot_name, hand_type)
        if assembly_dir and assembly_urdf:
            assembly_path = Path(robot_dir).parent / "assembly" / assembly_dir / assembly_urdf
            if assembly_path.exists():
                filepath = str(assembly_path)
                logger.info(f"[IK进程] 使用组合体URDF: {filepath}")
            else:
                logger.warning(f"[IK进程] 组合体文件不存在: {assembly_path}")
                use_arm = False
        else:
            use_arm = False
    
    if not use_arm:
        if "glb" not in robot_name:
            filepath = str(filepath).replace(".urdf", "_glb.urdf")
            logger.info(f"[IK进程] 使用GLB URDF: {filepath}")
        else:
            filepath = str(filepath)
    
    retargeting_joint_names = retargeting.joint_names
    
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(filepath))
    
    all_joint_names = [model.names[i] for i in range(1, model.njoints)]
    total_joints = model.nq
    
    if total_joints != len(all_joint_names):
        total_joints = len(all_joint_names)
    
    arm_joint_indices = []
    for i, joint_name in enumerate(all_joint_names):
        if joint_name not in retargeting_joint_names:
            arm_joint_indices.append(i)
    
    arm_joint_count = len(arm_joint_indices)
    logger.info(f"[IK进程] 机械臂关节数: {arm_joint_count}")
    
    ik_solver = None
    if use_arm and enable_arm_ik and arm_joint_count > 0:
        try:
            ee_frame = get_end_effector_frame(assembly_dir)
            ik_solver = ArmIKSolver(
                urdf_path=str(filepath),
                end_effector_frame=ee_frame,
                arm_joint_count=arm_joint_count,
            )
            logger.info(f"[IK进程] IK求解器初始化成功")
        except Exception as e:
            logger.warning(f"[IK进程] IK求解器初始化失败: {e}")
    
    retargeting_to_full = []
    for name in all_joint_names:
        if name in retargeting_joint_names:
            retargeting_to_full.append(retargeting_joint_names.index(name))
        else:
            retargeting_to_full.append(-1)
    
    retargeting_to_full = np.array(retargeting_to_full)
    
    init_qpos = np.zeros(total_joints)
    if use_arm and len(arm_joint_indices) > 0:
        arm_init_pose = [0, -0.5, 0, 1.0, 0, 0.8, 0]
        for idx, arm_idx in enumerate(arm_joint_indices[:min(len(arm_init_pose), len(arm_joint_indices))]):
            init_qpos[arm_idx] = arm_init_pose[idx]
    
    smoother = None
    if enable_smoothing:
        smoother = RealtimeSmoother3D(alpha=smoothing_alpha)
    
    base_position = np.array([0.2, 0.35, 0.2])
    axis_mapping = np.array([-0.5, 5, -0.5])
    
    logger.info(f"[IK进程] 初始化完成，总关节数: {total_joints}")
    
    while not stop_event.is_set():
        try:
            detection_result, depth_result = queue_in.get(timeout=0.1)
        except Empty:
            continue
        
        ee_position = None
        target_position = None
        
        if detection_result.joint_pos is None:
            full_qpos = init_qpos.copy()
        else:
            full_qpos = init_qpos.copy()
            
            joint_pos = detection_result.joint_pos
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            hand_qpos = retargeting.retarget(ref_value)
            
            if ik_solver is not None and detection_result.wrist_2d_pos is not None:
                wrist_2d = detection_result.wrist_2d_pos.copy()
                
                if depth_result is not None and depth_result.wrist_depth is not None:
                    y_position = depth_result.wrist_depth
                else:
                    y_position = fallback_y_position
                
                wrist_3d_raw = np.array([wrist_2d[0], wrist_2d[1], y_position])
                
                if smoother is not None:
                    wrist_3d_processed = smoother.smooth(wrist_3d_raw)
                    y_position = wrist_3d_processed[2]
                    wrist_2d_processed = wrist_3d_processed[:2]
                else:
                    wrist_2d_processed = wrist_2d
                
                target_position = np.array([
                    base_position[0] + wrist_2d_processed[0] * image_scale_factor * axis_mapping[0],
                    (y_position - base_position[1]) * image_scale_factor * axis_mapping[1],
                    base_position[2] + wrist_2d_processed[1] * image_scale_factor * axis_mapping[2],
                ])
                
                target_rotation = None
                if enable_orientation_control and detection_result.wrist_rot is not None:
                    target_rotation = detection_result.wrist_rot @ operator2mano
                
                try:
                    current_arm_qpos = np.array([full_qpos[idx] for idx in arm_joint_indices])
                    initial_qpos_for_ik = np.zeros(ik_solver.model.nq)
                    initial_qpos_for_ik[:arm_joint_count] = current_arm_qpos[:arm_joint_count]
                    
                    actual_orientation_weight = orientation_weight if enable_orientation_control else 0.0
                    
                    arm_qpos, success, error = ik_solver.solve_ik(
                        target_position=target_position,
                        target_rotation=target_rotation,
                        initial_qpos=initial_qpos_for_ik,
                        position_weight=1.0,
                        orientation_weight=actual_orientation_weight,
                    )
                    
                    for idx, arm_idx in enumerate(arm_joint_indices[:len(arm_qpos)]):
                        full_qpos[arm_idx] = arm_qpos[idx]
                    
                    ee_position = ik_solver.forward_kinematics(arm_qpos)
                    
                except Exception as e:
                    logger.debug(f"[IK进程] IK求解失败: {e}")
            
            for i, idx in enumerate(retargeting_to_full):
                if idx >= 0:
                    full_qpos[i] = hand_qpos[idx]
        
        joint_angles = {}
        for i, name in enumerate(all_joint_names):
            joint_angles[name] = full_qpos[i]
        
        ik_result = IKResult(
            full_qpos=full_qpos,
            joint_angles=joint_angles,
            ee_position=ee_position,
            target_position=target_position,
            frame_id=detection_result.frame_id,
            timestamp=detection_result.timestamp,
        )
        
        try:
            if queue_out.full():
                try:
                    queue_out.get_nowait()
                except Empty:
                    pass
            queue_out.put((ik_result, depth_result), timeout=0.1)
        except:
            pass
    
    logger.info("[IK进程] 进程结束")


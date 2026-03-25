"""
机械臂逆运动学求解器 - 修复版
"""
import numpy as np
import pinocchio as pin
from loguru import logger
from typing import Optional, Tuple


class ArmIKSolver:
    """使用Pinocchio的数值IK求解器 - 修复版本"""
    
    def __init__(
        self,
        urdf_path: str,
        end_effector_frame: str,
        arm_joint_count: int = 7,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        damping: float = 1e-6,
        step_size: float = 0.5,
    ):
        """
        初始化IK求解器
        
        Args:
            urdf_path: 机械臂+手的URDF文件路径
            end_effector_frame: 末端执行器（手腕）的frame名称
            arm_joint_count: 机械臂的关节数量（前N个关节）
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值
            damping: 阻尼系数（用于数值稳定性）
            step_size: 更新步长
        """
        # 加载URDF模型
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # IK参数
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self.step_size = step_size
        self.arm_joint_count = arm_joint_count
        
        # 获取末端执行器frame ID
        try:
            self.ee_frame_id = self.model.getFrameId(end_effector_frame)
            logger.info(f"找到末端执行器frame: {end_effector_frame} (ID: {self.ee_frame_id})")
        except Exception as e:
            logger.error(f"无法找到末端执行器frame '{end_effector_frame}': {e}")
            # 列出所有可用的frames
            logger.info("可用的frames:")
            for i, frame in enumerate(self.model.frames):
                logger.info(f"  {i}: {frame.name}")
            raise
        
        # 初始化关节配置
        self.q_current = pin.neutral(self.model)
        
        # 获取机械臂关节的限制
        self.joint_limits_lower = self.model.lowerPositionLimit[:arm_joint_count]
        self.joint_limits_upper = self.model.upperPositionLimit[:arm_joint_count]
        
        logger.info(f"IK求解器初始化完成")
        logger.info(f"  机械臂关节数: {arm_joint_count}")
        logger.info(f"  总关节数: {self.model.nq}")
        logger.info(f"  关节限制: {self.joint_limits_lower} ~ {self.joint_limits_upper}")
    
    def solve_ik(
        self,
        target_position: np.ndarray,
        target_rotation: Optional[np.ndarray] = None,
        initial_qpos: Optional[np.ndarray] = None,
        position_weight: float = 1.0,
        orientation_weight: float = 0.3,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        求解逆运动学 - 简化版本，提高稳定性
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_rotation: 目标旋转矩阵 3x3，如果为None则只考虑位置
            initial_qpos: 初始关节配置，如果为None则使用当前配置
            position_weight: 位置误差权重
            orientation_weight: 姿态误差权重
            
        Returns:
            (arm_qpos, success, final_error): 机械臂关节角度、是否成功、最终误差
        """
        # 初始化关节配置
        if initial_qpos is not None:
            q = initial_qpos.copy()
        else:
            q = self.q_current.copy()
        
        # 构建目标位姿
        target_pose = pin.SE3.Identity()
        target_pose.translation = target_position

        if target_rotation is not None:
            target_pose.rotation = target_rotation
        
        success = False
        final_error = float('inf')
        
        # 简化：先尝试只控制位置，不考虑姿态
        position_only = target_rotation is None
        
        for iteration in range(self.max_iterations):
            # 前向运动学
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # 获取当前末端执行器位姿
            current_pose = self.data.oMf[self.ee_frame_id]
            
            # 简化的误差计算
            if position_only:
                # 只考虑位置误差
                position_error = target_pose.translation - current_pose.translation
                error_norm = np.linalg.norm(position_error)
                final_error = error_norm
                
                # 检查收敛
                if error_norm < self.tolerance:
                    success = True
                    break
                
                # 计算位置雅可比（世界坐标系）
                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.WORLD
                )
                # 只取位置部分（前3行）和机械臂关节
                J_pos = J_full[:3, :self.arm_joint_count]
                
                # 使用伪逆求解：dq = J⁺ * error
                try:
                    J_pinv = np.linalg.pinv(J_pos)
                    dq_arm = J_pinv @ position_error
                except np.linalg.LinAlgError:
                    # 伪逆失败，使用阻尼最小二乘
                    J_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + self.damping * np.eye(3))
                    dq_arm = J_pinv @ position_error
                
            else:
                # 同时考虑位置和姿态
                error_se3 = pin.log(current_pose.inverse() * target_pose)
                error_norm = np.linalg.norm(error_se3)
                final_error = error_norm
                
                if error_norm < self.tolerance:
                    success = True
                    break
                
                # 计算完整雅可比
                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL
                )
                J = J_full[:, :self.arm_joint_count]
                
                # 使用伪逆求解
                try:
                    J_pinv = np.linalg.pinv(J)
                    dq_arm = J_pinv @ error_se3.vector
                except np.linalg.LinAlgError:
                    J_pinv = J.T @ np.linalg.inv(J @ J.T + self.damping * np.eye(6))
                    dq_arm = J_pinv @ error_se3.vector
            
            # 限制关节速度，避免过大步长
            dq_norm = np.linalg.norm(dq_arm)
            if dq_norm > 0.5:  # 限制最大步长
                dq_arm = dq_arm * 0.5 / dq_norm
            
            # 更新关节角度
            q[:self.arm_joint_count] += self.step_size * dq_arm
            
            # 应用关节限制
            q[:self.arm_joint_count] = np.clip(
                q[:self.arm_joint_count],
                self.joint_limits_lower + 0.01,  # 避免触及极限边界
                self.joint_limits_upper - 0.01
            )
            
            # 每10次迭代输出调试信息
            if iteration % 10 == 0:
                logger.debug(f"IK迭代 {iteration}: 误差 = {error_norm:.6f}")
        
        if success:
            logger.debug(f"IK收敛成功，最终误差: {final_error:.6f}")
        else:
            logger.warning(f"IK未收敛，最终误差: {final_error:.6f}")
        
        # 保存当前配置
        self.q_current = q.copy()
        
        # 只返回机械臂关节角度
        arm_qpos = q[:self.arm_joint_count]
        
        return arm_qpos, success, final_error
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """前向运动学，计算末端位置"""
        # 构建完整关节角度（机械臂+手部）
        full_q = np.concatenate([q, np.zeros(self.model.nq - len(q))])
        
        pin.forwardKinematics(self.model, self.data, full_q)
        pin.updateFramePlacements(self.model, self.data)
        
        current_pose = self.data.oMf[self.ee_frame_id]
        return current_pose.translation
    
    def reset(self):
        """重置到中性位置"""
        self.q_current = pin.neutral(self.model)


# 末端执行器frame名称映射表
END_EFFECTOR_FRAMES = {
    "xarm7_ability": "link7",  # xarm7的最后一个link
    "ur5e_shadow": "wrist_3_link",  # ur5e的手腕link
    "rm75_inspire": "rm75_link_7",  # rm75的最后一个link
}


def get_end_effector_frame(assembly_dir: str) -> str:
    """根据组合体目录名获取末端执行器frame名称"""
    if assembly_dir in END_EFFECTOR_FRAMES:
        return END_EFFECTOR_FRAMES[assembly_dir]
    else:
        # 尝试自动检测frame名称
        frame_candidates = [
            "link7", "wrist_3_link", "ee_link", "end_effector", 
            "flange", "tool0", "hand_frame"
        ]
        logger.warning(f"未找到 {assembly_dir} 的映射，使用默认候选: {frame_candidates[0]}")
        return frame_candidates[0]

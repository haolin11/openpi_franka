from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
from polymetis import RobotInterface, GripperInterface
import pyrealsense2 as rs
import cv2
import time
import torch

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="172.16.97.95", port=8000)


class DataTrans:

    
    def __init__(self, freq=10, save_dir="collected_data_right"):
        self.serial_1 = "136622074722"  # 全局相机序列号
        self.serial_2 = "233622071355"  # 腕部相机序列号
        self.freq = freq
        self.period = 1.0 / freq
        
        self.robot = RobotInterface(ip_address="localhost")
        self.gripper = GripperInterface(ip_address="localhost")

        
        # 初始化全局相机
        self.global_cam_pipeline = rs.pipeline()
        global_config = rs.config()
        # 使用设备序列号区分全局相机
        global_config.enable_device(self.serial_1)  # 请替换为实际全局相机的序列号
        global_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.global_cam_pipeline.start(global_config)
        
        # 初始化腕部相机
        self.wrist_cam_pipeline = rs.pipeline()
        wrist_config = rs.config()
        # 使用设备序列号区分腕部相机
        wrist_config.enable_device(self.serial_2)  # 请替换为实际腕部相机的序列号
        wrist_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.wrist_cam_pipeline.start(wrist_config)


        
    def _get_global_camera_image(self):
        """获取全局相机图像
        
        Returns:
            numpy.ndarray: 形状为(256, 256, 3)的RGB图像，数据类型为np.uint8
        """
        frames = self.global_cam_pipeline.wait_for_frames()
        # 获取彩色图像帧
        color_frame = frames.get_color_frame()
        
        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        
        # 将图像缩放到256x256
        color_image = cv2.resize(color_image, (256, 256))
        
        # 转换为RGB格式
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # 确保数据类型为uint8
        color_image = color_image.astype(np.uint8)
        
        return color_image
    
    def _get_wrist_camera_image(self):
        """获取腕部相机(Realsense)图像
        
        Returns:
            numpy.ndarray: 形状为(256, 256, 3)的RGB图像，数据类型为np.uint8
        """
        frames = self.wrist_cam_pipeline.wait_for_frames()
        # 获取彩色图像帧
        color_frame = frames.get_color_frame()
        
        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        
        # 将图像缩放到256x256
        color_image = cv2.resize(color_image, (256, 256))
        
        # 转换为RGB格式
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # 确保数据类型为uint8
        color_image = color_image.astype(np.uint8)
        
        return color_image
    
    def _get_robot_state(self):
        """获取机械臂的关节角和夹爪宽度
        
        Returns:
            numpy.ndarray: 形状为(8,)的浮点数数组，包含7个关节角和1个夹爪宽度，数据类型为np.float64
        """
        # 获取机械臂关节角度
        joint_state = self.robot.get_joint_positions()
        
        # 获取夹爪宽度
        gripper_width = self.gripper.get_state().width
        
        # 将关节角和夹爪宽度拼接，并转换为float64以匹配np.random.rand的类型
        robot_state = np.concatenate([joint_state, np.array([gripper_width])]).astype(np.float64)
        
        return robot_state
    
    def _get_robot_action(self):
        """获取机械臂末端位姿(3位置+4四元数)
        
        Returns:
            numpy.ndarray: 形状为(7,)的浮点数数组，包含3个位置和4个四元数，数据类型为np.float64
        """
        pos, quat = self.robot.get_ee_pose()
        # 确保四元数归一化
        # quat = np.random.uniform(-1, 1, (4,))
        # quat = quat / np.linalg.norm(quat)
        
        # 转换为float64以匹配np.random.rand的类型
        return np.concatenate([pos, quat]).astype(np.float64)


datatrans = DataTrans(freq=10)
state = datatrans._get_robot_state()
print(state)
global_image = datatrans._get_global_camera_image()
wrist_image = datatrans._get_wrist_camera_image()
time.sleep(1)

num_steps = 50

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/state": state,
        "observation/image": global_image,
        "observation/wrist_image": wrist_image,
        "prompt": "pick the box",
    }

    # 从策略服务器获取动作序列
    print("向策略服务器请求动作序列...")
    action_chunk = client.infer(observation)["actions"]
    print(f"获得动作序列，形状: {action_chunk.shape}")

    # # 启动关节阻抗控制
    # print("启动关节阻抗控制...")
    # datatrans.robot.start_joint_impedance(blocking=False)
    # time.sleep(0.1)  # 等待控制器初始化

    # 循环执行动作序列中的每一步
    try:
        # 记录上一次的夹爪宽度，避免频繁发送相同的指令
        last_gripper_width = None
        
        for i, action in enumerate(action_chunk):
            # 提取关节位置和夹爪宽度
            joint_positions = action[:7]  # 前7个值为关节角度
            gripper_width = action[7]     # 第8个值为夹爪宽度
            
            # 更新关节位置
            print(f"执行第 {i+1}/{len(action_chunk)} 步")
            # datatrans.robot.update_desired_joint_positions(torch.tensor(joint_positions))
            datatrans.robot.move_to_joint_positions(torch.tensor(joint_positions))
            
            # 控制夹爪宽度
            if last_gripper_width is None or abs(gripper_width - last_gripper_width) > 0.001:
                print(f"  夹爪宽度: {gripper_width:.4f}")
                datatrans.gripper.goto(width=float(gripper_width), speed=0.1, force=10.0, blocking=False)
                last_gripper_width = gripper_width
            
            # 等待下一个控制周期
            # time.sleep(0.1)  # 假设控制频率为10Hz
    finally:
        # 终止当前策略
        print("终止当前策略...")
        datatrans.robot.terminate_current_policy()
print("执行完成")


# 确保服务端已经启动
# uv run scripts/serve_policy.py policy:checkpoint  --policy.config=pi0_franka  --policy.dir=/home/ubuntu/openpi/checkpoints/pi0_franka/local_dataset_test/14999

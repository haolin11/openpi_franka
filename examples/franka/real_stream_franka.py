#!/usr/bin/env python3
"""
real_stream_franka.py

在真实机械臂上测试大小脑分频策略（StreamingPolicy）：
- prefix（慢脑）阶段，每秒发送图像+语言+状态
- suffix（快脑）阶段，每秒多次发送状态并执行单步动作
"""
import time
import numpy as np
import torch
from openpi_client import websocket_client_policy, image_tools
from polymetis import RobotInterface, GripperInterface
import pyrealsense2 as rs
import cv2
# from real_franka import DataTrans

# 配置参数
HOST = "localhost"
PORT = 6000
IMAGE_HZ = 0.5       # 慢脑（图像+语言）频率
STATE_HZ = 50      # 快脑（状态）频率
PROMPT = "pick the tissue box"


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


def main():
    # 初始化客户端
    client = websocket_client_policy.WebsocketClientPolicy(host=HOST, port=PORT)

    # 初始化数据采集
    datatrans = DataTrans(freq=STATE_HZ)

    # 机械臂初始准备
    datatrans.robot.go_home()
    datatrans.gripper.goto(width=0.08, speed=0.1, force=10.0, blocking=False)
    pos, qua = datatrans.robot.get_ee_pose()
    pos[2] -= 0.3
    pos[1] +=0.00
    # pos[0] += 0.1
    # time.sleep(0.3)
    datatrans.robot.move_to_ee_pose(pos, qua) 
    time.sleep(3)
    print("启动关节阻抗控制...")
    datatrans.robot.start_joint_impedance(blocking=False)
    time.sleep(0.1)

    # # 模型预热
    # print("模型预热：首次调用infer进行JIT编译，请耐心等待...")
    # warm_state = datatrans._get_robot_state()
    # warm_img = datatrans._get_global_camera_image()
    # warm_wrist = datatrans._get_wrist_camera_image()
    # dummy_obs = {
    #     "observation/state": warm_state,
    #     "observation/image": warm_img,
    #     "observation/wrist_image": warm_wrist,
    #     "prompt": "warmup",
    # }
    # _ = client.infer(dummy_obs)
    # print("预热完成，开始流式推理...")

    # 事件调度参数
    img_interval = 1.0 / IMAGE_HZ
    state_interval = 1.0 / STATE_HZ
    next_prefix_time = time.time()
    next_suffix_time = next_prefix_time
    start_time = next_prefix_time
    last_gripper_width = None

    try:
        while True:
            now = time.time()
            # 确定下一个事件类型
            if next_prefix_time <= next_suffix_time:
                event = "prefix"
                event_time = next_prefix_time
                next_prefix_time += img_interval
            else:
                event = "suffix"
                event_time = next_suffix_time
                next_suffix_time += state_interval

            # 等待触发
            sleep_t = event_time - now
            if sleep_t > 0:
                time.sleep(sleep_t)

            # 构造观测
            if event == "prefix":
                state = datatrans._get_robot_state()
                img = datatrans._get_global_camera_image()
                wrist = datatrans._get_wrist_camera_image()
                obs = {
                    "observation/state": state,
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "prompt": PROMPT,
                }
            else:
                state = datatrans._get_robot_state()
                obs = {"observation/state": state}

            # 推理
            t0 = time.time()
            action = client.infer(obs)["actions"]
            t_inf = time.time() - t0
            elapsed = time.time() - start_time
            print(f"{event} @ {elapsed:.3f}s (推理耗时 {t_inf:.3f}s) -> action: {action}")

            # 执行动作
            joint_pos = action[0, :7]
            gripper_w = action[0, 7]
            try:
                # datatrans.robot.update_desired_joint_positions(torch.tensor(joint_pos))
                pass
            except Exception:
                datatrans.robot.start_joint_impedance(blocking=False)
                # datatrans.robot.update_desired_joint_positions(torch.tensor(joint_pos))
            if last_gripper_width is None or abs(gripper_w - last_gripper_width) > 0.005:
                datatrans.gripper.goto(width=float(gripper_w), speed=0.1, force=10.0, blocking=False)
                last_gripper_width = gripper_w

    except KeyboardInterrupt:
        print("捕获到中断，终止流式推理...")
    finally:
        # 终止当前策略
        try:
            datatrans.robot.terminate_current_policy()
        except Exception:
            pass
        print("已关闭机械臂控制，退出。")


if __name__ == "__main__":
    main() 
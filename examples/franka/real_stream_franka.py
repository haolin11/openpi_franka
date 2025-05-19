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
from real_franka import DataTrans

# 配置参数
HOST = "localhost"
PORT = 8000
IMAGE_HZ = 1       # 慢脑（图像+语言）频率
STATE_HZ = 10      # 快脑（状态）频率
PROMPT = "pick the tissue box"


def main():
    # 初始化客户端
    client = websocket_client_policy.WebsocketClientPolicy(host=HOST, port=PORT)

    # 初始化数据采集
    datatrans = DataTrans(freq=STATE_HZ)

    # 机械臂初始准备
    datatrans.robot.go_home()
    datatrans.gripper.goto(width=0.08, speed=0.1, force=10.0, blocking=False)
    time.sleep(1)
    print("启动关节阻抗控制...")
    datatrans.robot.start_joint_impedance(blocking=False)
    time.sleep(0.1)

    # 模型预热
    print("模型预热：首次调用infer进行JIT编译，请耐心等待...")
    warm_state = datatrans._get_robot_state()
    warm_img = datatrans._get_global_camera_image()
    warm_wrist = datatrans._get_wrist_camera_image()
    warm_img = image_tools.resize_image(warm_img, 224)
    warm_wrist = image_tools.resize_image(warm_wrist, 224)
    dummy_obs = {
        "observation/state": warm_state,
        "observation/image": warm_img,
        "observation/wrist_image": warm_wrist,
        "prompt": "warmup",
    }
    _ = client.infer(dummy_obs)
    print("预热完成，开始流式推理...")

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
                img = image_tools.resize_image(img, 224)
                wrist = image_tools.resize_image(wrist, 224)
                obs = {
                    "observation/state": state,
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "prompt": PROMPT,
                }
            else:
                state = datatrans._get_robot_state()
                obs = {"state": state}

            # 推理
            t0 = time.time()
            action = client.infer(obs)["actions"]
            t_inf = time.time() - t0
            elapsed = time.time() - start_time
            print(f"{event} @ {elapsed:.3f}s (推理耗时 {t_inf:.3f}s) -> action: {action}")

            # 执行动作
            joint_pos = action[:7]
            gripper_w = action[7]
            try:
                datatrans.robot.update_desired_joint_positions(torch.tensor(joint_pos))
            except Exception:
                datatrans.robot.start_joint_impedance(blocking=False)
                datatrans.robot.update_desired_joint_positions(torch.tensor(joint_pos))
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
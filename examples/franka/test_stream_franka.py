#!/usr/bin/env python3
from openpi_client import websocket_client_policy
import numpy as np
import time

# 测试 StreamingPolicy 的快慢系统行为
# 确保已经启动 server_stream_policy.py (流式服务) 在 localhost:8000

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
print("模型预热：首次调用infer进行JIT编译，请耐心等待...")
dummy_prefix = {
    "observation/state": np.zeros(8),
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "prompt": "warmup",
}
_ = client.infer(dummy_prefix)
print("模型预热完成，开始测试...")

# 频率设置
img_hz, state_hz = 1, 5
img_interval, state_interval = 1.0/img_hz, 1.0/state_hz
duration = 30  # 测试时长（秒)
start = time.time()
# 独立调度前缀和后缀，跟踪绝对触发时间
next_prefix_time = start
next_suffix_time = start
while time.time() - start < duration:
    now = time.time()
    # 决定下一个事件类型及其触发时间
    if next_prefix_time <= next_suffix_time:
        event = "prefix"
        event_time = next_prefix_time
    else:
        event = "suffix"
        event_time = next_suffix_time
    # 等待直到触发时间
    sleep_time = event_time - now
    if sleep_time > 0:
        time.sleep(sleep_time)
    # 生成观测
    if event == "prefix":
        obs = {
            "observation/state": np.random.rand(8),
            "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "prompt": "pick the box",
        }
        stage = "prefix"
        # 安排下次前缀时间
        next_prefix_time += img_interval
    else:
        obs = {"state": np.random.rand(8)}
        stage = "suffix"
        # 安排下次后缀时间
        next_suffix_time += state_interval
    # 推理并测量真实 elapsed 时间
    inf_start = time.time()
    action = client.infer(obs)["actions"]
    inf_end = time.time()
    elapsed = inf_end - start
    print(f"{stage} @ {elapsed:.3f}s -> action: {action}")


# 确保服务端已经启动
# uv run scripts/serve_stream_policy.py policy:checkpoint  --policy.config=pi0_franka  --policy.dir=/home/chenhaolin/openpi_franka/checkpoints/pi0_franka/lora_fine_tune/29999

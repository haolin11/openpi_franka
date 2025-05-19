#!/usr/bin/env python3
import dataclasses
import enum
import logging
import socket
import pathlib

import tyro
import jax.numpy as jnp

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.policy import PolicyRecorder
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from openpi.shared import download
from openpi.models.model import restore_params
from openpi.streaming.streaming_policy import StreamingPolicy


class EnvMode(enum.Enum):
    """支持的环境"""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """从已有 checkpoint 加载模型"""
    config: str  # 训练配置名，如 "pi0_aloha_sim"
    dir: str     # checkpoint 路径，如 "checkpoints/pi0_aloha_sim/exp/10000"


@dataclasses.dataclass
class Default:
    """使用默认环境对应的 checkpoint"""


@dataclasses.dataclass
class Args:
    """Arguments for the streaming policy server script."""

    # 环境（仅在使用 Default 时生效）
    env: EnvMode = EnvMode.ALOHA_SIM

    # 如果没有输入 prompt，可用此默认值
    default_prompt: str | None = None

    # 服务端口
    port: int = 8000

    # 是否记录 policy 行为
    record: bool = False

    # 加载方式：指定 Checkpoint 或使用 Default
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # 扩散最大步数（控制 action_expert 的 diffusion 迭代次数）
    num_diffusion_steps: int = 10


# 默认 checkpoint 映射
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(config="pi0_aloha", dir="s3://openpi-assets/checkpoints/pi0_base"),
    EnvMode.ALOHA_SIM: Checkpoint(config="pi0_aloha_sim", dir="s3://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint(config="pi0_fast_droid", dir="s3://openpi-assets/checkpoints/pi0_fast_droid"),
    EnvMode.LIBERO: Checkpoint(config="pi0_fast_libero", dir="s3://openpi-assets/checkpoints/pi0_fast_libero"),
}


def create_default_checkpoint(env: EnvMode) -> Checkpoint:
    if cp := DEFAULT_CHECKPOINT.get(env):
        return cp
    raise ValueError(f"Unsupported environment: {env}")


def create_policy(args: Args) -> _policy.BasePolicy:
    """根据 args 构造 StreamingPolicy 实例"""
    # 1. 选择 train_config 和 checkpoint_dir
    match args.policy:
        case Checkpoint():
            train_cfg = _config.get_config(args.policy.config)
            ckpt_dir = args.policy.dir
        case Default():
            default_cp = create_default_checkpoint(args.env)
            train_cfg = _config.get_config(default_cp.config)
            ckpt_dir = default_cp.dir

    # 2. 先用已有逻辑加载 base policy，以获取 transforms, sample_kwargs, metadata
    base_policy = _policy_config.create_trained_policy(
        train_cfg,
        ckpt_dir,
        default_prompt=args.default_prompt,
    )

    # 3. 加载原始 model
    local_ckpt = download.maybe_download(str(ckpt_dir))
    params = restore_params(pathlib.Path(local_ckpt) / "params", dtype=jnp.bfloat16)
    model = train_cfg.model.load(params)

    # 4. 用 StreamingPolicy 包装
    streaming_policy = StreamingPolicy(
        model,
        num_diffusion_steps=args.num_diffusion_steps,
        transforms=[base_policy._input_transform],
        output_transforms=[base_policy._output_transform],
        sample_kwargs=base_policy._sample_kwargs,
        metadata=base_policy.metadata,
    )

    # 5. 可选地记录
    if args.record:
        streaming_policy = PolicyRecorder(streaming_policy, "policy_records")
    return streaming_policy


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    policy = create_policy(args)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Starting streaming policy server at %s (%s:%d)", hostname, local_ip, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args)) 
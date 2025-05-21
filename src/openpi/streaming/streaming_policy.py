import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.model import Observation
import openpi.models.pi0 as pi0_module
from openpi.policies.policy import Policy
import types
from openpi.shared.nnx_utils import module_jit
from openpi.transforms import pad_to_dim, Normalize  # 新增：后缀阶段 state pad & normalize

class StreamingPolicy(Policy):
    """大小脑分频策略：慢脑(视觉+语言)低频更新 kv_cache，快脑(动作专家)高频输出单步动作。"""
    def __init__(self, model, *, num_diffusion_steps: int = 10, transforms=(), output_transforms=(), sample_kwargs=None, metadata=None):
        # 调用父类 Policy __init__ 初始化 transforms, sample_kwargs, metadata
        super().__init__(model, transforms=transforms, output_transforms=output_transforms, sample_kwargs=sample_kwargs, metadata=metadata)
        # 保存底层模型以供 embed_prefix/suffix 使用
        self._model = model
        self._rng = jax.random.PRNGKey(0)  # 使用与原版pi0相同的随机种子
        # 小脑上下文状态
        self._kv_cache = None
        self._prefix_mask = None
        self._prefix_len = 0
        # 扩散状态
        self._x_t = None
        self._time = None
        self._num_diffusion_steps = num_diffusion_steps
        # 新增：把后缀推理函数编译成 XLA 内核
        self._suffix_step = jax.jit(self._suffix_step)
        # 使用 module_jit 绑定并编译 embed_prefix + llm decode
        def _embed_prefix_and_llm(model, obs_jax):
            prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs_jax)
            attn_mask = pi0_module.make_attn_mask(prefix_mask, prefix_ar)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (_, _), kv = model.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
            prefix_len = prefix_mask.sum()
            return kv, prefix_mask, prefix_len
        self._model._embed_prefix_and_llm = types.MethodType(_embed_prefix_and_llm, self._model)
        self._embed_prefix_fn = module_jit(self._model._embed_prefix_and_llm)
        # 递归查找 Normalize 实例，用于后缀阶段对 state 归一化
        def _find_normalize(transform):
            if isinstance(transform, Normalize):
                return transform
            if hasattr(transform, 'transforms'):
                for sub in transform.transforms:
                    found = _find_normalize(sub)
                    if found is not None:
                        return found
            return None
        self._normalize = _find_normalize(self._input_transform)
        if self._normalize is None:
            raise RuntimeError('无法找到 Normalize 变换')
        # 预生成固定的扩散噪声
        self._fixed_noise = None

    def _transform_state(self, state: np.ndarray) -> np.ndarray:
        """专门用于处理状态的转换函数"""
        # 先进行维度填充
        state = pad_to_dim(state, self._model.action_dim)
        # 再进行归一化
        if self._normalize is not None:
            state = self._normalize({"state": state})["state"]
        return state

    def infer(self, obs: dict) -> dict:
        # 判断是否有新的视觉/语言输入
        has_visual = any(k.startswith("image") for k in obs) or ("prompt" in obs)
        # 前缀（慢脑）阶段，进行完整的 transforms
        if has_visual:
            # 预处理输入：执行 repack, data_transforms, model_transforms
            inputs = self._input_transform(obs)
            # 构造 batch_size=1 的 Observation
            obs_jax = Observation.from_dict(jax.tree_map(lambda x: jnp.asarray(x)[None], inputs))
            # 缓存 prefix Observation，用于后续 suffix
            self._prefix_obs_jax = obs_jax
            # 使用 module_jit 编译的 embed_prefix + llm 函数获取缓存
            self._kv_cache, self._prefix_mask, self._prefix_len = self._embed_prefix_fn(obs_jax)
            # 初始化扩散噪声/时间
            batch = obs_jax.state.shape[0]
            self._rng, subkey = jax.random.split(self._rng)
            self._x_t = jax.random.normal(subkey, (batch, self._model.action_horizon, self._model.action_dim))
            self._time = jnp.ones((batch,))
            # 执行完整的扩散过程
            for _ in range(self._num_diffusion_steps):
                v_t, self._x_t, self._time = self._suffix_step(
                    obs_jax, self._x_t, self._time, self._kv_cache, self._prefix_mask, self._prefix_len
                )
            # 缓存最终的动作值和状态
            self._last_prefix_action = self._x_t.copy()
            self._last_prefix_state = obs_jax.state.copy()
        else:
            # 快脑（suffix）阶段，仅更新 state
            if not hasattr(self, '_prefix_obs_jax') or self._prefix_obs_jax is None:
                raise RuntimeError('Suffix inference before any prefix step')
            if 'observation/state' not in obs:
                raise KeyError('observation/state key is required for suffix inference')
            
            # 使用专门的状态转换函数处理状态
            new_state = self._transform_state(obs['observation/state'])
            new_state = jnp.asarray(new_state)[None]
            
            # 使用缓存的前缀观察，只更新状态
            obs_jax = self._prefix_obs_jax.replace(state=new_state)
            
            # 生成新的随机噪声
            batch = obs_jax.state.shape[0]
            self._rng, subkey = jax.random.split(self._rng)
            self._x_t = jax.random.normal(subkey, (batch, self._model.action_horizon, self._model.action_dim))
            self._time = jnp.ones((obs_jax.state.shape[0],))
            
            # 执行完整的扩散过程
            for _ in range(self._num_diffusion_steps):
                v_t, self._x_t, self._time = self._suffix_step(
                    obs_jax, self._x_t, self._time, self._kv_cache, self._prefix_mask, self._prefix_len
                )

        # 提取本帧动作并返回
        action = np.asarray(self._x_t[0])
        # 同时返回state，供输出变换使用
        state_out = np.asarray(obs_jax.state[0])
        outputs = {"state": state_out, "actions": action}
        return self._output_transform(outputs)

    # 新增：后缀推理函数，JIT 编译后只保留核心计算
    def _suffix_step(self, obs_jax, x_t, time, kv_cache, prefix_mask, prefix_len):
        suffix_tokens, suffix_mask, suffix_ar = self._model.embed_suffix(obs_jax, x_t, time)
        suffix_attn = pi0_module.make_attn_mask(suffix_mask, suffix_ar)
        prefix_attn = jnp.broadcast_to(
            prefix_mask[:, None, :],
            suffix_attn.shape[:-1] + (prefix_mask.shape[-1],),
        )
        full_attn = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
        positions = prefix_len + (jnp.cumsum(suffix_mask, axis=1) - 1)
        (_, suffix_out), _ = self._model.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn, positions=positions, kv_cache=kv_cache
        )
        v_t = self._model.action_out_proj(suffix_out[:, -self._model.action_horizon :])
        dt = -1.0 / self._num_diffusion_steps
        new_x_t = x_t + dt * v_t
        new_time = time + dt
        return v_t, new_x_t, new_time

# 使用示例：
# from openpi.streaming.streaming_policy import StreamingPolicy
# policy = StreamingPolicy(model, num_diffusion_steps=10, transforms=..., output_transforms=...) 
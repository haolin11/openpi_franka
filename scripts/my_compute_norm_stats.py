"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

修改版：不使用 JAX 分片，避免多设备问题
"""

import numpy as np
import tqdm
import tyro
import os
import jax

# 设置 JAX 只使用 CPU，避免 GPU 分片问题
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    # 修改：使用普通的PyTorch DataLoader而不是JAX分片的DataLoader
    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,  # 减少工作进程数
        shuffle=shuffle,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    
    # 限制处理的样本数
    processed_samples = 0
    
    # 增加错误处理
    for batch in tqdm.tqdm(data_loader, total=min(num_frames, len(data_loader)), desc="Computing stats"):
        try:
            for key in keys:
                if key in batch:
                    values = np.asarray(batch[key][0])
                    stats[key].update(values.reshape(-1, values.shape[-1]))
            
            processed_samples += 1
            if processed_samples >= num_frames:
                break
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue

    print(f"成功处理了 {processed_samples} 个样本")
    
    # 修复：直接使用 RunningStats.get_statistics() 获取统计信息
    norm_stats = {}
    for key in keys:
        if key in stats:
            norm_stats[key] = stats[key].get_statistics()
    
    # 打印统计信息以便验证，注意 NormStats 的正确访问方式
    for key, stat in norm_stats.items():
        mean = stat.mean if hasattr(stat, 'mean') else stat['mean']
        std = stat.std if hasattr(stat, 'std') else stat['std']
        
        print(f"键 '{key}':")
        print(f"  均值形状: {mean.shape}, 范围: [{np.min(mean):.4f}, {np.max(mean):.4f}]")
        print(f"  标准差形状: {std.shape}, 范围: [{np.min(std):.4f}, {np.max(std):.4f}]")
    
    # 确保路径存在
    output_dir = None
    if hasattr(data_config, 'asset_id') and data_config.asset_id:
        output_dir = config.assets_dirs / data_config.asset_id
    else:
        # 替代方案：使用 repo_id 或一个默认值
        repo_id = data_config.repo_id
        if '/' in repo_id:
            # 如果是 "namespace/repo_name" 格式，只取后半部分
            repo_id = repo_id.split('/')[-1]
        output_dir = config.assets_dirs / repo_id
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"写入统计数据到: {output_dir}")
    
    try:
        normalize.save(output_dir, norm_stats)
        print(f"统计数据已成功保存")
    except Exception as e:
        print(f"保存统计数据时出错: {e}")
        # 备用方法：使用标准 pickle
        import pickle
        with open(os.path.join(output_dir, "norm_stats.pkl"), "wb") as f:
            pickle.dump(norm_stats, f)
        print(f"使用备用方法保存了统计数据")


if __name__ == "__main__":
    tyro.cli(main)
from __future__ import absolute_import, print_function

import os
import datetime
from shutil import copyfile
import configparser

from d3qn_training_simulation import D3QNSimulation
from generator import TrafficGenerator
from d3qn_model import D3QNModel, PrioritizedReplayMemory
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


def read_extra_from_ini(path="training_settings.ini"):
    """读取 [d3qn]/[PER] 里的可选配置（兼容 utils 未解析这些键的情况）"""
    cp = configparser.ConfigParser()
    cp.read(path, encoding='utf-8')
    extra = {}
    # d3qn
    if cp.has_section("d3qn"):
        extra["target_update_freq"] = cp["d3qn"].getint("target_update_freq", fallback=1000)
        extra["per_alpha"] = cp["d3qn"].getfloat("per_alpha", fallback=None)
        extra["per_beta"] = cp["d3qn"].getfloat("per_beta", fallback=None)
        extra["per_eps"] = cp["d3qn"].getfloat("per_eps", fallback=None)
        extra["per_beta_increment"] = cp["d3qn"].getfloat("per_beta_increment", fallback=None)
    # PER
    if cp.has_section("PER"):
        extra["use_per"] = cp["PER"].getboolean("use_per", fallback=True)
        extra["per_alpha"] = cp["PER"].getfloat("per_alpha", fallback=extra.get("per_alpha", 0.6))
        extra["per_beta"] = cp["PER"].getfloat("per_beta", fallback=extra.get("per_beta", 0.4))
        extra["per_eps"] = cp["PER"].getfloat("per_eps", fallback=extra.get("per_eps", 1e-6))
        extra["per_beta_increment"] = cp["PER"].getfloat("per_beta_increment", fallback=extra.get("per_beta_increment", 1e-3))
        # 新增可选：混合采样与优先级裁剪（若 utils 已读取，这里做兜底）
        extra["uniform_mix_ratio"] = cp["PER"].getfloat("uniform_mix_ratio", fallback=None)
        extra["priority_clip_max"] = cp["PER"].getfloat("priority_clip_max", fallback=None)
    else:
        extra.setdefault("use_per", True)
    
    # 新增：读取 reward 和 action_constraints 参数
    if cp.has_section("reward"):
        extra["flow_reward_weight"] = cp["reward"].getfloat("flow_reward_weight", fallback=0.05)
    
    if cp.has_section("action_constraints"):
        extra["max_green"] = cp["action_constraints"].getint("max_green", fallback=40)
        extra["imbalance_ratio"] = cp["action_constraints"].getfloat("imbalance_ratio", fallback=1.5)
        extra["max_wait_threshold"] = cp["action_constraints"].getfloat("max_wait_threshold", fallback=60.0)
    
    return extra


if __name__ == "__main__":
    # Load config
    cfg = import_train_configuration(config_file="training_settings.ini")
    extra = read_extra_from_ini("training_settings.ini")

    sumo_cmd = set_sumo(cfg["gui"], cfg["construction_config_file_name"], cfg["max_steps"])
    path = set_train_path(cfg["models_path_name"])

    # PER switch（优先用 cfg，fallback 到 extra）
    use_per = bool(cfg.get("use_per", extra.get("use_per", True)))

    # Build model (use target_update_freq)
    tuf = int(cfg.get("target_update_freq", extra.get("target_update_freq", 1000)))
    Model = D3QNModel(
        cfg["num_layers"],
        cfg["width_layers"],
        cfg["batch_size"],
        cfg["learning_rate"],
        input_dim=cfg["num_states"],
        output_dim=cfg["num_actions"],
        target_update_freq=tuf,
        dueling=True,
        target_value_clip_min=cfg.get("target_value_clip_min", -100.0),
        target_value_clip_max=cfg.get("target_value_clip_max", 100.0),
    )

    # Replay memory（尽量把新增参数传进去；不支持就回退）
    if use_per:
        per_kwargs = dict(
            size_max=cfg["memory_size_max"],
            size_min=cfg["memory_size_min"],
            alpha=float(cfg.get("per_alpha", extra.get("per_alpha", 0.6))),
            beta=float(cfg.get("per_beta", extra.get("per_beta", 0.4))),
            beta_increment=float(cfg.get("per_beta_increment", extra.get("per_beta_increment", 1e-3))),
            epsilon_priority=float(cfg.get("per_eps", extra.get("per_eps", 1e-6))),
        )
        # 新增：如果你的 PER 类已经支持，传 uniform_mix_ratio / priority_clip_max
        if "uniform_mix_ratio" in cfg:
            per_kwargs["uniform_mix_ratio"] = float(cfg["uniform_mix_ratio"])
        elif extra.get("uniform_mix_ratio") is not None:
            per_kwargs["uniform_mix_ratio"] = float(extra["uniform_mix_ratio"])
        if "priority_clip_max" in cfg:
            per_kwargs["priority_clip_max"] = float(cfg["priority_clip_max"])
        elif extra.get("priority_clip_max") is not None:
            per_kwargs["priority_clip_max"] = float(extra["priority_clip_max"])

        try:
            Memory = PrioritizedReplayMemory(**per_kwargs)
        except TypeError:
            # 兼容老版本 PER（无新增参数）
            per_kwargs.pop("uniform_mix_ratio", None)
            per_kwargs.pop("priority_clip_max", None)
            Memory = PrioritizedReplayMemory(**per_kwargs)
    else:
        from collections import deque
        import random
        class ReplayMemory:
            def __init__(self, size_max):
                self._buf = deque(maxlen=size_max)
                self._steps_done = 0
            def add_sample(self, sample):
                self._buf.append(sample)
                self._steps_done += 1
            def get_samples(self, n):
                if len(self._buf) < n:
                    return [], None, None
                batch = random.sample(self._buf, n)
                return batch, None, None
            def __len__(self): return len(self._buf)
            @property
            def steps_done(self): return self._steps_done
            @property
            def size_min(self): return 0  # for warmup check
        Memory = ReplayMemory(cfg["memory_size_max"])

    # Traffic generator & visualization
    TrafficGen = TrafficGenerator(cfg["max_steps"], cfg["n_cars_generated"])
    viz = Visualization(path, dpi=96)

    # Simulation（新增参数已接入：flow_reward_weight / 动作约束护栏）
    Simulation = D3QNSimulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        cfg["gamma"],
        cfg["max_steps"],
        cfg["green_duration"],
        cfg["yellow_duration"],
        cfg["num_states"],
        cfg["num_actions"],
        cfg["training_epochs"],
        warmup_min_samples=cfg["memory_size_min"],
        congestion_threshold=cfg.get("congestion_threshold", 20),
        congestion_penalty=cfg.get("congestion_penalty", -5.0),
        waiting_time_penalty_scale=cfg.get("waiting_time_penalty_scale", 1.0),
        state_normalization=cfg.get("state_normalization", "binary"),
        # 新增 ↓↓↓
        flow_reward_weight=cfg.get("flow_reward_weight", extra.get("flow_reward_weight", 0.05)),
        max_green=cfg.get("max_green", extra.get("max_green", 40)),
        imbalance_ratio=cfg.get("imbalance_ratio", extra.get("imbalance_ratio", 1.5)),
        max_wait_threshold=cfg.get("max_wait_threshold", extra.get("max_wait_threshold", 60.0)),
    )

    episode = 0
    total_steps = 0  # 跟踪总步数
    t0 = datetime.datetime.now()

    # 读取探索率参数
    epsilon_start = float(cfg.get("epsilon_start", 1.0))
    epsilon_end = float(cfg.get("epsilon_end", 0.05))
    decay_rate = float(cfg.get("decay_rate", 0.9995))
    decay_type = cfg.get("decay_type", "steps")

    print("开始 D3QN 交通信号控制训练...")
    print("模型配置:")
    print(f"- 网络层数: {cfg['num_layers']}")
    print(f"- 隐藏层宽度: {cfg['width_layers']}")
    print(f"- 批次大小: {cfg['batch_size']}")
    print(f"- 学习率: {cfg['learning_rate']}")
    print(f"- 折扣因子: {cfg['gamma']}")
    print(f"- 训练回合: {cfg['total_episodes']}")
    print(f"- 目标网络更新频率: {tuf}")
    print(f"- Dueling: 启用")
    print(f"- Double DQN: 启用")
    print(f"- 优先经验回放: {'启用' if use_per else '关闭'}")
    print(f"- 探索率衰减: 基于{decay_type}的指数衰减 ({decay_rate}^{decay_type})")
    print(f"- 探索率范围: {epsilon_start} → {epsilon_end}")

    while episode < cfg["total_episodes"]:
        print(f"\n----- Episode {episode + 1} of {cfg['total_episodes']}")

        # 基于步数或episode的指数衰减探索率
        if decay_type == "steps":
            epsilon = max(epsilon_end, epsilon_start * (decay_rate ** total_steps))
        else:  # episodes
            epsilon = max(epsilon_end, epsilon_start * (decay_rate ** episode))

        sim_time, train_time = Simulation.run(episode, epsilon)
        print(f"仿真时间: {sim_time}s - 训练时间: {train_time}s - 总计: {round(sim_time + train_time, 1)}s")

        # 更新总步数（每个episode的步数）
        episode_steps = Simulation.get_episode_steps() if hasattr(Simulation, 'get_episode_steps') else cfg["max_steps"]
        total_steps += episode_steps
        print(f"当前总步数: {total_steps}, 探索率: {epsilon:.4f}")

        # checkpoint
        if (episode + 1) % 10 == 0:
            ckpt = os.path.join(path, f"checkpoint_episode_{episode + 1}")
            os.makedirs(ckpt, exist_ok=True)
            Model.save_model(ckpt)
            print(f"模型检查点已保存到: {ckpt}")

        episode += 1

    print("\n----- 训练开始时间:", t0)
    print("----- 训练结束时间:", datetime.datetime.now())
    print("----- 会话信息保存在:", path)

    # Save final model & config
    Model.save_model(path)
    try:
        copyfile(src="training_settings.ini", dst=os.path.join(path, "training_settings.ini"))
    except Exception:
        pass

    # 保存训练经验数据用于第二阶段迁移学习（原逻辑保留）
    print("保存训练经验数据...")
    try:
        import numpy as np
        all_samples = []
        if hasattr(Simulation._Memory, '_buf'):  # 普通经验回放
            all_samples = list(Simulation._Memory._buf)
        elif hasattr(Simulation._Memory, '_sumtree'):
            # PER 的内部结构不一定有 _sumtree；用 _get_sample_by_index 兜底
            for i in range(len(Simulation._Memory)):
                if hasattr(Simulation._Memory, "_get_sample_by_index"):
                    sample = Simulation._Memory._get_sample_by_index(i)
                    if sample is not None:
                        all_samples.append(sample)

        if all_samples:
            states = np.array([s[0] for s in all_samples])
            actions = np.array([s[1] for s in all_samples])
            rewards = np.array([s[2] for s in all_samples])
            next_states = np.array([s[3] for s in all_samples])
            dones = np.array([s[4] for s in all_samples])

            experience_file = os.path.join(path, "phase1_experience.npz")
            np.savez_compressed(
                experience_file,
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones
            )
            print(f"经验数据已保存到: {experience_file}")
            print(f"保存了 {len(all_samples)} 条经验样本")
        else:
            print("警告: 无法获取经验数据")
    except Exception as e:
        print(f"保存经验数据失败: {e}")

    # Curves
    viz.save_data_and_plot(Simulation.reward_store, "d3qn_reward", "Episode", "累积负奖励")
    viz.save_data_and_plot(Simulation.cumulative_wait_store, "d3qn_delay", "Episode", "累积延迟 (近似)")
    viz.save_data_and_plot(Simulation.avg_queue_length_store, "d3qn_queue", "Episode", "平均队列长度 (车辆)")
    if Simulation.episode_rewards:
        viz.save_data_and_plot(Simulation.episode_rewards, "d3qn_episode_rewards", "Episode", "Episode奖励")
    if Simulation.episode_losses:
        viz.save_data_and_plot(Simulation.episode_losses, "d3qn_episode_losses", "Episode", "训练损失")

    print("D3QN 训练完成！")
    print("最终模型和训练结果已保存到:", path)

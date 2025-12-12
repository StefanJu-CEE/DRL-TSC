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


def epsilon_by_steps(total_steps: int, eps_start=1, eps_end=0.05, decay_rate=0.9998):
    """基于步数的指数衰减探索率"""
    return max(eps_end, eps_start * (decay_rate ** total_steps))


def epsilon_by_episodes(episode_num: int, eps_start=1, eps_end=0.05, decay_rate=0.97):
    """基于episodes的指数衰减探索率"""
    return max(eps_end, eps_start * (decay_rate ** episode_num))


def read_extra_from_ini(path="training_settings.ini"):
    """补充读取 [d3qn]、[PER]（兼容 utils 未解析这些键的情况）"""
    cp = configparser.ConfigParser()
    cp.read(path, encoding='utf-8')
    extra = {}
    if cp.has_section("d3qn"):
        extra["target_update_freq"] = cp["d3qn"].getint("target_update_freq", fallback=1000)
        if "use_per" not in extra:
            try:
                extra["use_per"] = cp["d3qn"].getboolean("use_per")
            except Exception:
                pass
        extra["per_alpha"] = cp["d3qn"].getfloat("per_alpha", fallback=None)
        extra["per_beta"] = cp["d3qn"].getfloat("per_beta", fallback=None)
        extra["per_eps"] = cp["d3qn"].getfloat("per_eps", fallback=None)
        extra["per_beta_increment"] = cp["d3qn"].getfloat("per_beta_increment", fallback=None)
    if cp.has_section("PER"):
        extra["use_per"] = cp["PER"].getboolean("use_per", fallback=extra.get("use_per", True))
        extra["per_alpha"] = cp["PER"].getfloat("per_alpha", fallback=extra.get("per_alpha", 0.6))
        extra["per_beta"] = cp["PER"].getfloat("per_beta", fallback=extra.get("per_beta", 0.4))
        extra["per_eps"] = cp["PER"].getfloat("per_eps", fallback=extra.get("per_eps", 1e-6))
        extra["per_beta_increment"] = cp["PER"].getfloat("per_beta_increment", fallback=extra.get("per_beta_increment", 1e-3))
        # 新增：混合采样比例与优先级裁剪（如 PER 类未实现，会自动回退）
        extra["uniform_mix_ratio"] = cp["PER"].getfloat("uniform_mix_ratio", fallback=None)
        extra["priority_clip_max"] = cp["PER"].getfloat("priority_clip_max", fallback=None)
    else:
        extra.setdefault("use_per", True)
        extra.setdefault("per_alpha", 0.6)
        extra.setdefault("per_beta", 0.4)
        extra.setdefault("per_eps", 1e-6)
        extra.setdefault("per_beta_increment", 1e-3)
    
    # 新增：读取 reward 和 action_constraints 参数
    if cp.has_section("reward"):
        extra["flow_reward_weight"] = cp["reward"].getfloat("flow_reward_weight", fallback=0.05)
        # 新增全局效率参数
        extra["global_efficiency_weight"] = cp["reward"].getfloat("global_efficiency_weight", fallback=0.1)
        extra["efficiency_threshold"] = cp["reward"].getfloat("efficiency_threshold", fallback=10.0)
        extra["efficiency_bonus"] = cp["reward"].getfloat("efficiency_bonus", fallback=2.0)
        extra["efficiency_penalty_multiplier"] = cp["reward"].getfloat("efficiency_penalty_multiplier", fallback=1.5)
    
    if cp.has_section("action_constraints"):
        extra["max_green"] = cp["action_constraints"].getint("max_green", fallback=40)
        extra["imbalance_ratio"] = cp["action_constraints"].getfloat("imbalance_ratio", fallback=1.5)
        extra["max_wait_threshold"] = cp["action_constraints"].getfloat("max_wait_threshold", fallback=60.0)
    
    return extra


if __name__ == "__main__":
    # 读取配置
    config = import_train_configuration(config_file="training_settings.ini")
    extra = read_extra_from_ini("training_settings.ini")

    # ★ 使用施工/突变期的 sumo 配置
    sumo_cmd = set_sumo(
        config["gui"],
        config.get("construction_sumocfg_file_name", config["construction_sumocfg_file_name"]),
        config["max_steps"],
    )
    out_dir = set_train_path(config["models_path_name"])

    # PER 开关/参数
    use_per = bool(config.get("use_per", extra.get("use_per", True)))
    per_alpha = float(config.get("per_alpha", extra.get("per_alpha", 0.6)))
    per_beta = float(config.get("per_beta", extra.get("per_beta", 0.4)))
    per_eps = float(config.get("per_eps", extra.get("per_eps", 1e-6)))
    per_beta_inc = float(config.get("per_beta_increment", extra.get("per_beta_increment", 1e-3)))
    uniform_mix_ratio = config.get("uniform_mix_ratio", extra.get("uniform_mix_ratio", None))
    priority_clip_max = config.get("priority_clip_max", extra.get("priority_clip_max", None))

    # 目标网络更新频率
    target_update_freq = int(config.get("target_update_freq", extra.get("target_update_freq", 1000)))

    # 模型（结构与阶段1一致；加入目标值裁剪）
    model = D3QNModel(
        num_layers=config["num_layers"],
        width=config["width_layers"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        input_dim=config["num_states"],
        output_dim=config["num_actions"],
        target_update_freq=target_update_freq,
        dueling=True,
        target_value_clip_min=config.get("target_value_clip_min", -100.0),
        target_value_clip_max=config.get("target_value_clip_max", 100.0),
    )

    # 迁移加载第一阶段模型（按你的目录命名保留）
    phase1_dir = os.path.join(config["models_path_name"], "model_34")
    if not os.path.isdir(phase1_dir):
        raise FileNotFoundError(f"[Transfer] 找不到阶段1模型目录: {phase1_dir}")
    model.load_model(phase1_dir)
    print("[Transfer] Loaded phase-1 model from:", phase1_dir)
    
    # 加载第一阶段的部分经验数据
    phase1_experience_file = os.path.join(phase1_dir, "phase1_experience.npz")
    if os.path.exists(phase1_experience_file):
        print("[Transfer] 找到第一阶段经验数据，准备加载...")
        phase1_experience_loaded = True
    else:
        print("[Transfer] 未找到第一阶段经验数据，将创建新的经验缓冲区")
        phase1_experience_loaded = False

    # 经验回放
    if use_per:
        per_kwargs = dict(
            size_max=config["memory_size_max"],
            size_min=config["memory_size_min"],
            alpha=per_alpha,
            beta=per_beta,
            beta_increment=per_beta_inc,
            epsilon_priority=per_eps,
        )
        if uniform_mix_ratio is not None:
            per_kwargs["uniform_mix_ratio"] = float(uniform_mix_ratio)
        if priority_clip_max is not None:
            per_kwargs["priority_clip_max"] = float(priority_clip_max)
        try:
            memory = PrioritizedReplayMemory(**per_kwargs)
        except TypeError:
            # 兼容旧版 PER（不支持新增参数）
            per_kwargs.pop("uniform_mix_ratio", None)
            per_kwargs.pop("priority_clip_max", None)
            memory = PrioritizedReplayMemory(**per_kwargs)
    else:
        # 简单均匀采样内存（支持 warm-up）
        from collections import deque
        import random
        class ReplayMemory:
            def __init__(self, size_max, size_min=0):
                self._buf = deque(maxlen=size_max)
                self._steps_done = 0
                self._size_min = int(size_min)
            def add_sample(self, sample):
                self._buf.append(sample)
                self._steps_done += 1
            def get_samples(self, n):
                if len(self._buf) < n or len(self._buf) < self._size_min:
                    return [], None, None
                batch = random.sample(self._buf, n)
                return batch, None, None
            def __len__(self): return len(self._buf)
            @property
            def steps_done(self): return self._steps_done
            @property
            def size_min(self): return self._size_min
        memory = ReplayMemory(config["memory_size_max"], size_min=config["memory_size_min"])
    
    # 加载第一阶段经验到缓冲区（按你的策略）
    if phase1_experience_loaded:
        try:
            import numpy as np
            data = np.load(phase1_experience_file, allow_pickle=True)
            s, a, r, ns, d = data['states'], data['actions'], data['rewards'], data['next_states'], data['dones']

            experience_retention_ratio = config.get("experience_retention_ratio", 0.2)
            max_buffer_usage = config.get("max_buffer_usage", 0.2)

            total = len(s)
            keep = min(int(total * experience_retention_ratio),
                       int(config["memory_size_max"] * max_buffer_usage))
            if keep > 0:
                idx = np.random.choice(total, keep, replace=False)
                for i in idx:
                    memory.add_sample((s[i], a[i], r[i], ns[i], d[i]))
                print(f"[Transfer] 成功加载 {keep} 条第一阶段经验到缓冲区；当前缓冲区大小: {len(memory)}")
            else:
                print("[Transfer] 缓冲区空间不足，未加载经验")
        except Exception as e:
            print(f"[Transfer] 加载第一阶段经验失败: {e}")
            print("[Transfer] 将使用空的经验缓冲区开始训练")
    else:
        print("[Transfer] 使用空的经验缓冲区开始训练")

    # 交通生成与可视化
    traffic_gen = TrafficGenerator(config["max_steps"], config["n_cars_generated"])
    viz = Visualization(out_dir, dpi=96)

    # 仿真器（传入 warm-up 与新增奖励/护栏参数）
    simulation = D3QNSimulation(
        Model=model,
        Memory=memory,
        TrafficGen=traffic_gen,
        sumo_cmd=sumo_cmd,
        gamma=config["gamma"],
        max_steps=config["max_steps"],
        green_duration=config["green_duration"],
        yellow_duration=config["yellow_duration"],
        num_states=config["num_states"],
        num_actions=config["num_actions"],
        training_epochs=config["training_epochs"],
        warmup_min_samples=config["memory_size_min"],
        congestion_threshold=config.get("congestion_threshold", 20),
        congestion_penalty=config.get("congestion_penalty", -5.0),
        waiting_time_penalty_scale=config.get("waiting_time_penalty_scale", 1.0),
        state_normalization=config.get("state_normalization", "binary"),
        # 新增 ↓↓↓
        flow_reward_weight=config.get("flow_reward_weight", extra.get("flow_reward_weight", 0.05)),
        max_green=config.get("max_green", extra.get("max_green", 40)),
        imbalance_ratio=config.get("imbalance_ratio", extra.get("imbalance_ratio", 1.5)),
        max_wait_threshold=config.get("max_wait_threshold", extra.get("max_wait_threshold", 60.0)),
        # 新增全局效率参数 ↓↓↓
        global_efficiency_weight=config.get("global_efficiency_weight", extra.get("global_efficiency_weight", 0.1)),
        efficiency_threshold=config.get("efficiency_threshold", extra.get("efficiency_threshold", 10.0)),
        efficiency_bonus=config.get("efficiency_bonus", extra.get("efficiency_bonus", 2.0)),
        efficiency_penalty_multiplier=config.get("efficiency_penalty_multiplier", extra.get("efficiency_penalty_multiplier", 1.5)),
    )

    total_episodes = config["total_episodes"]
    total_steps = 0  # 跟踪总步数
    print("开始 D3QN 阶段2（施工/封路突变）迁移训练...")
    print(f"- Double DQN: 启用")
    print(f"- Dueling: 启用")
    print(f"- 优先经验回放: {'启用' if use_per else '关闭'}")
    print(f"- 目标网更新频率: {target_update_freq}")
    print(f"- 探索率衰减: 基于episodes的指数衰减 (0.97^episodes)")

    for ep in range(total_episodes):
        # 第二阶段使用基于episodes的epsilon衰减
        eps = epsilon_by_episodes(ep, eps_start=1, eps_end=0.05, decay_rate=0.97)
        print(f"\n----- Episode {ep + 1}/{total_episodes} | epsilon={eps:.3f} | 总步数={total_steps}")
        sim_time, train_time = simulation.run(ep, eps)
        print("仿真:", sim_time, "s | 训练:", train_time, "s | 合计:", round(sim_time + train_time, 1), "s")
        
        # 更新总步数
        episode_steps = simulation.get_episode_steps() if hasattr(simulation, 'get_episode_steps') else config["max_steps"]
        total_steps += episode_steps

        if (ep + 1) % 10 == 0:
            ckpt = os.path.join(out_dir, f"checkpoint_ep_{ep + 1}")
            os.makedirs(ckpt, exist_ok=True)
            model.save_model(ckpt)
            print("Checkpoint ->", ckpt)

    # 保存与画图
    model.save_model(out_dir)
    try:
        copyfile("training_settings.ini", os.path.join(out_dir, "training_settings.ini"))
    except Exception:
        pass

    if getattr(simulation, "reward_store", None):
        viz.save_data_and_plot(simulation.reward_store, "d3qn_reward", "Episode", "Total Negative Reward")
    if getattr(simulation, "cumulative_wait_store", None):
        viz.save_data_and_plot(simulation.cumulative_wait_store, "d3qn_delay", "Episode", "Cumulative Delay (s)")
    if getattr(simulation, "avg_queue_length_store", None):
        viz.save_data_and_plot(simulation.avg_queue_length_store, "d3qn_queue", "Episode", "Avg Queue Length (veh)")
    if getattr(simulation, "episode_rewards", None):
        viz.save_data_and_plot(simulation.episode_rewards, "d3qn_episode_rewards", "Episode", "Episode Reward")
    if getattr(simulation, "episode_losses", None):
        viz.save_data_and_plot(simulation.episode_losses, "d3qn_episode_losses", "Episode", "Loss")

    print("D3QN 阶段2迁移训练完成。")
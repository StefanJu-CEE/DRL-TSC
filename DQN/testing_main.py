from __future__ import absolute_import, print_function

import os
from shutil import copyfile
import numpy as np
from testing_simulation import Simulation as TestingSimulation
from generator import TrafficGenerator
from improved_model import TestModel
from visualization import Visualization
from delay_visualization import DelayVisualization
from utils import import_test_configuration, set_sumo, set_test_path


def cat(a, b):
    """安全拼接两个序列（None/空都能处理）"""
    a = list(a) if a is not None else []
    b = list(b) if b is not None else []
    return a + b


if __name__ == "__main__":

    # ---------- 公共初始化 ----------
    cfg = import_test_configuration(config_file="testing_settings.ini")
    model_path, plot_path = set_test_path(cfg["models_path_name"], cfg["model_to_test"])

    model = TestModel(input_dim=cfg["num_states"], model_path=model_path)
    viz = Visualization(plot_path, dpi=96)
    delay_viz = DelayVisualization(plot_path, dpi=96)

    # 先复制一份配置以便溯源
    try:
        copyfile(src="testing_settings.ini", dst=os.path.join(plot_path, "testing_settings.ini"))
    except Exception:
        pass

    # ========== Phase 1：正常路网 ==========
    print("\n----- Phase 1: Normal conditions")

    sumo_cmd_p1 = set_sumo(cfg["gui"], cfg["sumocfg_file_name"], cfg["max_steps"])
    traffic_gen_p1 = TrafficGenerator(cfg["max_steps"], cfg["n_cars_generated"])

    sim1 = TestingSimulation(
        Model=model,
        TrafficGen=traffic_gen_p1,
        sumo_cmd=sumo_cmd_p1,
        max_steps=cfg["max_steps"],
        green_duration=cfg["green_duration"],
        yellow_duration=cfg["yellow_duration"],
        num_states=cfg["num_states"],
        num_actions=cfg["num_actions"],
        construction=False,
    )

    sim_time1 = sim1.run(cfg["episode_seed"])
    print("Phase 1 simulation time:", sim_time1, "s")

    # 收集 Phase 1 度量
    reward_1      = getattr(sim1, "reward_episode", []) or []
    queue_1       = getattr(sim1, "queue_length_episode", []) or []
    wait_step_1   = getattr(sim1, "waiting_time_episode", []) or []
    cum_delay_1   = getattr(sim1, "cumulative_wait_store", []) or []
    sum_wait_1    = getattr(sim1, "sum_waiting_time", None)

    # ========== Phase 2：施工/高流量 ==========
    print("\n----- Phase 2: 1000 vehicles with construction conditions")

    # 调整流量和随机种子
    n_cars_phase2 = 1000  # 你也可以从 ini 读取
    seed_phase2   = cfg["episode_seed"] + 1

    # 施工路网 sumocfg（不使用 route_file 参数，避免未定义）
    sumo_cmd_p2 = set_sumo(
        cfg["gui"],
        # 如果 testing_settings.ini 没有该键，会回退到普通 sumocfg
        cfg.get("construction_sumocfg_file_name", cfg["sumocfg_file_name"]),
        cfg["max_steps"],
    )
    traffic_gen_p2 = TrafficGenerator(cfg["max_steps"], n_cars_phase2)

    sim2 = TestingSimulation(
        Model=model,
        TrafficGen=traffic_gen_p2,
        sumo_cmd=sumo_cmd_p2,
        max_steps=cfg["max_steps"],
        green_duration=cfg["green_duration"],
        yellow_duration=cfg["yellow_duration"],
        num_states=cfg["num_states"],
        num_actions=cfg["num_actions"],
        construction=True,
    )

    sim_time2 = sim2.run(seed_phase2)
    print("Phase 2 simulation time:", sim_time2, "s")
    print("----- Testing info saved at:", plot_path)

    # 收集 Phase 2 度量
    reward_2      = getattr(sim2, "reward_episode", []) or []
    queue_2       = getattr(sim2, "queue_length_episode", []) or []
    wait_step_2   = getattr(sim2, "waiting_time_episode", []) or []
    cum_delay_2   = getattr(sim2, "cumulative_wait_store", []) or []
    sum_wait_2    = getattr(sim2, "sum_waiting_time", None)

    # ========== 合并 & 可视化到同一组图 ==========
    reward_all    = cat(reward_1,    reward_2)
    queue_all     = cat(queue_1,     queue_2)
    wait_step_all = cat(wait_step_1, wait_step_2)
    cum_delay_all = cat(cum_delay_1, cum_delay_2)

    # 单目录、单文件名（覆盖式，不区分 phase）
    if reward_all:
        viz.save_data_and_plot(reward_all, "reward_combined", "Action step", "Reward")
    if queue_all:
        viz.save_data_and_plot(queue_all, "queue_combined", "Step", "Queue length (vehicles)")
    if cum_delay_all:
        viz.save_data_and_plot(cum_delay_all, "cumulative_delay_combined", "Episode/Phase", "Cumulative Delay (s)")
    if wait_step_all:
        viz.save_data_and_plot(wait_step_all, "step_delay_combined", "Step", "Waiting Time (s)")

    # 延迟综合分析：用合并后的四条序列
    if wait_step_all:
        print("\n----- 生成综合延迟分析（合并两阶段）...")
        delay_viz.plot_comprehensive_delay_analysis(
            wait_step_all,      # 每步等待
            cum_delay_all,      # 累积延迟（如果是按回合统计，也会显示两点/两段）
            queue_all,          # 队列长度
            reward_all          # 每步奖励
        )
        delay_viz.plot_delay_statistics(
            wait_step_all,
            cum_delay_all
        )
        delay_viz.generate_delay_report(
            wait_step_all,
            cum_delay_all,
            queue_all,
            reward_all
        )
        print("----- 延迟分析完成！")

    # 汇总打印
    if sum_wait_1 is not None or sum_wait_2 is not None:
        total_cum_delay = (sum_wait_1 or 0.0) + (sum_wait_2 or 0.0)
        print(f"\n===== 汇总：两阶段总累积延迟: {total_cum_delay:.2f} 秒 =====")
        if wait_step_all:
            print(f"平均每步延迟: {np.mean(wait_step_all):.2f} 秒 | 最大单步延迟: {np.max(wait_step_all):.2f} 秒")

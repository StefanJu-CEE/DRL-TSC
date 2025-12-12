from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile
import numpy as np

from testing_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path
from delay_visualization import DelayVisualization
from d3qn_model import TestModel


if __name__ == "__main__":
    # 读取配置
    config = import_test_configuration(config_file='testing_settings.ini')

    # Phase 1：正常路网
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    traffic_gen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    dpi_used = int(config.get('delay_plot_dpi', 120))
    visualization = Visualization(plot_path, dpi=dpi_used)
    delay_viz = DelayVisualization(plot_path, dpi=dpi_used)

    # 新增参数：tiny_eval_epsilon / warmup_steps / relief_check_near_cells / starvation_bias
    simulation = Simulation(
        Model=model,
        TrafficGen=traffic_gen,
        sumo_cmd=sumo_cmd,
        max_steps=config['max_steps'],
        green_duration=config['green_duration'],
        yellow_duration=config['yellow_duration'],
        num_states=config['num_states'],
        num_actions=config['num_actions'],
        global_seed_offset=int(config.get('global_seed_offset', 1234)),
        use_action_mask=bool(config.get('use_action_mask', True)),
        min_green=int(config.get('min_green', 8)),
        max_green=int(config.get('max_green', 40)),
        no_relief_limit=int(config.get('no_relief_limit', 3)),
        relief_drop=int(config.get('relief_drop', 1)),
        state_normalization=config.get('state_normalization', 'z_score'),
        tiny_eval_epsilon=float(config.get('tiny_eval_epsilon', 0.02)),
        relief_check_near_cells=int(config.get('relief_check_near_cells', 3)),
        waiting_time_penalty_scale=config.get('waiting_time_penalty_scale', 0.0),
        congestion_threshold=config.get('congestion_threshold', 25),
        congestion_penalty=config.get('congestion_penalty', -10),
        flow_reward_weight=config.get('flow_reward_weight', 0.15),
    )

    print('\n----- Phase 1: Normal conditions')
    result1 = simulation.run(config['episode_seed'])
    print('Simulation time:', result1.get('simulation_time', 0.0), 's')

    print("----- Testing info saved at:", plot_path)
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    visualization.save_data_and_plot(simulation.reward_episode, 'reward', 'Action step', 'Reward')
    visualization.save_data_and_plot(simulation.queue_length_episode, 'queue', 'Step', 'Queue length (vehicles)')

    if getattr(simulation, 'cumulative_wait_store', None):
        visualization.save_data_and_plot(simulation.cumulative_wait_store, 'cumulative_delay', 'Episode', 'Cumulative Delay (s)')
    if getattr(simulation, 'waiting_time_episode', None):
        visualization.save_data_and_plot(simulation.waiting_time_episode, 'step_delay', 'Step', 'Waiting Time (s)')

        print("\n----- 生成延迟分析图表...")
        delay_viz.plot_comprehensive_delay_analysis(
            simulation.waiting_time_episode,
            simulation.cumulative_wait_store,
            simulation.queue_length_episode,
            simulation.reward_episode
        )
        delay_viz.plot_delay_statistics(
            simulation.waiting_time_episode,
            simulation.cumulative_wait_store
        )
        delay_viz.generate_delay_report(
            simulation.waiting_time_episode,
            simulation.cumulative_wait_store,
            simulation.queue_length_episode,
            simulation.reward_episode
        )
        print("----- 延迟分析完成！")

        print(f"----- 总累积延迟: {simulation.sum_waiting_time:.2f} 秒")
        avg_delay = float(np.mean(simulation.waiting_time_episode)) if simulation.waiting_time_episode else 0.0
        max_delay = float(np.max(simulation.waiting_time_episode)) if simulation.waiting_time_episode else 0.0
        print(f"----- 平均每步延迟: {avg_delay:.2f} 秒")
        print(f"----- 最大单步延迟: {max_delay:.2f} 秒")

    # Phase 2：施工/封闭路网
    print('\n----- Phase 2: Construction conditions')
    phase2_plot_path = os.path.join(plot_path, 'phase2')
    os.makedirs(phase2_plot_path, exist_ok=True)

    config['n_cars_generated'] = int(config.get('n_cars_generated', 2000))
    config['episode_seed'] = int(config['episode_seed']) + 1

    traffic_gen2 = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    sumo_cmd2 = set_sumo(config['gui'], config.get('sumocfg_file_name', config['sumocfg_file_name']), config['max_steps'])

    visualization2 = Visualization(phase2_plot_path, dpi=dpi_used)
    delay_viz2 = DelayVisualization(phase2_plot_path, dpi=dpi_used)

    simulation2 = Simulation(
        model,
        traffic_gen2,
        sumo_cmd2,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        global_seed_offset = int(config.get('global_seed_offset', 1234)),
        use_action_mask    = bool(config.get('use_action_mask', True)),
        min_green          = int(config.get('min_green', 8)),
        max_green          = int(config.get('max_green', 40)),
        no_relief_limit    = int(config.get('no_relief_limit', 3)),
        relief_drop        = int(config.get('relief_drop', 1)),
        state_normalization= config.get('state_normalization', 'z_score'),
        tiny_eval_epsilon  = float(config.get('tiny_eval_epsilon', 0.02)),
        warmup_steps       = int(config.get('warmup_steps', 120)),
        relief_check_near_cells = int(config.get('relief_check_near_cells', 3)),
        starvation_bias    = float(config.get('starvation_bias', 0.15)),
    )

    result2 = simulation2.run(config['episode_seed'])
    print('Simulation time:', result2.get('simulation_time', 0.0), 's')

    print("----- Testing info saved at:", phase2_plot_path)
    copyfile(src='testing_settings.ini', dst=os.path.join(phase2_plot_path, 'testing_settings.ini'))

    visualization2.save_data_and_plot(simulation2.reward_episode, 'reward', 'Action step', 'Reward')
    visualization2.save_data_and_plot(simulation2.queue_length_episode, 'queue', 'Step', 'Queue length (vehicles)')

    if getattr(simulation2, 'cumulative_wait_store', None):
        visualization2.save_data_and_plot(simulation2.cumulative_wait_store, 'cumulative_delay', 'Episode', 'Cumulative Delay (s)')
    if getattr(simulation2, 'waiting_time_episode', None):
        visualization2.save_data_and_plot(simulation2.waiting_time_episode, 'step_delay', 'Step', 'Waiting Time (s)')

        print("\n----- 生成延迟分析图表...")
        delay_viz2.plot_comprehensive_delay_analysis(
            simulation2.waiting_time_episode,
            simulation2.cumulative_wait_store,
            simulation2.queue_length_episode,
            simulation2.reward_episode
        )
        delay_viz2.plot_delay_statistics(
            simulation2.waiting_time_episode,
            simulation2.cumulative_wait_store
        )
        delay_viz2.generate_delay_report(
            simulation2.waiting_time_episode,
            simulation2.cumulative_wait_store,
            simulation2.queue_length_episode,
            simulation2.reward_episode
        )
        print("----- 延迟分析完成！")

        print(f"----- 总累积延迟: {simulation2.sum_waiting_time:.2f} 秒")
        avg_delay2 = float(np.mean(simulation2.waiting_time_episode)) if simulation2.waiting_time_episode else 0.0
        max_delay2 = float(np.max(simulation2.waiting_time_episode)) if simulation2.waiting_time_episode else 0.0
        print(f"----- 平均每步延迟: {avg_delay2:.2f} 秒")
        print(f"----- 最大单步延迟: {max_delay2:.2f} 秒")

from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from improved_training_simulation import ImprovedSimulation
from generator import TrafficGenerator
from improved_model import ImprovedTrainModel, ImprovedMemory
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path



def run_phase_training(phase_name, config, Model, Memory, TrafficGen, Visualization, 
                      sumo_cmd, episodes_per_phase=50):
    """
    运行单个阶段的训练
    """
    print(f"\n{'='*config['total_episodes']}")
    print(f"开始 {phase_name} 训练")
    print(f"{'='*config['total_episodes']}")
    
    simulation = ImprovedSimulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )
    
    episode = 0
    phase_start_time = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print(f'\n----- {phase_name} Episode {episode+1} of {config["total_episodes"]}')
        epsilon = 1.0 - (episode / config['total_episodes'])  # epsilon-greedy policy
        simulation_time, training_time = simulation.run(episode, epsilon)
        print(f'Simulation time: {simulation_time} s - Training time: {training_time} s - Total: {round(simulation_time+training_time, 1)} s')
        print(f'Learn steps: {Model.learn_steps} - Gamma: {Model.gamma:.3f}')
        episode += 1
    
    phase_end_time = datetime.datetime.now()
    print(f"\n----- {phase_name} 训练完成")
    print(f"----- 开始时间: {phase_start_time}")
    print(f"----- 结束时间: {phase_end_time}")
    
    return simulation, phase_start_time, phase_end_time


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings.ini')
    
    sumo_cmd_construction = set_sumo(config['gui'], config.get('construction_sumocfg_file_name', config['construction_sumocfg_file_name']), config['max_steps'])
    
    # 创建模型保存路径
    path = set_train_path(config['models_path_name'])

    # 创建改进的模型
    Model = ImprovedTrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions'],
        use_sgd=False,  # 使用Adam优化器
        target_update_freq=3000  # 目标网络更新频率
    )
    phase1_dir = 'models/model_15'
    Model.load_model(phase1_dir)
    print("[Transfer] Loaded phase-1 model from:", phase1_dir)

    Memory = ImprovedMemory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
    
    sim_phase2, start2, end2 = run_phase_training(
        "阶段二：施工道路", config, Model, Memory, TrafficGen, Visualization,
        sumo_cmd_construction
    )
    total_episodes = config['total_episodes']

    # 保存最终模型
    Model.save_model(path)
    
    # 保存配置文件
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
    
    # 保存训练结果
    print("\n----- 保存训练结果")
    
    
    # 阶段二结果
    Visualization.save_data_and_plot(
        data=sim_phase2.reward_store, 
        filename='improved_reward_phase2', 
        xlabel='Episode', 
        ylabel='Cumulative negative reward'
    )
    Visualization.save_data_and_plot(
        data=sim_phase2.cumulative_wait_store, 
        filename='improved_delay_phase2', 
        xlabel='Episode', 
        ylabel='Cumulative delay (s)'
    )
    Visualization.save_data_and_plot(
        data=sim_phase2.avg_queue_length_store, 
        filename='improved_queue_phase2', 
        xlabel='Episode', 
        ylabel='Average queue length (vehicles)'
    )
    
    # 新增：损失函数可视化
    print("\n----- 生成损失函数可视化")
    
    # 绘制损失函数变化
    Visualization.plot_loss_function(
        loss_history=Model.loss_history,
        episode_losses=sim_phase2.avg_loss_store,
        filename='improved_loss_function_phase2'
    )
    
    # 绘制综合训练指标
    Visualization.plot_comprehensive_training_metrics(
        reward_store=sim_phase2.reward_store,
        cumulative_wait_store=sim_phase2.cumulative_wait_store,
        avg_queue_length_store=sim_phase2.avg_queue_length_store,
        avg_loss_store=sim_phase2.avg_loss_store,
        filename='improved_comprehensive_metrics_phase2'
    )
    
    print(f"----- 训练完成！结果保存在: {path}")
    print(f"----- 最终学习步数: {Model.learn_steps}")
    print(f"----- 总训练步数: {len(Model.loss_history)}")
    if sim_phase2.avg_loss_store:
        print(f"----- 最终Episode平均损失: {sim_phase2.avg_loss_store[-1]:.6f}")
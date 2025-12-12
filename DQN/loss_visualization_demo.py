#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
损失函数可视化演示脚本
展示如何使用新添加的损失可视化功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from improved_model import ImprovedTrainModel
from visualization import Visualization

def create_demo_loss_data():
    """
    创建演示用的损失数据
    """
    # 模拟训练过程中的损失变化（从高到低，带有噪声）
    np.random.seed(42)
    n_training_steps = 1000
    n_episodes = 50
    
    # 生成训练损失（逐渐下降，带有噪声）
    base_loss = np.linspace(1.0, 0.1, n_training_steps)
    noise = np.random.normal(0, 0.05, n_training_steps)
    training_losses = np.maximum(0, base_loss + noise)
    
    # 生成episode平均损失
    episode_losses = []
    steps_per_episode = n_training_steps // n_episodes
    
    for i in range(n_episodes):
        start_idx = i * steps_per_episode
        end_idx = min((i + 1) * steps_per_episode, n_training_steps)
        episode_avg = np.mean(training_losses[start_idx:end_idx])
        episode_losses.append(episode_avg)
    
    return training_losses, episode_losses

def demo_loss_visualization():
    """
    演示损失可视化功能
    """
    print("=== DQN模型损失函数可视化演示 ===\n")
    
    # 创建演示数据
    training_losses, episode_losses = create_demo_loss_data()
    
    print(f"生成了 {len(training_losses)} 个训练步骤的损失数据")
    print(f"生成了 {len(episode_losses)} 个episode的平均损失数据")
    
    # 创建可视化对象
    output_path = "demo_output"
    os.makedirs(output_path, exist_ok=True)
    viz = Visualization(output_path, dpi=96)
    
    print(f"\n输出路径: {output_path}")
    
    # 1. 绘制损失函数变化
    print("\n1. 生成损失函数可视化...")
    viz.plot_loss_function(
        loss_history=training_losses,
        episode_losses=episode_losses,
        filename='demo_loss_function'
    )
    
    # 2. 绘制综合训练指标（模拟数据）
    print("\n2. 生成综合训练指标图表...")
    # 创建模拟的其他指标数据
    reward_store = np.random.uniform(-100, -10, len(episode_losses))
    cumulative_wait_store = np.random.uniform(100, 1000, len(episode_losses))
    avg_queue_length_store = np.random.uniform(5, 25, len(episode_losses))
    
    viz.plot_comprehensive_training_metrics(
        reward_store=reward_store,
        cumulative_wait_store=cumulative_wait_store,
        avg_queue_length_store=avg_queue_length_store,
        avg_loss_store=episode_losses,
        filename='demo_comprehensive_metrics'
    )
    
    # 3. 创建模型并演示损失记录功能
    print("\n3. 演示模型损失记录功能...")
    
    # 创建模型实例
    model = ImprovedTrainModel(
        num_layers=3,
        width=64,
        batch_size=32,
        learning_rate=0.001,
        input_dim=80,
        output_dim=4
    )
    
    # 模拟训练过程，记录损失
    print("模拟训练过程...")
    for i, loss in enumerate(training_losses[:100]):  # 只模拟前100步
        # 模拟训练步骤
        model._loss_history.append(loss)
        if i % 20 == 0:  # 每20步开始新episode
            model.start_new_episode()
            model._episode_losses.extend(training_losses[i:i+20])
    
    print(f"模型记录了 {len(model.loss_history)} 个损失值")
    print(f"当前episode有 {len(model.episode_losses)} 个损失值")
    
    # 4. 保存模型（包含损失历史）
    print("\n4. 保存模型和损失历史...")
    model.save_model(output_path)
    
    print("\n=== 演示完成 ===")
    print(f"所有文件已保存到: {output_path}")
    print("\n生成的文件包括:")
    print("- demo_loss_function.png: 损失函数变化图")
    print("- demo_comprehensive_metrics.png: 综合训练指标图")
    print("- loss_history.txt: 损失历史数据")
    print("- improved_training_params.npy: 训练参数和损失历史")

def analyze_loss_patterns():
    """
    分析损失模式
    """
    print("\n=== 损失模式分析 ===")
    
    training_losses, episode_losses = create_demo_loss_data()
    
    # 计算统计信息
    print(f"训练损失统计:")
    print(f"  最小值: {np.min(training_losses):.4f}")
    print(f"  最大值: {np.max(training_losses):.4f}")
    print(f"  平均值: {np.mean(training_losses):.4f}")
    print(f"  标准差: {np.std(training_losses):.4f}")
    
    print(f"\nEpisode损失统计:")
    print(f"  最小值: {np.min(episode_losses):.4f}")
    print(f"  最大值: {np.max(episode_losses):.4f}")
    print(f"  平均值: {np.mean(episode_losses):.4f}")
    print(f"  标准差: {np.std(episode_losses):.4f}")
    
    # 计算收敛性指标
    first_quarter = training_losses[:len(training_losses)//4]
    last_quarter = training_losses[-len(training_losses)//4:]
    
    convergence_ratio = np.mean(last_quarter) / np.mean(first_quarter)
    print(f"\n收敛性分析:")
    print(f"  前1/4平均损失: {np.mean(first_quarter):.4f}")
    print(f"  后1/4平均损失: {np.mean(last_quarter):.4f}")
    print(f"  收敛比例: {convergence_ratio:.4f}")
    
    if convergence_ratio < 0.5:
        print("  ✓ 模型收敛良好")
    elif convergence_ratio < 0.8:
        print("  ⚠ 模型收敛一般")
    else:
        print("  ✗ 模型收敛较差")

if __name__ == "__main__":
    try:
        # 运行演示
        demo_loss_visualization()
        
        # 分析损失模式
        analyze_loss_patterns()
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

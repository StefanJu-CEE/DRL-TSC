"""
延迟可视化专用脚本
提供交通信号控制系统的延迟分析和可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from visualization import Visualization

class DelayVisualization:
    def __init__(self, plot_path, dpi=96):
        self.plot_path = plot_path
        self.dpi = dpi
        self.plt = plt
        
    def plot_comprehensive_delay_analysis(self, waiting_time_episode, cumulative_wait_store, 
                                        queue_length_episode, reward_episode):
        """
        综合延迟分析可视化
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('交通信号控制延迟综合分析', fontsize=16, fontweight='bold')
        
        # 1. 每步等待时间
        axes[0, 0].plot(waiting_time_episode, 'b-', linewidth=1.5, alpha=0.8)
        axes[0, 0].set_title('每步等待时间变化')
        axes[0, 0].set_xlabel('仿真步数')
        axes[0, 0].set_ylabel('等待时间 (秒)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 累积延迟
        if cumulative_wait_store:
            axes[0, 1].plot(cumulative_wait_store, 'r-', linewidth=2, marker='o', markersize=4)
            axes[0, 1].set_title('累积延迟变化')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('累积延迟 (秒)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 队列长度与延迟关系
        if len(queue_length_episode) == len(waiting_time_episode):
            axes[1, 0].scatter(queue_length_episode, waiting_time_episode, 
                              alpha=0.6, c='green', s=20)
            axes[1, 0].set_title('队列长度 vs 等待时间')
            axes[1, 0].set_xlabel('队列长度 (车辆数)')
            axes[1, 0].set_ylabel('等待时间 (秒)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 奖励与延迟关系
        if len(reward_episode) == len(waiting_time_episode):
            axes[1, 1].scatter(waiting_time_episode, reward_episode, 
                              alpha=0.6, c='orange', s=20)
            axes[1, 1].set_title('等待时间 vs 奖励')
            axes[1, 1].set_xlabel('等待时间 (秒)')
        axes[1, 1].set_ylabel('奖励值')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        delay_analysis_path = os.path.join(self.plot_path, 'comprehensive_delay_analysis.png')
        plt.savefig(delay_analysis_path, dpi=self.dpi, bbox_inches='tight')
        print(f"综合延迟分析图已保存到: {delay_analysis_path}")
        
        return delay_analysis_path
    
    def plot_delay_statistics(self, waiting_time_episode, cumulative_wait_store):
        """
        延迟统计信息可视化
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('延迟统计信息', fontsize=14, fontweight='bold')
        
        # 延迟分布直方图
        if waiting_time_episode:
            axes[0].hist(waiting_time_episode, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].axvline(np.mean(waiting_time_episode), color='red', linestyle='--', 
                           label=f'平均值: {np.mean(waiting_time_episode):.2f}s')
            axes[0].axvline(np.median(waiting_time_episode), color='green', linestyle='--', 
                           label=f'中位数: {np.median(waiting_time_episode):.2f}s')
            axes[0].set_title('等待时间分布')
            axes[0].set_xlabel('等待时间 (秒)')
            axes[0].set_ylabel('频次')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # 累积延迟趋势
        if cumulative_wait_store:
            axes[1].plot(cumulative_wait_store, 'b-', linewidth=2, marker='s', markersize=4)
            axes[1].set_title('累积延迟趋势')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('累积延迟 (秒)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        delay_stats_path = os.path.join(self.plot_path, 'delay_statistics.png')
        plt.savefig(delay_stats_path, dpi=self.dpi, bbox_inches='tight')
        print(f"延迟统计图已保存到: {delay_stats_path}")
        
        return delay_stats_path
    
    def generate_delay_report(self, waiting_time_episode, cumulative_wait_store, 
                            queue_length_episode, reward_episode):
        """
        生成延迟分析报告
        """
        report_path = os.path.join(self.plot_path, 'delay_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("交通信号控制延迟分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本统计信息
            f.write("1. 基本统计信息\n")
            f.write("-" * 30 + "\n")
            if waiting_time_episode:
                f.write(f"总仿真步数: {len(waiting_time_episode)}\n")
                f.write(f"平均等待时间: {np.mean(waiting_time_episode):.2f} 秒\n")
                f.write(f"最大等待时间: {np.max(waiting_time_episode):.2f} 秒\n")
                f.write(f"最小等待时间: {np.min(waiting_time_episode):.2f} 秒\n")
                f.write(f"等待时间标准差: {np.std(waiting_time_episode):.2f} 秒\n")
                f.write(f"等待时间中位数: {np.median(waiting_time_episode):.2f} 秒\n\n")
            
            # 累积延迟信息
            if cumulative_wait_store:
                f.write("2. 累积延迟信息\n")
                f.write("-" * 30 + "\n")
                f.write(f"总累积延迟: {cumulative_wait_store[-1]:.2f} 秒\n")
                f.write(f"平均每步累积延迟: {np.mean(cumulative_wait_store):.2f} 秒\n\n")
            
            # 队列长度分析
            if queue_length_episode:
                f.write("3. 队列长度分析\n")
                f.write("-" * 30 + "\n")
                f.write(f"平均队列长度: {np.mean(queue_length_episode):.2f} 车辆\n")
                f.write(f"最大队列长度: {np.max(queue_length_episode):.2f} 车辆\n")
                f.write(f"队列长度标准差: {np.std(queue_length_episode):.2f} 车辆\n\n")
            
            # 奖励分析
            if reward_episode:
                f.write("4. 奖励分析\n")
                f.write("-" * 30 + "\n")
                f.write(f"总奖励: {np.sum(reward_episode):.2f}\n")
                f.write(f"平均奖励: {np.mean(reward_episode):.2f}\n")
                f.write(f"最大奖励: {np.max(reward_episode):.2f}\n")
                f.write(f"最小奖励: {np.min(reward_episode):.2f}\n\n")
            
            # 性能评估
            f.write("5. 性能评估\n")
            f.write("-" * 30 + "\n")
            if waiting_time_episode and queue_length_episode:
                # 计算效率指标
                avg_wait = np.mean(waiting_time_episode)
                avg_queue = np.mean(queue_length_episode)
                efficiency_score = 100 / (1 + avg_wait * 0.1 + avg_queue * 0.05)
                f.write(f"系统效率评分: {efficiency_score:.2f}/100\n")
                
                if avg_wait < 10:
                    f.write("等待时间表现: 优秀\n")
                elif avg_wait < 20:
                    f.write("等待时间表现: 良好\n")
                elif avg_wait < 30:
                    f.write("等待时间表现: 一般\n")
                else:
                    f.write("等待时间表现: 需要改进\n")
        
        print(f"延迟分析报告已保存到: {report_path}")
        return report_path

def main():
    """
    延迟可视化演示
    """
    print("延迟可视化模块已加载")
    print("使用方法:")
    print("1. 在测试主方法中导入: from delay_visualization import DelayVisualization")
    print("2. 创建实例: delay_viz = DelayVisualization(plot_path)")
    print("3. 调用可视化方法")

if __name__ == "__main__":
    main()

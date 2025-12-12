import matplotlib.pyplot as plt
import os
import numpy as np

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = min(data)
        max_val = max(data)

        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)

    def plot_loss_function(self, loss_history, episode_losses=None, filename='loss_function'):
        """
        可视化损失函数的变化
        Args:
            loss_history: 完整的损失历史（每次训练的损失）
            episode_losses: episode平均损失（可选）
            filename: 保存的文件名
        """
        plt.rcParams.update({'font.size': 16})
        
        # 创建子图
        if episode_losses is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
        
        # 绘制训练过程中的损失变化
        ax1.plot(loss_history, 'b-', alpha=0.7, linewidth=0.8, label='训练损失')
        
        # 计算移动平均，使曲线更平滑
        if len(loss_history) > 100:
            window_size = min(100, len(loss_history) // 10)
            moving_avg = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(loss_history)), moving_avg, 'r-', 
                    linewidth=2, label=f'移动平均 (窗口={window_size})')
        
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('损失值')
        ax1.set_title('训练过程中的损失变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 如果提供了episode损失，绘制episode平均损失
        if episode_losses is not None:
            ax2.plot(episode_losses, 'g-', linewidth=2, marker='o', markersize=4, label='Episode平均损失')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('平均损失')
            ax2.set_title('Episode平均损失变化')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi, bbox_inches='tight')
        plt.close("all")
        
        # 保存损失数据
        with open(os.path.join(self._path, 'plot_'+filename+'_data.txt'), "w") as file:
            file.write("训练损失历史:\n")
            for i, loss in enumerate(loss_history):
                file.write(f"{i+1}\t{loss}\n")
            
            if episode_losses is not None:
                file.write("\nEpisode平均损失:\n")
                for i, loss in enumerate(episode_losses):
                    file.write(f"{i+1}\t{loss}\n")
        
        print(f"损失函数可视化已保存到: {os.path.join(self._path, 'plot_'+filename+'.png')}")

    def plot_comprehensive_training_metrics(self, reward_store, cumulative_wait_store, avg_queue_length_store, 
                                          avg_loss_store=None, filename='comprehensive_training_metrics'):
        """
        绘制综合训练指标图表
        """
        plt.rcParams.update({'font.size': 14})
        
        # 确定子图数量
        num_plots = 3 if avg_loss_store is None else 4
        fig, axes = plt.subplots(num_plots, 1, figsize=(16, 4*num_plots))
        
        if num_plots == 3:
            ax1, ax2, ax3 = axes
        else:
            ax1, ax2, ax3, ax4 = axes
        
        # 1. 奖励变化
        ax1.plot(reward_store, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('总奖励')
        ax1.set_title('训练过程中的奖励变化')
        ax1.grid(True, alpha=0.3)
        
        # 2. 累积等待时间
        ax2.plot(cumulative_wait_store, 'r-', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('累积等待时间')
        ax2.set_title('累积等待时间变化')
        ax2.grid(True, alpha=0.3)
        
        # 3. 平均队列长度
        ax3.plot(avg_queue_length_store, 'g-', linewidth=2, marker='^', markersize=3)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('平均队列长度')
        ax3.set_title('平均队列长度变化')
        ax3.grid(True, alpha=0.3)
        
        # 4. 平均损失（如果提供）
        if avg_loss_store is not None:
            ax4.plot(avg_loss_store, 'm-', linewidth=2, marker='d', markersize=3)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('平均损失')
            ax4.set_title('Episode平均损失变化')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi, bbox_inches='tight')
        plt.close("all")
        
        print(f"综合训练指标图表已保存到: {os.path.join(self._path, 'plot_'+filename+'.png')}")
    
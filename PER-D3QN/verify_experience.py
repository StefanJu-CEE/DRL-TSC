#!/usr/bin/env python3
"""
经验数据验证工具
用于检查保存的经验文件是否正确
"""

import os
import numpy as np
import json
from pathlib import Path

def verify_experience_file(file_path):
    """验证经验数据文件"""
    print(f"🔍 验证经验文件: {file_path}")
    print("=" * 60)
    
    try:
        # 加载数据
        data = np.load(file_path, allow_pickle=True)
        
        print("✅ 文件加载成功")
        print(f"📊 文件大小: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
        print(f"📁 包含的键: {list(data.keys())}")
        
        # 检查基本数据
        if 'states' in data:
            states = data['states']
            print(f"✅ 状态数据:")
            print(f"   - 形状: {states.shape}")
            print(f"   - 数据类型: {states.dtype}")
            print(f"   - 值范围: [{states.min():.4f}, {states.max():.4f}]")
            print(f"   - 非零元素: {np.count_nonzero(states)}")
        
        if 'actions' in data:
            actions = data['actions']
            print(f"✅ 动作数据:")
            print(f"   - 数量: {len(actions)}")
            print(f"   - 数据类型: {actions.dtype}")
            print(f"   - 动作分布: {np.bincount(actions)}")
        
        if 'rewards' in data:
            rewards = data['rewards']
            print(f"✅ 奖励数据:")
            print(f"   - 数量: {len(rewards)}")
            print(f"   - 数据类型: {rewards.dtype}")
            print(f"   - 值范围: [{rewards.min():.4f}, {rewards.max():.4f}]")
            print(f"   - 均值: {rewards.mean():.4f}")
            print(f"   - 标准差: {rewards.std():.4f}")
        
        if 'next_states' in data:
            next_states = data['next_states']
            print(f"✅ 下一状态数据:")
            print(f"   - 形状: {next_states.shape}")
            print(f"   - 数据类型: {next_states.dtype}")
        
        if 'dones' in data:
            dones = data['dones']
            print(f"✅ 完成状态数据:")
            print(f"   - 数量: {len(dones)}")
            print(f"   - 完成数量: {np.sum(dones)}")
            print(f"   - 完成比例: {np.sum(dones) / len(dones):.2%}")
        
        # 检查元数据
        if 'retention_info' in data:
            retention_info = data['retention_info'].item()
            print(f"✅ 保留信息:")
            for key, value in retention_info.items():
                print(f"   - {key}: {value}")
        
        if 'checkpoint_info' in data:
            checkpoint_info = data['checkpoint_info'].item()
            print(f"✅ 检查点信息:")
            for key, value in checkpoint_info.items():
                print(f"   - {key}: {value}")
        
        # 数据一致性检查
        print("\n🔍 数据一致性检查:")
        lengths = []
        for key in ['states', 'actions', 'rewards', 'next_states', 'dones']:
            if key in data:
                lengths.append(len(data[key]))
                print(f"   - {key}: {len(data[key])}")
        
        if len(set(lengths)) == 1:
            print("✅ 所有数据长度一致")
        else:
            print("❌ 数据长度不一致！")
            return False
        
        print("\n✅ 经验文件验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 文件验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_experience_files(models_path):
    """查找所有经验数据文件"""
    experience_files = []
    
    for root, dirs, files in os.walk(models_path):
        for file in files:
            if file.endswith('.npz') and ('experience' in file.lower() or 'checkpoint' in file.lower()):
                file_path = os.path.join(root, file)
                experience_files.append(file_path)
    
    return experience_files

def main():
    """主函数"""
    models_path = "models"
    
    if not os.path.exists(models_path):
        print(f"❌ 模型目录不存在: {models_path}")
        return
    
    print("🔍 搜索经验数据文件...")
    experience_files = find_experience_files(models_path)
    
    if not experience_files:
        print("❌ 未找到任何经验数据文件")
        return
    
    print(f"✅ 找到 {len(experience_files)} 个经验数据文件:")
    for file_path in experience_files:
        print(f"   - {file_path}")
    
    print("\n" + "=" * 80)
    
    # 验证每个文件
    valid_files = []
    for file_path in experience_files:
        print(f"\n")
        if verify_experience_file(file_path):
            valid_files.append(file_path)
        print("-" * 60)
    
    # 总结
    print(f"\n📊 验证总结:")
    print(f"   - 总文件数: {len(experience_files)}")
    print(f"   - 有效文件数: {len(valid_files)}")
    print(f"   - 无效文件数: {len(experience_files) - len(valid_files)}")
    
    if valid_files:
        print(f"\n✅ 可用的经验数据文件:")
        for file_path in valid_files:
            print(f"   - {file_path}")
    else:
        print(f"\n❌ 没有找到有效的经验数据文件")

if __name__ == "__main__":
    main()






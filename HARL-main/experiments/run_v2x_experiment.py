"""
V2X第一个创新点快速验证脚本

使用方法:
python run_v2x_experiment.py --mode quick   # 快速验证（较少训练步数）
python run_v2x_experiment.py --mode full    # 完整实验
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
from pathlib import Path

# 添加HARL路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from harl.utils.configs_tools import get_defaults_yaml_args
from harl.utils.envs_tools import make_train_env, make_eval_env
from harl.runners.off_policy_ha_runner import OffPolicyHARunner


def modify_config_for_experiment(base_args, experiment_type):
    """修改配置以适应不同实验类型"""
    
    config = base_args.copy()
    
    if experiment_type == "baseline":
        # 标准HASAC配置
        config.update({
            "use_transformer": False,
            "use_contrastive_learning": False,
            "exp_name": "v2x_baseline_hasac"
        })
        
    elif experiment_type == "transformer":
        # HASAC + Transformer配置
        config.update({
            "use_transformer": True,
            "use_contrastive_learning": False,
            "transformer_d_model": 256,
            "transformer_nhead": 8,
            "transformer_num_layers": 4,
            "max_seq_length": 50,
            "exp_name": "v2x_hasac_transformer"
        })
        
    elif experiment_type == "full_innovation":
        # 完整第一个创新点配置
        config.update({
            "use_transformer": True,
            "use_contrastive_learning": True,
            "transformer_d_model": 256,
            "transformer_nhead": 8,
            "transformer_num_layers": 4,
            "max_seq_length": 50,
            "contrastive_temperature": 0.1,
            "similarity_threshold": 0.8,
            "temporal_weight": 0.1,
            "lambda_cl": 0.1,
            "exp_name": "v2x_hasac_full_innovation"
        })
    
    return config


def run_single_training(config, scenario_name="medium_dynamics"):
    """运行单个训练实验"""
    
    print(f"开始训练: {config['exp_name']} - {scenario_name}")
    
    # 场景特定配置
    scenario_configs = {
        "low_dynamics": {
            "vehicle_speed_range": [10.0, 30.0],
            "task_generation_prob": 0.2,
            "num_agents": 8,
            "communication_range": 400.0,
            "max_episode_steps": 150
        },
        "medium_dynamics": {
            "vehicle_speed_range": [20.0, 60.0],
            "task_generation_prob": 0.3,
            "num_agents": 10,
            "communication_range": 300.0,
            "max_episode_steps": 200
        },
        "high_dynamics": {
            "vehicle_speed_range": [40.0, 100.0],
            "task_generation_prob": 0.4,
            "num_agents": 12,
            "communication_range": 200.0,
            "max_episode_steps": 250
        }
    }
    
    # 应用场景配置
    if scenario_name in scenario_configs:
        config.update(scenario_configs[scenario_name])
    
    try:
        # 创建环境
        train_envs = make_train_env(config["env_name"], config["seed"], config["n_training_threads"], config)
        eval_envs = make_eval_env(config["env_name"], config["seed"], config["n_eval_rollout_threads"], config)
        
        # 创建训练器
        runner = OffPolicyHARunner(config, train_envs, eval_envs)
        
        # 开始训练
        runner.run()
        
        # 清理
        train_envs.close()
        eval_envs.close()
        
        print(f"训练完成: {config['exp_name']}")
        return True
        
    except Exception as e:
        print(f"训练失败: {config['exp_name']}, 错误: {str(e)}")
        return False


def run_quick_comparison():
    """运行快速对比实验"""
    
    print("="*60)
    print("开始快速对比实验 - 验证第一个创新点效果")
    print("="*60)
    
    # 基础配置
    config_path = "HARL-main/harl/configs/envs_cfgs/v2x.yaml"
    base_args = get_defaults_yaml_args(config_path)
    
    # 快速验证的配置调整
    base_args.update({
        "num_env_steps": 50000,  # 减少训练步数用于快速验证
        "eval_interval": 5000,
        "save_interval": 25000,
        "log_interval": 1000,
        "n_training_threads": 4,
        "n_eval_rollout_threads": 2
    })
    
    # 实验配置列表
    experiments = [
        ("baseline", "基线HASAC"),
        ("transformer", "HASAC+Transformer"),
        ("full_innovation", "完整创新点")
    ]
    
    results = {}
    
    for exp_type, exp_name in experiments:
        print(f"\n开始运行: {exp_name}")
        
        # 修改配置
        config = modify_config_for_experiment(base_args, exp_type)
        
        # 运行训练
        success = run_single_training(config, "medium_dynamics")
        results[exp_type] = {"success": success, "name": exp_name}
        
        if success:
            print(f"✓ {exp_name} 训练成功")
        else:
            print(f"✗ {exp_name} 训练失败")
    
    # 打印总结
    print("\n" + "="*60)
    print("快速验证实验总结")
    print("="*60)
    
    for exp_type, result in results.items():
        status = "成功" if result["success"] else "失败"
        print(f"{result['name']}: {status}")
    
    print("\n注意: 这是快速验证，完整实验需要更多训练步数")
    
    return results


def run_full_evaluation():
    """运行完整评估实验"""
    
    print("="*60)
    print("开始完整评估实验")
    print("="*60)
    
    # 基础配置
    config_path = "HARL-main/harl/configs/envs_cfgs/v2x.yaml"
    base_args = get_defaults_yaml_args(config_path)
    
    # 完整实验配置
    base_args.update({
        "num_env_steps": 200000,  # 完整训练步数
        "eval_interval": 10000,
        "save_interval": 50000,
        "log_interval": 2000
    })
    
    # 实验配置
    experiments = ["baseline", "transformer", "full_innovation"]
    scenarios = ["low_dynamics", "medium_dynamics", "high_dynamics"]
    
    results = {}
    
    for exp_type in experiments:
        results[exp_type] = {}
        
        for scenario in scenarios:
            print(f"\n运行: {exp_type} - {scenario}")
            
            # 修改配置
            config = modify_config_for_experiment(base_args, exp_type)
            config["exp_name"] = f"{config['exp_name']}_{scenario}"
            
            # 运行训练
            success = run_single_training(config, scenario)
            results[exp_type][scenario] = success
    
    # 保存结果
    with open("full_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n完整实验结果已保存到 full_experiment_results.json")
    
    return results


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="V2X第一个创新点验证实验")
    parser.add_argument("--mode", type=str, choices=["quick", "full"], default="quick",
                       help="实验模式: quick(快速验证) 或 full(完整实验)")
    parser.add_argument("--seed", type=int, default=0,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    if args.mode == "quick":
        results = run_quick_comparison()
    elif args.mode == "full":
        results = run_full_evaluation()
    
    print("\n实验完成!")


if __name__ == "__main__":
    main() 
"""
V2X第一个创新点实验结果分析工具

用于分析训练日志，生成对比图表，验证创新点效果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import glob
from pathlib import Path
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class V2XResultAnalyzer:
    """V2X实验结果分析器"""
    
    def __init__(self, result_dir="results"):
        self.result_dir = result_dir
        self.figures_dir = os.path.join(result_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def load_training_logs(self, log_pattern="*/logs/*.txt"):
        """加载训练日志文件"""
        
        log_files = glob.glob(os.path.join(self.result_dir, log_pattern))
        training_data = {}
        
        for log_file in log_files:
            exp_name = self.extract_experiment_name(log_file)
            data = self.parse_log_file(log_file)
            if data:
                training_data[exp_name] = data
                
        return training_data
    
    def extract_experiment_name(self, log_file):
        """从日志文件路径提取实验名称"""
        
        path_parts = Path(log_file).parts
        for part in path_parts:
            if "v2x_" in part:
                return part
        
        return os.path.basename(os.path.dirname(log_file))
    
    def parse_log_file(self, log_file):
        """解析训练日志文件"""
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            episodes = []
            rewards = []
            task_completion_rates = []
            energy_costs = []
            contrastive_losses = []
            
            for line in lines:
                if "Episode" in line and "Reward" in line:
                    # 解析episode奖励
                    try:
                        parts = line.strip().split()
                        episode = int(parts[1])
                        reward = float(parts[3])
                        episodes.append(episode)
                        rewards.append(reward)
                    except:
                        continue
                        
                elif "Task completion rate" in line:
                    # 解析任务完成率
                    try:
                        rate = float(line.split(":")[-1].strip())
                        task_completion_rates.append(rate)
                    except:
                        continue
                        
                elif "Energy cost" in line:
                    # 解析能耗
                    try:
                        cost = float(line.split(":")[-1].strip())
                        energy_costs.append(cost)
                    except:
                        continue
                        
                elif "Contrastive loss" in line:
                    # 解析对比学习损失
                    try:
                        loss = float(line.split(":")[-1].strip())
                        contrastive_losses.append(loss)
                    except:
                        continue
            
            return {
                "episodes": episodes,
                "rewards": rewards,
                "task_completion_rates": task_completion_rates,
                "energy_costs": energy_costs,
                "contrastive_losses": contrastive_losses
            }
            
        except Exception as e:
            print(f"解析日志文件失败 {log_file}: {str(e)}")
            return None
    
    def analyze_convergence(self, training_data):
        """分析收敛性"""
        
        convergence_analysis = {}
        
        for exp_name, data in training_data.items():
            if not data["rewards"]:
                continue
                
            rewards = np.array(data["rewards"])
            
            # 计算收敛步数（奖励稳定的点）
            window_size = 20
            convergence_step = len(rewards)
            
            if len(rewards) > window_size:
                for i in range(window_size, len(rewards)):
                    recent_rewards = rewards[i-window_size:i]
                    if np.std(recent_rewards) < 0.1:  # 奖励方差小于阈值
                        convergence_step = i
                        break
            
            # 计算最终性能（最后20%数据的平均值）
            final_performance = np.mean(rewards[-len(rewards)//5:]) if len(rewards) > 20 else np.mean(rewards)
            
            convergence_analysis[exp_name] = {
                "convergence_step": convergence_step,
                "final_performance": final_performance,
                "stability": np.std(rewards[-len(rewards)//5:]) if len(rewards) > 20 else np.std(rewards)
            }
        
        return convergence_analysis
    
    def plot_learning_curves(self, training_data):
        """绘制学习曲线对比图"""
        
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (exp_name, data) in enumerate(training_data.items()):
            if data["rewards"]:
                # 平滑处理
                rewards = np.array(data["rewards"])
                episodes = np.array(data["episodes"]) if data["episodes"] else np.arange(len(rewards))
                
                # 移动平均平滑
                window_size = min(20, len(rewards) // 10)
                if window_size > 1:
                    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='same')
                else:
                    smoothed_rewards = rewards
                
                plt.plot(episodes, smoothed_rewards, 
                        label=self.format_experiment_name(exp_name),
                        color=colors[i % len(colors)], linewidth=2)
        
        plt.xlabel('训练Episode')
        plt.ylabel('累积奖励')
        plt.title('不同算法的学习曲线对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.figures_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("学习曲线图已保存到", os.path.join(self.figures_dir, 'learning_curves.png'))
    
    def plot_performance_comparison(self, convergence_analysis):
        """绘制性能对比图"""
        
        if not convergence_analysis:
            print("没有收敛性分析数据")
            return
        
        exp_names = list(convergence_analysis.keys())
        final_performances = [convergence_analysis[name]["final_performance"] for name in exp_names]
        convergence_steps = [convergence_analysis[name]["convergence_step"] for name in exp_names]
        
        # 性能对比柱状图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 最终性能对比
        formatted_names = [self.format_experiment_name(name) for name in exp_names]
        ax1.bar(formatted_names, final_performances, color=['lightblue', 'lightcoral', 'lightgreen'][:len(exp_names)])
        ax1.set_ylabel('最终性能 (累积奖励)')
        ax1.set_title('不同算法的最终性能对比')
        ax1.tick_params(axis='x', rotation=45)
        
        # 收敛速度对比
        ax2.bar(formatted_names, convergence_steps, color=['lightblue', 'lightcoral', 'lightgreen'][:len(exp_names)])
        ax2.set_ylabel('收敛步数')
        ax2.set_title('不同算法的收敛速度对比')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("性能对比图已保存到", os.path.join(self.figures_dir, 'performance_comparison.png'))
    
    def plot_innovation_analysis(self, training_data):
        """绘制创新点分析图"""
        
        # 提取包含对比学习损失的实验
        cl_experiments = {name: data for name, data in training_data.items() 
                         if data["contrastive_losses"]}
        
        if not cl_experiments:
            print("没有找到对比学习损失数据")
            return
        
        # 绘制对比学习损失变化
        plt.figure(figsize=(12, 6))
        
        for exp_name, data in cl_experiments.items():
            losses = np.array(data["contrastive_losses"])
            steps = np.arange(len(losses))
            
            plt.plot(steps, losses, label=self.format_experiment_name(exp_name), linewidth=2)
        
        plt.xlabel('训练步数')
        plt.ylabel('对比学习损失')
        plt.title('对比学习损失变化 - 验证第一个创新点效果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.figures_dir, 'contrastive_learning_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("对比学习损失图已保存到", os.path.join(self.figures_dir, 'contrastive_learning_loss.png'))
    
    def format_experiment_name(self, exp_name):
        """格式化实验名称为中文"""
        
        name_mapping = {
            "v2x_baseline_hasac": "基线HASAC",
            "v2x_hasac_transformer": "HASAC+Transformer",
            "v2x_hasac_full_innovation": "完整创新点",
            "baseline": "基线HASAC",
            "transformer": "HASAC+Transformer", 
            "full_innovation": "完整创新点"
        }
        
        for key, value in name_mapping.items():
            if key in exp_name:
                return value
        
        return exp_name
    
    def generate_summary_report(self, training_data, convergence_analysis):
        """生成摘要报告"""
        
        report = []
        report.append("="*60)
        report.append("V2X第一个创新点实验结果摘要")
        report.append("="*60)
        report.append("")
        
        # 性能对比
        if convergence_analysis:
            report.append("1. 性能对比:")
            report.append("")
            
            sorted_exps = sorted(convergence_analysis.items(), 
                               key=lambda x: x[1]["final_performance"], reverse=True)
            
            for i, (exp_name, analysis) in enumerate(sorted_exps):
                report.append(f"   {i+1}. {self.format_experiment_name(exp_name)}")
                report.append(f"      最终性能: {analysis['final_performance']:.4f}")
                report.append(f"      收敛步数: {analysis['convergence_step']}")
                report.append(f"      稳定性: {analysis['stability']:.4f}")
                report.append("")
        
        # 创新点效果分析
        baseline_performance = None
        innovation_performance = None
        
        for exp_name, analysis in convergence_analysis.items():
            if "baseline" in exp_name.lower():
                baseline_performance = analysis["final_performance"]
            elif "full_innovation" in exp_name.lower():
                innovation_performance = analysis["final_performance"]
        
        if baseline_performance and innovation_performance:
            improvement = (innovation_performance - baseline_performance) / baseline_performance * 100
            report.append("2. 创新点效果:")
            report.append("")
            report.append(f"   基线性能: {baseline_performance:.4f}")
            report.append(f"   创新点性能: {innovation_performance:.4f}")
            report.append(f"   性能提升: {improvement:.2f}%")
            report.append("")
            
            if improvement > 0:
                report.append("   ✓ 第一个创新点验证成功！")
            else:
                report.append("   ✗ 第一个创新点需要进一步调优")
        
        report.append("")
        report.append("3. 建议:")
        report.append("")
        
        if innovation_performance and baseline_performance:
            if improvement > 10:
                report.append("   - 创新点效果显著，建议进行更大规模实验")
            elif improvement > 0:
                report.append("   - 创新点有一定效果，建议调优超参数")
            else:
                report.append("   - 需要检查实现或调整网络结构")
        
        report.append("   - 可以尝试更长的训练时间")
        report.append("   - 可以测试不同的场景配置")
        
        # 保存报告
        report_text = "\n".join(report)
        
        with open(os.path.join(self.result_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n摘要报告已保存到 {os.path.join(self.result_dir, 'summary_report.txt')}")
    
    def run_analysis(self):
        """运行完整分析"""
        
        print("开始分析实验结果...")
        
        # 加载训练数据
        training_data = self.load_training_logs()
        
        if not training_data:
            print("没有找到训练日志文件")
            return
        
        print(f"找到 {len(training_data)} 个实验的数据")
        
        # 分析收敛性
        convergence_analysis = self.analyze_convergence(training_data)
        
        # 生成图表
        self.plot_learning_curves(training_data)
        self.plot_performance_comparison(convergence_analysis)
        self.plot_innovation_analysis(training_data)
        
        # 生成摘要报告
        self.generate_summary_report(training_data, convergence_analysis)
        
        print("\n分析完成！")


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="V2X实验结果分析工具")
    parser.add_argument("--result_dir", type=str, default="results",
                       help="结果目录路径")
    
    args = parser.parse_args()
    
    analyzer = V2XResultAnalyzer(args.result_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 
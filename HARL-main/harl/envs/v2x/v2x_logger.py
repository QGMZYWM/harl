"""Logger for V2X task offloading environment."""
import numpy as np
from harl.common.base_logger import BaseLogger


class V2XLogger(BaseLogger):
    """V2X Environment Logger."""
    
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """Initialize V2X logger."""
        super(V2XLogger, self).__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        
        # V2X specific metrics
        self.task_completion_rates = []
        self.task_failure_rates = []
        self.energy_consumptions = []
        self.load_balances = []
        self.communication_qualities = []
        
    def get_task_name(self):
        """Get task name for V2X environment."""
        return "v2x_task_offloading"
    
    def eval_log(self, eval_episode_rewards, num_eval_episodes):
        """Log evaluation results."""
        super().eval_log(eval_episode_rewards, num_eval_episodes)
        
        # Calculate V2X specific metrics
        if len(self.task_completion_rates) > 0:
            avg_completion_rate = np.mean(self.task_completion_rates)
            avg_failure_rate = np.mean(self.task_failure_rates)
            avg_energy_consumption = np.mean(self.energy_consumptions)
            
            print(f"Average Task Completion Rate: {avg_completion_rate:.4f}")
            print(f"Average Task Failure Rate: {avg_failure_rate:.4f}")
            print(f"Average Energy Consumption: {avg_energy_consumption:.4f}")
            
            if self.writter is not None:
                self.writter.add_scalars(
                    "eval_task_completion_rate",
                    {"task_completion_rate": avg_completion_rate},
                    self.total_num_steps
                )
                self.writter.add_scalars(
                    "eval_task_failure_rate", 
                    {"task_failure_rate": avg_failure_rate},
                    self.total_num_steps
                )
                self.writter.add_scalars(
                    "eval_energy_consumption",
                    {"energy_consumption": avg_energy_consumption},
                    self.total_num_steps
                )
    
    def episode_log(self, infos, episode_rewards, episode_length):
        """Log episode information."""
        super().episode_log(infos, episode_rewards, episode_length)
        
        # Extract V2X specific metrics from infos
        if infos is not None and len(infos) > 0:
            episode_completion_rate = []
            episode_failure_rate = []
            episode_energy_consumption = []
            
            for agent_info in infos:
                if isinstance(agent_info, dict):
                    completed = agent_info.get('completed_tasks', 0)
                    failed = agent_info.get('failed_tasks', 0)
                    total_tasks = completed + failed
                    
                    if total_tasks > 0:
                        completion_rate = completed / total_tasks
                        failure_rate = failed / total_tasks
                    else:
                        completion_rate = 0.0
                        failure_rate = 0.0
                    
                    episode_completion_rate.append(completion_rate)
                    episode_failure_rate.append(failure_rate)
                    episode_energy_consumption.append(agent_info.get('energy_consumed', 0.0))
            
            if len(episode_completion_rate) > 0:
                self.task_completion_rates.append(np.mean(episode_completion_rate))
                self.task_failure_rates.append(np.mean(episode_failure_rate))
                self.energy_consumptions.append(np.mean(episode_energy_consumption))
    
    def train_log(self, train_infos):
        """Log training information."""
        super().train_log(train_infos)
        
        # Add V2X specific training logs if needed
        if "contrastive_loss" in train_infos:
            contrastive_loss = train_infos["contrastive_loss"]
            print(f"Contrastive Loss: {contrastive_loss:.6f}")
            
            if self.writter is not None:
                self.writter.add_scalars(
                    "train_contrastive_loss",
                    {"contrastive_loss": contrastive_loss},
                    self.total_num_steps
                )
    
    def log_clear(self):
        """Clear logging data."""
        super().log_clear()
        
        # Clear V2X specific metrics
        self.task_completion_rates.clear()
        self.task_failure_rates.clear()
        self.energy_consumptions.clear()
        self.load_balances.clear()
        self.communication_qualities.clear() 
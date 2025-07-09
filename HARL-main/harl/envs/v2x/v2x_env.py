import numpy as np
import gym
from gym import spaces
import torch
import random
import math
from typing import Dict, List, Tuple, Optional


class V2XTaskOffloadingEnv(gym.Env):
    """
    V2X车联网任务卸载环境
    
    实现车联网环境下的多智能体任务卸载决策问题
    支持车辆移动、RSU、任务生成、通信质量变化等
    """
    
    def __init__(self, args):
        super(V2XTaskOffloadingEnv, self).__init__()
        
        # 环境基本参数
        self.args = args
        self.num_agents = args.get("num_agents", 10)
        self.num_rsus = args.get("num_rsus", 4)
        self.map_size = args.get("map_size", 1000.0)  # 地图大小（米）
        self.max_episode_steps = args.get("max_episode_steps", 200)
        
        # 车辆参数
        self.vehicle_speed_range = args.get("vehicle_speed_range", (20.0, 80.0))  # km/h
        self.vehicle_compute_range = args.get("vehicle_compute_range", (1.0, 10.0))  # GHz
        self.vehicle_battery_range = args.get("vehicle_battery_range", (0.3, 1.0))  # 电池比例
        
        # 任务参数
        self.task_generation_prob = args.get("task_generation_prob", 0.3)
        self.task_compute_range = args.get("task_compute_range", (0.5, 5.0))  # 计算需求 GHz*s
        self.task_deadline_range = args.get("task_deadline_range", (5, 20))  # 任务截止时间（步数）
        self.task_data_size_range = args.get("task_data_size_range", (1.0, 50.0))  # MB
        
        # 通信参数
        self.communication_range = args.get("communication_range", 300.0)  # 通信范围（米）
        self.rsu_coverage = args.get("rsu_coverage", 500.0)  # RSU覆盖范围（米）
        self.bandwidth = args.get("bandwidth", 20.0)  # MHz
        self.noise_power = args.get("noise_power", -110.0)  # dBm
        
        # 奖励参数
        self.reward_task_completion = args.get("reward_task_completion", 10.0)
        self.reward_task_failure = args.get("reward_task_failure", -5.0)
        self.reward_energy_efficiency = args.get("reward_energy_efficiency", 1.0)
        self.reward_load_balance = args.get("reward_load_balance", 2.0)
        
        # 动作空间：0-本地处理，1-卸载到RSU1，2-卸载到RSU2，...，N-卸载到邻近车辆
        self.max_offload_targets = self.num_rsus + 5  # RSU + 最多5个邻近车辆
        self.action_space = spaces.Discrete(self.max_offload_targets + 1)
        
        # 观测空间维度计算
        # 自身状态：位置(2) + 速度(2) + 计算能力(1) + 电池(1) + 当前任务信息(3) = 9
        # RSU信息：每个RSU的位置(2) + 负载(1) + 连接质量(1) = 4 * num_rsus
        # 邻近车辆：最多5个邻近车辆，每个车辆位置(2) + 计算能力(1) + 负载(1) + 连接质量(1) = 5 * 5 = 25
        # 历史信息：过去几步的负载和连接质量 = 6
        obs_dim = 9 + 4 * self.num_rsus + 25 + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # 初始化环境状态
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        
        # 初始化车辆状态
        self.vehicles = {}
        for i in range(self.num_agents):
            self.vehicles[i] = {
                'position': np.random.uniform(0, self.map_size, 2),
                'velocity': np.random.uniform(*self.vehicle_speed_range) / 3.6,  # 转换为m/s
                'direction': np.random.uniform(0, 2 * np.pi),
                'compute_capacity': np.random.uniform(*self.vehicle_compute_range),
                'battery_level': np.random.uniform(*self.vehicle_battery_range),
                'current_load': 0.0,
                'tasks': [],
                'completed_tasks': 0,
                'failed_tasks': 0,
                'energy_consumed': 0.0
            }
        
        # 初始化RSU状态
        self.rsus = {}
        for i in range(self.num_rsus):
            # RSU均匀分布在地图上
            angle = 2 * np.pi * i / self.num_rsus
            center_x, center_y = self.map_size / 2, self.map_size / 2
            radius = self.map_size / 3
            self.rsus[i] = {
                'position': np.array([
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle)
                ]),
                'compute_capacity': 50.0,  # RSU计算能力更强
                'current_load': 0.0,
                'served_tasks': 0
            }
        
        # 历史信息记录
        self.history = {i: {'load': [], 'communication': []} for i in range(self.num_agents)}
        
        # 生成初始观测
        observations = {}
        for i in range(self.num_agents):
            observations[i] = self._get_observation(i)
        
        return observations
    
    def step(self, actions):
        """环境步进"""
        self.current_step += 1
        
        # 更新车辆位置
        self._update_vehicle_positions()
        
        # 生成新任务
        self._generate_tasks()
        
        # 执行动作（任务卸载决策）
        rewards = {}
        for agent_id, action in actions.items():
            rewards[agent_id] = self._execute_action(agent_id, action)
        
        # 更新任务处理
        self._update_task_processing()
        
        # 检查任务完成和超时
        self._check_task_completion()
        
        # 生成新观测
        observations = {}
        for i in range(self.num_agents):
            observations[i] = self._get_observation(i)
        
        # 检查episode结束条件
        done = self.current_step >= self.max_episode_steps
        dones = {i: done for i in range(self.num_agents)}
        dones['__all__'] = done
        
        # 生成信息字典
        infos = {}
        for i in range(self.num_agents):
            infos[i] = {
                'completed_tasks': self.vehicles[i]['completed_tasks'],
                'failed_tasks': self.vehicles[i]['failed_tasks'],
                'energy_consumed': self.vehicles[i]['energy_consumed'],
                'current_load': self.vehicles[i]['current_load']
            }
        
        return observations, rewards, dones, infos
    
    def _get_observation(self, agent_id):
        """获取智能体观测"""
        vehicle = self.vehicles[agent_id]
        obs = []
        
        # 自身状态
        obs.extend(vehicle['position'] / self.map_size)  # 归一化位置
        obs.extend([
            np.cos(vehicle['direction']), np.sin(vehicle['direction']),  # 方向向量
            vehicle['compute_capacity'] / 10.0,  # 归一化计算能力
            vehicle['battery_level'],  # 电池水平
        ])
        
        # 当前任务信息
        if vehicle['tasks']:
            current_task = vehicle['tasks'][0]  # 最紧急的任务
            obs.extend([
                current_task['compute_requirement'] / 5.0,  # 归一化计算需求
                current_task['deadline'] / 20.0,  # 归一化截止时间
                current_task['data_size'] / 50.0  # 归一化数据大小
            ])
        else:
            obs.extend([0.0, 0.0, 0.0])
        
        # RSU信息
        for rsu_id, rsu in self.rsus.items():
            distance = np.linalg.norm(vehicle['position'] - rsu['position'])
            communication_quality = self._calculate_communication_quality(distance)
            
            obs.extend([
                (rsu['position'][0] - vehicle['position'][0]) / self.map_size,  # 相对位置x
                (rsu['position'][1] - vehicle['position'][1]) / self.map_size,  # 相对位置y
                rsu['current_load'] / rsu['compute_capacity'],  # 负载比率
                communication_quality  # 通信质量
            ])
        
        # 邻近车辆信息（最多5个最近的）
        nearby_vehicles = self._get_nearby_vehicles(agent_id, max_count=5)
        for i in range(5):
            if i < len(nearby_vehicles):
                other_id, other_vehicle, distance = nearby_vehicles[i]
                communication_quality = self._calculate_communication_quality(distance)
                obs.extend([
                    (other_vehicle['position'][0] - vehicle['position'][0]) / self.map_size,
                    (other_vehicle['position'][1] - vehicle['position'][1]) / self.map_size,
                    other_vehicle['compute_capacity'] / 10.0,
                    other_vehicle['current_load'] / other_vehicle['compute_capacity'],
                    communication_quality
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 历史信息
        history = self.history[agent_id]
        # 过去3步的平均负载
        recent_loads = history['load'][-3:] if len(history['load']) >= 3 else [0.0] * 3
        obs.extend([sum(recent_loads) / len(recent_loads), len(recent_loads) / 3.0])
        
        # 过去3步的平均通信质量
        recent_comm = history['communication'][-3:] if len(history['communication']) >= 3 else [0.0] * 3
        obs.extend([sum(recent_comm) / len(recent_comm), len(recent_comm) / 3.0])
        
        # 环境动态指标
        obs.extend([
            self.current_step / self.max_episode_steps,  # 时间进度
            len(vehicle['tasks']) / 10.0  # 任务队列长度（归一化）
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _execute_action(self, agent_id, action):
        """执行卸载动作"""
        vehicle = self.vehicles[agent_id]
        reward = 0.0
        
        if not vehicle['tasks']:
            return reward  # 没有任务就不执行动作
        
        task = vehicle['tasks'][0]  # 处理最紧急的任务
        
        if action == 0:
            # 本地处理
            if vehicle['current_load'] + task['compute_requirement'] <= vehicle['compute_capacity']:
                task['processing_location'] = 'local'
                task['processing_agent'] = agent_id
                vehicle['current_load'] += task['compute_requirement']
                reward += 2.0  # 本地处理的小奖励
            else:
                # 本地计算能力不足
                reward -= 1.0
        
        elif 1 <= action <= self.num_rsus:
            # 卸载到RSU
            rsu_id = action - 1
            rsu = self.rsus[rsu_id]
            distance = np.linalg.norm(vehicle['position'] - rsu['position'])
            
            if distance <= self.rsu_coverage:
                communication_quality = self._calculate_communication_quality(distance)
                transmission_time = task['data_size'] / (self.bandwidth * communication_quality)
                
                if transmission_time + task['compute_requirement'] / rsu['compute_capacity'] <= task['deadline']:
                    task['processing_location'] = 'rsu'
                    task['processing_agent'] = rsu_id
                    task['transmission_time'] = transmission_time
                    rsu['current_load'] += task['compute_requirement']
                    reward += 5.0 * communication_quality  # RSU处理的奖励与通信质量相关
                else:
                    reward -= 2.0  # 无法在截止时间前完成
            else:
                reward -= 3.0  # 超出RSU覆盖范围
        
        else:
            # 卸载到邻近车辆
            nearby_vehicles = self._get_nearby_vehicles(agent_id, max_count=5)
            target_index = action - self.num_rsus - 1
            
            if target_index < len(nearby_vehicles):
                target_id, target_vehicle, distance = nearby_vehicles[target_index]
                
                if distance <= self.communication_range:
                    communication_quality = self._calculate_communication_quality(distance)
                    transmission_time = task['data_size'] / (self.bandwidth * communication_quality * 0.5)  # V2V带宽较低
                    
                    if (target_vehicle['current_load'] + task['compute_requirement'] <= target_vehicle['compute_capacity'] 
                        and transmission_time + task['compute_requirement'] / target_vehicle['compute_capacity'] <= task['deadline']):
                        task['processing_location'] = 'vehicle'
                        task['processing_agent'] = target_id
                        task['transmission_time'] = transmission_time
                        target_vehicle['current_load'] += task['compute_requirement']
                        reward += 3.0 * communication_quality  # 车辆协作奖励
                    else:
                        reward -= 1.5  # 目标车辆能力不足或时间不够
                else:
                    reward -= 2.0  # 超出通信范围
            else:
                reward -= 1.0  # 无效的目标车辆
        
        return reward
    
    def _update_vehicle_positions(self):
        """更新车辆位置"""
        for vehicle in self.vehicles.values():
            # 简单的移动模型：直线行驶，边界处转向
            dx = vehicle['velocity'] * np.cos(vehicle['direction'])
            dy = vehicle['velocity'] * np.sin(vehicle['direction'])
            
            new_x = vehicle['position'][0] + dx
            new_y = vehicle['position'][1] + dy
            
            # 边界检查和转向
            if new_x < 0 or new_x > self.map_size:
                vehicle['direction'] = np.pi - vehicle['direction']
                new_x = np.clip(new_x, 0, self.map_size)
            
            if new_y < 0 or new_y > self.map_size:
                vehicle['direction'] = -vehicle['direction']
                new_y = np.clip(new_y, 0, self.map_size)
            
            vehicle['position'] = np.array([new_x, new_y])
    
    def _generate_tasks(self):
        """生成新任务"""
        for agent_id, vehicle in self.vehicles.items():
            if random.random() < self.task_generation_prob and len(vehicle['tasks']) < 5:
                task = {
                    'id': f"task_{agent_id}_{self.current_step}_{len(vehicle['tasks'])}",
                    'compute_requirement': np.random.uniform(*self.task_compute_range),
                    'deadline': np.random.randint(*self.task_deadline_range),
                    'data_size': np.random.uniform(*self.task_data_size_range),
                    'creation_time': self.current_step,
                    'processing_location': None,
                    'processing_agent': None,
                    'transmission_time': 0.0
                }
                vehicle['tasks'].append(task)
                # 按截止时间排序
                vehicle['tasks'].sort(key=lambda x: x['deadline'])
    
    def _update_task_processing(self):
        """更新任务处理进度"""
        # 更新车辆任务处理
        for vehicle in self.vehicles.values():
            if vehicle['tasks']:
                task = vehicle['tasks'][0]
                if task['processing_location'] == 'local':
                    task['deadline'] -= 1
                    if task['deadline'] <= 0:
                        # 任务完成
                        vehicle['tasks'].pop(0)
                        vehicle['current_load'] -= task['compute_requirement']
                        vehicle['completed_tasks'] += 1
                        vehicle['energy_consumed'] += task['compute_requirement'] * 0.1
        
        # 更新RSU任务处理
        for rsu in self.rsus.values():
            if rsu['current_load'] > 0:
                # 简化的处理模型：每步处理固定量的计算
                processing_rate = rsu['compute_capacity'] * 0.1
                rsu['current_load'] = max(0, rsu['current_load'] - processing_rate)
    
    def _check_task_completion(self):
        """检查任务完成和超时"""
        for vehicle in self.vehicles.values():
            tasks_to_remove = []
            for i, task in enumerate(vehicle['tasks']):
                if task['deadline'] <= 0:
                    if task['processing_location'] is None:
                        # 任务超时失败
                        vehicle['failed_tasks'] += 1
                        tasks_to_remove.append(i)
            
            # 移除超时任务
            for i in reversed(tasks_to_remove):
                vehicle['tasks'].pop(i)
    
    def _get_nearby_vehicles(self, agent_id, max_count=5):
        """获取邻近车辆"""
        vehicle = self.vehicles[agent_id]
        nearby = []
        
        for other_id, other_vehicle in self.vehicles.items():
            if other_id != agent_id:
                distance = np.linalg.norm(vehicle['position'] - other_vehicle['position'])
                if distance <= self.communication_range:
                    nearby.append((other_id, other_vehicle, distance))
        
        # 按距离排序，返回最近的几个
        nearby.sort(key=lambda x: x[2])
        return nearby[:max_count]
    
    def _calculate_communication_quality(self, distance):
        """计算通信质量"""
        if distance == 0:
            return 1.0
        
        # 简化的路径损耗模型
        path_loss = 32.45 + 20 * np.log10(distance / 1000)  # 自由空间路径损耗
        signal_power = 20 - path_loss  # 假设发射功率20dBm
        snr = signal_power - self.noise_power
        
        # 转换为连接质量（0-1之间）
        quality = 1.0 / (1.0 + np.exp(-(snr - 10) / 5))
        return max(0.1, min(1.0, quality))
    
    def get_env_info(self):
        """获取环境信息"""
        return {
            "state_shape": self.observation_space.shape[0],
            "obs_shape": self.observation_space.shape[0], 
            "n_actions": self.action_space.n,
            "n_agents": self.num_agents,
            "episode_limit": self.max_episode_steps
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数，用于优化V2X环境中的状态表征
    
    该损失函数将相似时空上下文的嵌入向量拉近，将不相似的推远，
    从而提升Transformer编码器生成的状态表征的判别性和质量。
    """
    
    def __init__(self, temperature=0.1, similarity_threshold=0.8):
        """
        Args:
            temperature: InfoNCE损失中的温度参数，控制分布的尖锐程度
            similarity_threshold: 判断两个状态是否相似的阈值
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        
    def forward(self, embeddings, states_info, reduce=True):
        """
        计算对比学习损失
        
        Args:
            embeddings: 状态嵌入 [batch_size, embedding_dim]
            states_info: 状态信息字典，包含用于判断相似性的特征
            reduce: 是否对batch进行平均
            
        Returns:
            loss: 对比学习损失
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # L2归一化嵌入向量
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # 创建正样本mask（相似的状态对）
        positive_mask = self._create_positive_mask(states_info, batch_size, embeddings.device)
        
        # 创建负样本mask（排除自身）
        negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        
        # 计算InfoNCE损失
        loss = self._info_nce_loss(similarity_matrix, positive_mask, negative_mask)
        
        if reduce:
            loss = loss.mean()
            
        return loss
    
    def _create_positive_mask(self, states_info, batch_size, device):
        """
        根据状态信息创建正样本mask
        
        Args:
            states_info: 包含状态特征的字典
            batch_size: batch大小
            device: 计算设备
            
        Returns:
            positive_mask: 正样本mask [batch_size, batch_size]
        """
        positive_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
        
        # 如果没有提供状态信息，使用基于嵌入相似度的简单策略
        if states_info is None:
            return positive_mask
        
        # 根据不同的状态特征判断相似性
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if self._are_states_similar(states_info, i, j):
                    positive_mask[i, j] = True
                    positive_mask[j, i] = True
        
        return positive_mask
    
    def _are_states_similar(self, states_info, i, j):
        """
        判断两个状态是否相似
        
        在V2X环境中，相似的状态可能包括：
        - 相似的车辆速度和位置
        - 相似的网络连接质量
        - 相似的计算负载情况
        """
        # 速度相似性
        if 'velocity' in states_info:
            vel_i = states_info['velocity'][i]
            vel_j = states_info['velocity'][j]
            vel_similarity = 1.0 - abs(vel_i - vel_j) / max(abs(vel_i) + abs(vel_j) + 1e-8, 1.0)
            if vel_similarity < self.similarity_threshold:
                return False
        
        # 位置相似性（基于距离）
        if 'position' in states_info:
            pos_i = states_info['position'][i]
            pos_j = states_info['position'][j]
            distance = torch.norm(pos_i - pos_j)
            # 距离小于阈值时认为位置相似
            if distance > 50.0:  # 50米阈值，可配置
                return False
        
        # 网络质量相似性
        if 'network_quality' in states_info:
            net_i = states_info['network_quality'][i]
            net_j = states_info['network_quality'][j]
            net_similarity = 1.0 - abs(net_i - net_j)
            if net_similarity < self.similarity_threshold:
                return False
        
        # 计算负载相似性
        if 'cpu_usage' in states_info:
            cpu_i = states_info['cpu_usage'][i]
            cpu_j = states_info['cpu_usage'][j]
            cpu_similarity = 1.0 - abs(cpu_i - cpu_j)
            if cpu_similarity < self.similarity_threshold:
                return False
        
        return True
    
    def _info_nce_loss(self, similarity_matrix, positive_mask, negative_mask):
        """
        计算InfoNCE损失
        
        Args:
            similarity_matrix: 相似度矩阵 [batch_size, batch_size]
            positive_mask: 正样本mask
            negative_mask: 负样本mask
            
        Returns:
            loss: InfoNCE损失
        """
        batch_size = similarity_matrix.size(0)
        
        # 应用温度参数
        logits = similarity_matrix / self.temperature
        
        # 计算每个样本的损失
        losses = []
        for i in range(batch_size):
            # 获取正样本
            positive_logits = logits[i][positive_mask[i]]
            
            if positive_logits.numel() == 0:
                # 如果没有正样本，跳过这个样本
                continue
            
            # 获取负样本（排除自身）
            negative_logits = logits[i][negative_mask[i]]
            
            # 计算InfoNCE损失
            # log(exp(positive) / (exp(positive) + sum(exp(negatives))))
            positive_exp = torch.exp(positive_logits)
            negative_exp = torch.exp(negative_logits)
            
            denominator = positive_exp.sum() + negative_exp.sum()
            loss = -torch.log(positive_exp.sum() / denominator)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=similarity_matrix.device)
        
        return torch.stack(losses)


class V2XStatesSimilarity:
    """
    V2X环境中状态相似性的计算工具类
    """
    
    @staticmethod
    def extract_state_features(observations, actions=None):
        """
        从观测中提取用于相似性计算的特征
        
        Args:
            observations: 观测数据 [batch_size, obs_dim]
            actions: 动作数据 [batch_size, action_dim] (可选)
            
        Returns:
            states_info: 包含特征的字典
        """
        batch_size = observations.size(0)
        states_info = {}
        
        # 假设观测向量的前几个维度包含关键信息
        # 这需要根据实际的V2X环境观测结构进行调整
        
        if observations.size(1) >= 2:
            # 假设前2维是位置信息
            states_info['position'] = observations[:, :2]
        
        if observations.size(1) >= 3:
            # 假设第3维是速度信息
            states_info['velocity'] = observations[:, 2]
        
        if observations.size(1) >= 4:
            # 假设第4维是网络质量
            states_info['network_quality'] = observations[:, 3]
        
        if observations.size(1) >= 5:
            # 假设第5维是CPU使用率
            states_info['cpu_usage'] = observations[:, 4]
        
        return states_info
    
    @staticmethod
    def compute_temporal_similarity(embeddings_t, embeddings_t_minus_1, alpha=0.9):
        """
        计算时间上连续状态的相似性损失
        
        Args:
            embeddings_t: 当前时刻的嵌入 [batch_size, embedding_dim]
            embeddings_t_minus_1: 前一时刻的嵌入 [batch_size, embedding_dim]
            alpha: 时间连续性权重
            
        Returns:
            temporal_loss: 时间连续性损失
        """
        if embeddings_t_minus_1 is None:
            return torch.tensor(0.0, device=embeddings_t.device)
        
        # L2归一化
        emb_t = F.normalize(embeddings_t, p=2, dim=1)
        emb_t_1 = F.normalize(embeddings_t_minus_1, p=2, dim=1)
        
        # 计算相似度
        similarity = torch.sum(emb_t * emb_t_1, dim=1)
        
        # 时间连续性损失：相邻时刻的嵌入应该相似
        temporal_loss = alpha * (1.0 - similarity).mean()
        
        return temporal_loss


class EnhancedContrastiveLoss(nn.Module):
    """
    增强的对比学习损失，结合空间和时间对比学习
    """
    
    def __init__(self, temperature=0.1, similarity_threshold=0.8, temporal_weight=0.1):
        super(EnhancedContrastiveLoss, self).__init__()
        self.spatial_contrastive = ContrastiveLoss(temperature, similarity_threshold)
        self.temporal_weight = temporal_weight
        
    def forward(self, current_embeddings, states_info, previous_embeddings=None):
        """
        计算增强的对比学习损失
        
        Args:
            current_embeddings: 当前时刻的嵌入
            states_info: 状态信息
            previous_embeddings: 前一时刻的嵌入（可选）
            
        Returns:
            total_loss: 总对比学习损失
        """
        # 空间对比学习损失
        spatial_loss = self.spatial_contrastive(current_embeddings, states_info)
        
        # 时间对比学习损失
        temporal_loss = V2XStatesSimilarity.compute_temporal_similarity(
            current_embeddings, previous_embeddings
        )
        
        # 组合损失
        total_loss = spatial_loss + self.temporal_weight * temporal_loss
        
        return total_loss, spatial_loss, temporal_loss 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码模块，为Transformer提供序列位置信息"""
    
    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """
    用于V2X动态上下文感知状态表征的Transformer编码器
    
    这个模块处理智能体的历史观测-动作序列，生成丰富的上下文感知状态嵌入
    """
    
    def __init__(self, args, obs_dim, action_dim, device=torch.device("cpu")):
        """
        Args:
            args: 配置参数字典
            obs_dim: 观测维度
            action_dim: 动作维度  
            device: 计算设备
        """
        super(TransformerEncoder, self).__init__()
        
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 从args中获取参数，设置默认值
        self.d_model = args.get("transformer_d_model", 256)
        self.nhead = args.get("transformer_nhead", 8)
        self.num_layers = args.get("transformer_num_layers", 4)
        self.dim_feedforward = args.get("transformer_dim_feedforward", 512)
        self.dropout = args.get("transformer_dropout", 0.1)
        self.max_seq_length = args.get("max_seq_length", 50)
        
        # 输入投影层：将观测-动作拼接向量投影到d_model维度
        input_dim = obs_dim + action_dim
        self.input_projection = nn.Linear(input_dim, self.d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='relu',
            batch_first=False  # [seq_len, batch, feature]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # 输出投影层：将最终隐藏状态投影到期望的输出维度
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.to(device)
    
    def forward(self, obs_seq, action_seq, seq_lengths=None):
        """
        前向传播
        
        Args:
            obs_seq: 观测序列 [batch_size, seq_len, obs_dim]
            action_seq: 动作序列 [batch_size, seq_len, action_dim] 
            seq_lengths: 每个序列的实际长度 [batch_size] (可选)
            
        Returns:
            context_embedding: 上下文感知状态嵌入 [batch_size, d_model]
            sequence_embeddings: 序列中每个时间步的嵌入 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = obs_seq.shape
        
        # 拼接观测和动作
        obs_action = torch.cat([obs_seq, action_seq], dim=-1)  # [batch, seq_len, obs_dim + action_dim]
        
        # 输入投影
        embedded = self.input_projection(obs_action)  # [batch, seq_len, d_model]
        
        # 转换为Transformer期望的格式: [seq_len, batch, d_model]
        embedded = embedded.transpose(0, 1)
        
        # 添加位置编码
        embedded = self.pos_encoder(embedded)
        
        # 创建padding mask（如果提供了序列长度）
        src_key_padding_mask = None
        if seq_lengths is not None:
            src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
            for i, length in enumerate(seq_lengths):
                if length < seq_len:
                    src_key_padding_mask[i, length:] = True
        
        # Transformer编码
        transformer_output = self.transformer_encoder(
            embedded, 
            src_key_padding_mask=src_key_padding_mask
        )  # [seq_len, batch, d_model]
        
        # 转换回 [batch, seq_len, d_model]
        transformer_output = transformer_output.transpose(0, 1)
        
        # 获取序列表示（使用最后一个时间步的输出或平均池化）
        if seq_lengths is not None:
            # 使用每个序列的最后一个有效时间步
            context_embedding = transformer_output[range(batch_size), seq_lengths - 1]
        else:
            # 使用最后一个时间步
            context_embedding = transformer_output[:, -1, :]  # [batch, d_model]
        
        # 应用输出投影和归一化
        context_embedding = self.output_projection(context_embedding)
        context_embedding = self.layer_norm(context_embedding)
        
        # 同时返回每个时间步的嵌入（用于对比学习）
        sequence_embeddings = self.output_projection(transformer_output)
        sequence_embeddings = self.layer_norm(sequence_embeddings)
        
        return context_embedding, sequence_embeddings
    
    def get_context_embedding_dim(self):
        """返回上下文嵌入的维度"""
        return self.d_model


class HistoryBuffer:
    """
    用于存储和管理智能体历史观测-动作序列的缓冲区
    """
    
    def __init__(self, max_length, obs_dim, action_dim, device=torch.device("cpu")):
        self.max_length = max_length
        self.obs_dim = obs_dim  
        self.action_dim = action_dim
        self.device = device
        
        # 初始化缓冲区
        self.obs_buffer = torch.zeros(max_length, obs_dim, device=device)
        self.action_buffer = torch.zeros(max_length, action_dim, device=device)
        self.current_length = 0
        self.current_idx = 0
    
    def add(self, obs, action):
        """添加新的观测-动作对"""
        self.obs_buffer[self.current_idx] = torch.tensor(obs, device=self.device, dtype=torch.float32)
        self.action_buffer[self.current_idx] = torch.tensor(action, device=self.device, dtype=torch.float32)
        
        self.current_idx = (self.current_idx + 1) % self.max_length
        self.current_length = min(self.current_length + 1, self.max_length)
    
    def get_sequence(self):
        """获取当前存储的序列"""
        if self.current_length == 0:
            return None, None, 0
        
        if self.current_length < self.max_length:
            # 缓冲区还没满
            obs_seq = self.obs_buffer[:self.current_length]
            action_seq = self.action_buffer[:self.current_length] 
        else:
            # 缓冲区已满，需要按正确顺序重排
            obs_seq = torch.cat([
                self.obs_buffer[self.current_idx:],
                self.obs_buffer[:self.current_idx]
            ], dim=0)
            action_seq = torch.cat([
                self.action_buffer[self.current_idx:],
                self.action_buffer[:self.current_idx]
            ], dim=0)
        
        return obs_seq.unsqueeze(0), action_seq.unsqueeze(0), self.current_length
    
    def reset(self):
        """重置缓冲区"""
        self.current_length = 0
        self.current_idx = 0
        self.obs_buffer.zero_()
        self.action_buffer.zero_()

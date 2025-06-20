import random
import numpy as np
import copy
from collections import deque
from typing import Optional, List, Tuple
import pickle
import os
import time

# 尝试导入PyTorch，如果没有安装会给出提示
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Please install pytorch for RL AI: pip install torch")

from ..base_ai import BaseAI
from . import config as dqn_config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from game_2048 import Game2048


class DQN(nn.Module):
    """Deep Q-Network神经网络"""
    
    def __init__(self, input_size=16, hidden_size=256, output_size=4, dropout_rate=0.2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class RLAI(BaseAI):
    """基于强化学习（DQN）的2048游戏AI"""
    
    def __init__(self, model_path=None, training_mode=False):
        super().__init__("RL AI (DQN)")
        self.training_mode = training_mode
        
        # 设置模型路径
        if model_path is None:
            self.model_path = dqn_config.MODEL_CONFIG['default_path']
        else:
            # 如果提供的是相对路径，转换为绝对路径
            if not os.path.isabs(model_path):
                self.model_path = os.path.join('/workspaces/player-infinity/workspace/models', model_path)
            else:
                self.model_path = model_path
        
        # 确保模型目录存在
        model_dir = os.path.dirname(self.model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        if not TORCH_AVAILABLE:
            print("PyTorch not available. RL AI will use fallback greedy strategy.")
            self.use_fallback = True
            return
        
        self.use_fallback = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 从配置文件加载超参数
        self.learning_rate = dqn_config.RL_CONFIG['learning_rate']
        self.epsilon = dqn_config.RL_CONFIG['epsilon']
        self.epsilon_min = dqn_config.RL_CONFIG['epsilon_min']
        self.epsilon_decay = dqn_config.RL_CONFIG['epsilon_decay']
        self.gamma = dqn_config.RL_CONFIG['gamma']
        self.batch_size = dqn_config.RL_CONFIG['batch_size']
        self.update_target_freq = dqn_config.RL_CONFIG['update_target_freq']
        
        # 初始化网络
        network_config = dqn_config.NETWORK_CONFIG
        self.q_network = DQN(
            input_size=network_config['input_size'],
            hidden_size=network_config['hidden_size'],
            output_size=network_config['output_size'],
            dropout_rate=network_config['dropout_rate']
        ).to(self.device)
        self.target_network = DQN(
            input_size=network_config['input_size'],
            hidden_size=network_config['hidden_size'],
            output_size=network_config['output_size'],
            dropout_rate=network_config['dropout_rate']
        ).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(dqn_config.RL_CONFIG['buffer_size'])
        
        # 训练统计
        self.steps = 0
        self.episode = 0
        
        # 训练日志统计
        self.training_stats = {
            'total_loss': 0.0,
            'loss_count': 0,
            'episode_rewards': [],
            'episode_scores': [],
            'episode_max_tiles': [],
            'episode_moves': [],
            'best_score': 0,
            'best_max_tile': 0,
            'avg_reward_window': deque(maxlen=100),
            'avg_score_window': deque(maxlen=100),
            'loss_history': deque(maxlen=1000)
        }
        
        # 加载预训练模型
        self.load_model()
        
        # 更新目标网络
        self.update_target_network()
    
    def state_to_vector(self, game: Game2048) -> np.ndarray:
        """将游戏状态转换为神经网络输入向量"""
        # 将4x4网格转换为16维向量，并进行归一化
        grid = np.array(game.grid).flatten()
        # 对数变换减少数值差异，安全处理0值
        log_grid = np.zeros_like(grid, dtype=np.float32)
        nonzero_mask = grid > 0
        log_grid[nonzero_mask] = np.log2(grid[nonzero_mask])
        # 归一化到[0,1]范围
        normalized = log_grid / 11.0  # log2(2048) = 11
        return normalized.astype(np.float32)
    
    def calculate_reward(self, old_game: Game2048, new_game: Game2048, moved: bool) -> float:
        """计算奖励函数"""
        weights = dqn_config.REWARD_WEIGHTS
        
        if not moved:
            return weights['invalid_move_penalty']
        
        # 基础分数奖励
        score_reward = (new_game.score - old_game.score) / 100.0 * weights['score_weight']
        
        # 空格奖励
        old_empty = len(old_game.get_empty_cells())
        new_empty = len(new_game.get_empty_cells())
        empty_reward = (new_empty - old_empty) * weights['empty_weight']
        
        # 最大数字奖励
        old_max = old_game.get_max_tile()
        new_max = new_game.get_max_tile()
        max_tile_reward = 0
        if new_max > old_max and old_max > 0:  # 确保不会出现除零
            max_tile_reward = np.log2(new_max / old_max) * weights['max_tile_weight']
        
        # 单调性奖励（鼓励有序排列）
        monotonicity_reward = self.calculate_monotonicity(new_game.grid) * weights['monotonicity_weight']
        
        # 游戏结束惩罚
        game_over_penalty = weights['game_over_penalty'] if new_game.game_over else 0
        
        # 获胜奖励
        win_reward = weights['win_reward'] if new_game.game_won and not old_game.game_won else 0
        
        total_reward = (score_reward + empty_reward + max_tile_reward + 
                       monotonicity_reward + game_over_penalty + win_reward)
        
        return total_reward
    
    def calculate_monotonicity(self, grid) -> float:
        """计算网格的单调性"""
        monotonicity = 0
        
        # 行单调性
        for i in range(4):
            # 递增和递减
            inc = sum(1 for j in range(3) if grid[i][j] <= grid[i][j+1])
            dec = sum(1 for j in range(3) if grid[i][j] >= grid[i][j+1])
            monotonicity += max(inc, dec)
        
        # 列单调性
        for j in range(4):
            inc = sum(1 for i in range(3) if grid[i][j] <= grid[i+1][j])
            dec = sum(1 for i in range(3) if grid[i][j] >= grid[i+1][j])
            monotonicity += max(inc, dec)
        
        return monotonicity / 24.0  # 归一化
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """获取最佳移动"""
        if game.game_over or not game.can_move():
            return None
        
        # 如果PyTorch不可用，使用fallback策略
        if self.use_fallback:
            return self._fallback_move(game)
        
        # 获取有效移动
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        # 获取状态向量
        state = self.state_to_vector(game)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 如果处于训练模式且使用epsilon-greedy策略
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # 使用神经网络预测Q值
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values = q_values.cpu().numpy()[0]
        
        # 将动作索引映射到方向
        action_map = ['up', 'down', 'left', 'right']
        
        # 选择Q值最高的有效动作
        valid_q_values = [(i, q_values[i]) for i in range(4) if action_map[i] in valid_moves]
        if not valid_q_values:
            return random.choice(valid_moves)
        
        best_action_idx = max(valid_q_values, key=lambda x: x[1])[0]
        return action_map[best_action_idx]
    
    def _fallback_move(self, game: Game2048) -> Optional[str]:
        """当PyTorch不可用时的fallback策略（简单贪心）"""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        best_move = None
        best_score = -1
        
        for direction in valid_moves:
            test_game = game.clone()
            old_score = test_game.score
            if test_game.move(direction):
                score_gain = test_game.score - old_score
                empty_cells = len(test_game.get_empty_cells())
                # 综合考虑分数和空格数
                total_score = score_gain + empty_cells * 10
                if total_score > best_score:
                    best_score = total_score
                    best_move = direction
        
        return best_move if best_move else random.choice(valid_moves)
    
    def train_step(self, state, action, reward, next_state, done):
        """单步训练"""
        if self.use_fallback:
            return
        
        # 存储经验
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # 如果经验足够，进行训练
        if len(self.replay_buffer) >= self.batch_size:
            self._train_network()
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 定期更新目标网络
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.update_target_network()
    
    def _train_network(self):
        """训练神经网络"""
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 下一状态的最大Q值
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 记录损失统计
        loss_value = loss.item()
        self.training_stats['total_loss'] += loss_value
        self.training_stats['loss_count'] += 1
        self.training_stats['loss_history'].append(loss_value)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 详细训练日志（每1000步输出一次）
        if self.steps % 1000 == 0 and self.steps > 0:
            avg_loss = self.training_stats['total_loss'] / self.training_stats['loss_count']
            recent_losses = list(self.training_stats['loss_history'])[-100:]
            recent_avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            
            print(f"[TRAIN] Step {self.steps}: "
                  f"Loss={loss_value:.6f}, "
                  f"Avg_Loss={avg_loss:.6f}, "
                  f"Recent_Avg_Loss={recent_avg_loss:.6f}, "
                  f"Buffer_Size={len(self.replay_buffer)}, "
                  f"Epsilon={self.epsilon:.4f}")
            
            # 重置损失统计（避免数值过大）
            if self.training_stats['loss_count'] > 10000:
                self.training_stats['total_loss'] = avg_loss * 1000
                self.training_stats['loss_count'] = 1000
    
    def update_target_network(self):
        """更新目标网络"""
        if not self.use_fallback:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"[NETWORK] Target network updated at step {self.steps}")
    
    def save_model(self):
        """保存模型"""
        if self.use_fallback:
            return
        
        try:
            # 确保模型目录存在
            model_dir = os.path.dirname(self.model_path)
            os.makedirs(model_dir, exist_ok=True)
            
            # 如果模型文件已存在，先创建备份
            backup_path = dqn_config.MODEL_CONFIG['backup_path']
            if os.path.exists(self.model_path):
                backup_dir = os.path.dirname(backup_path)
                os.makedirs(backup_dir, exist_ok=True)
                import shutil
                shutil.copy2(self.model_path, backup_path)
                print(f"[BACKUP] Previous model backed up to {backup_path}")
            
            model_data = {
                'q_network_state_dict': self.q_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
                'episode': self.episode,
                'training_stats': self.training_stats
            }
            torch.save(model_data, self.model_path)
            print(f"[SAVE] Model saved to {self.model_path} "
                  f"(Episode: {self.episode}, Steps: {self.steps}, Epsilon: {self.epsilon:.4f})")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            # 尝试保存到备用位置
            try:
                fallback_path = os.path.join('/tmp', f"rl_model_fallback_{int(time.time())}.pth")
                torch.save(model_data, fallback_path)
                print(f"[FALLBACK] Model saved to temporary location: {fallback_path}")
            except Exception as fallback_error:
                print(f"[CRITICAL] Failed to save model to any location: {fallback_error}")
    
    def save_checkpoint(self, episode, suffix=""):
        """保存检查点模型"""
        if self.use_fallback:
            return
        
        try:
            checkpoint_dir = dqn_config.MODEL_CONFIG['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 生成检查点文件名
            timestamp = int(time.time())
            if suffix:
                checkpoint_name = f"checkpoint_ep{episode}_{suffix}_{timestamp}.pth"
            else:
                checkpoint_name = f"checkpoint_ep{episode}_{timestamp}.pth"
            
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            model_data = {
                'q_network_state_dict': self.q_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
                'episode': self.episode,
                'training_stats': self.training_stats,
                'checkpoint_info': {
                    'episode': episode,
                    'timestamp': timestamp,
                    'suffix': suffix
                }
            }
            
            torch.save(model_data, checkpoint_path)
            print(f"[CHECKPOINT] Saved checkpoint to {checkpoint_path}")
            
            # 清理旧的检查点文件（保留最近10个）
            self._cleanup_old_checkpoints(checkpoint_dir, keep_count=10)
            
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self, checkpoint_dir, keep_count=10):
        """清理旧的检查点文件"""
        try:
            checkpoint_files = []
            for file in os.listdir(checkpoint_dir):
                if file.startswith('checkpoint_') and file.endswith('.pth'):
                    file_path = os.path.join(checkpoint_dir, file)
                    checkpoint_files.append((file_path, os.path.getmtime(file_path)))
            
            # 按修改时间排序，保留最新的文件
            checkpoint_files.sort(key=lambda x: x[1], reverse=True)
            
            # 删除多余的文件
            for file_path, _ in checkpoint_files[keep_count:]:
                os.remove(file_path)
                print(f"[CLEANUP] Removed old checkpoint: {file_path}")
                
        except Exception as e:
            print(f"[WARN] Failed to cleanup old checkpoints: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点模型"""
        if self.use_fallback:
            return False
        
        try:
            if not os.path.exists(checkpoint_path):
                print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
                return False
            
            # 使用weights_only=False来兼容PyTorch 2.6
            model_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            self.q_network.load_state_dict(model_data['q_network_state_dict'])
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.steps = model_data.get('steps', 0)
            self.episode = model_data.get('episode', 0)
            
            if 'training_stats' in model_data:
                self.training_stats = model_data['training_stats']
            
            checkpoint_info = model_data.get('checkpoint_info', {})
            print(f"[CHECKPOINT] Loaded checkpoint from {checkpoint_path}")
            print(f"[CHECKPOINT] Episode: {checkpoint_info.get('episode', 'unknown')}")
            print(f"[CHECKPOINT] Timestamp: {checkpoint_info.get('timestamp', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            return False
    
    def load_model(self):
        """加载预训练模型"""
        if self.use_fallback:
            return
        
        try:
            if os.path.exists(self.model_path):
                # 使用weights_only=False来兼容PyTorch 2.6
                model_data = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.q_network.load_state_dict(model_data['q_network_state_dict'])
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                self.epsilon = model_data.get('epsilon', self.epsilon)
                self.steps = model_data.get('steps', 0)
                self.episode = model_data.get('episode', 0)
                
                # 加载训练统计（如果存在）
                if 'training_stats' in model_data:
                    self.training_stats = model_data['training_stats']
                    print(f"[LOAD] Model loaded from {self.model_path}")
                    print(f"[LOAD] Previous training: Episode {self.episode}, Steps {self.steps}")
                    print(f"[LOAD] Best score: {self.training_stats.get('best_score', 0)}")
                    print(f"[LOAD] Best max tile: {self.training_stats.get('best_max_tile', 0)}")
                else:
                    print(f"[LOAD] Model loaded from {self.model_path} (no training stats)")
            else:
                print(f"[INFO] No existing model found at {self.model_path}, starting fresh")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[INFO] Starting with fresh model")
    
    def train_episodes(self, num_episodes=1000):
        """训练多个回合"""
        if self.use_fallback:
            print("[ERROR] Cannot train without PyTorch")
            return
        
        self.training_mode = True
        print(f"[TRAIN] Starting training for {num_episodes} episodes...")
        print(f"[TRAIN] Device: {self.device}")
        print(f"[TRAIN] Learning rate: {self.learning_rate}")
        print(f"[TRAIN] Initial epsilon: {self.epsilon}")
        print(f"[TRAIN] Batch size: {self.batch_size}")
        print(f"[TRAIN] Buffer size: {dqn_config.RL_CONFIG['buffer_size']}")
        
        training_config = dqn_config.TRAINING_CONFIG
        max_moves = training_config['max_moves_per_episode']
        save_freq = training_config['save_frequency']
        
        # 训练开始时间
        start_time = time.time()
        
        # 性能统计
        episode_start_time = time.time()
        total_moves = 0
        total_reward_sum = 0
        
        for episode in range(num_episodes):
            game = Game2048()
            episode_reward = 0
            episode_moves = 0
            episode_start = time.time()
            
            # 记录初始状态
            initial_max_tile = game.get_max_tile()
            
            while not game.game_over and episode_moves < max_moves:
                old_game = game.clone()
                state = self.state_to_vector(game)
                
                # 获取动作
                move = self.get_best_move(game)
                if move is None:
                    print(f"[WARN] Episode {episode}: No valid moves available")
                    break
                
                # 执行动作
                moved = game.move(move)
                next_state = self.state_to_vector(game)
                
                # 计算奖励
                reward = self.calculate_reward(old_game, game, moved)
                episode_reward += reward
                
                # 动作映射
                action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
                action = action_map[move]
                
                # 训练
                self.train_step(state, action, reward, next_state, game.game_over)
                episode_moves += 1
                total_moves += 1
            
            # 更新训练统计
            self.episode += 1
            final_score = game.score
            final_max_tile = game.get_max_tile()
            episode_time = time.time() - episode_start
            
            # 更新最佳记录
            if final_score > self.training_stats['best_score']:
                self.training_stats['best_score'] = final_score
                print(f"[RECORD] New best score: {final_score} (Episode {episode})")
            
            if final_max_tile > self.training_stats['best_max_tile']:
                self.training_stats['best_max_tile'] = final_max_tile
                print(f"[RECORD] New best max tile: {final_max_tile} (Episode {episode})")
            
            # 记录统计数据
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_scores'].append(final_score)
            self.training_stats['episode_max_tiles'].append(final_max_tile)
            self.training_stats['episode_moves'].append(episode_moves)
            self.training_stats['avg_reward_window'].append(episode_reward)
            self.training_stats['avg_score_window'].append(final_score)
            
            total_reward_sum += episode_reward
            
            # 计算滑动平均
            avg_reward = sum(self.training_stats['avg_reward_window']) / len(self.training_stats['avg_reward_window'])
            avg_score = sum(self.training_stats['avg_score_window']) / len(self.training_stats['avg_score_window'])
            
            # 详细日志输出
            if episode % save_freq == 0 or episode == num_episodes - 1:
                # 计算训练统计
                total_time = time.time() - start_time
                episodes_per_second = (episode + 1) / total_time
                moves_per_second = total_moves / total_time
                avg_episode_time = total_time / (episode + 1)
                
                # 计算损失统计
                avg_loss = (self.training_stats['total_loss'] / self.training_stats['loss_count'] 
                           if self.training_stats['loss_count'] > 0 else 0)
                
                print(f"\n{'='*80}")
                print(f"[EPISODE] {episode + 1}/{num_episodes}")
                print(f"[STATS] Score: {final_score}, Max Tile: {final_max_tile}, Moves: {episode_moves}")
                print(f"[STATS] Episode Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
                print(f"[STATS] Avg Score: {avg_score:.0f}, Best Score: {self.training_stats['best_score']}")
                print(f"[STATS] Best Max Tile: {self.training_stats['best_max_tile']}")
                print(f"[TRAIN] Epsilon: {self.epsilon:.4f}, Steps: {self.steps}")
                print(f"[TRAIN] Avg Loss: {avg_loss:.6f}, Buffer Size: {len(self.replay_buffer)}")
                print(f"[TIME] Episode: {episode_time:.2f}s, Total: {total_time/60:.1f}min")
                print(f"[TIME] Episodes/sec: {episodes_per_second:.2f}, Moves/sec: {moves_per_second:.1f}")
                print(f"{'='*80}")
                
                # 保存模型
                self.save_model()
                
                # 每100个episode输出一次详细统计
                if episode % 100 == 0 and episode > 0:
                    self._print_detailed_stats()
        
        # 训练完成统计
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"[COMPLETE] Training completed!")
        print(f"[FINAL] Total episodes: {num_episodes}")
        print(f"[FINAL] Total steps: {self.steps}")
        print(f"[FINAL] Total moves: {total_moves}")
        print(f"[FINAL] Total time: {total_time/60:.1f} minutes")
        print(f"[FINAL] Best score: {self.training_stats['best_score']}")
        print(f"[FINAL] Best max tile: {self.training_stats['best_max_tile']}")
        print(f"[FINAL] Final epsilon: {self.epsilon:.4f}")
        print(f"[FINAL] Final avg loss: {avg_loss:.6f}")
        print(f"{'='*80}")
        
        self.training_mode = False
    
    def _print_detailed_stats(self):
        """打印详细的训练统计信息"""
        if len(self.training_stats['episode_scores']) < 10:
            return
        
        recent_scores = self.training_stats['episode_scores'][-100:]
        recent_rewards = self.training_stats['episode_rewards'][-100:]
        recent_max_tiles = self.training_stats['episode_max_tiles'][-100:]
        
        print(f"\n[DETAILED STATS] Last 100 episodes:")
        print(f"  Scores - Avg: {sum(recent_scores)/len(recent_scores):.0f}, "
              f"Min: {min(recent_scores)}, Max: {max(recent_scores)}")
        print(f"  Rewards - Avg: {sum(recent_rewards)/len(recent_rewards):.2f}, "
              f"Min: {min(recent_rewards):.2f}, Max: {max(recent_rewards):.2f}")
        print(f"  Max Tiles - Avg: {sum(recent_max_tiles)/len(recent_max_tiles):.1f}, "
              f"Min: {min(recent_max_tiles)}, Max: {max(recent_max_tiles)}")
        
        # 计算达到不同数字的百分比
        tile_counts = {}
        for tile in recent_max_tiles:
            tile_counts[tile] = tile_counts.get(tile, 0) + 1
        
        print(f"  Max Tile Distribution:")
        for tile in sorted(tile_counts.keys(), reverse=True):
            percentage = tile_counts[tile] / len(recent_max_tiles) * 100
            print(f"    {tile}: {tile_counts[tile]} times ({percentage:.1f}%)")
    
    def get_training_stats(self):
        """获取训练统计信息"""
        if not self.training_stats:
            return None
        
        stats = {
            'episode': self.episode,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'best_score': self.training_stats['best_score'],
            'best_max_tile': self.training_stats['best_max_tile'],
            'buffer_size': len(self.replay_buffer),
            'avg_loss': (self.training_stats['total_loss'] / self.training_stats['loss_count'] 
                        if self.training_stats['loss_count'] > 0 else 0)
        }
        
        # 添加最近的统计数据
        if self.training_stats['episode_scores']:
            recent_scores = self.training_stats['episode_scores'][-100:]
            recent_rewards = self.training_stats['episode_rewards'][-100:]
            recent_max_tiles = self.training_stats['episode_max_tiles'][-100:]
            
            stats.update({
                'recent_avg_score': sum(recent_scores) / len(recent_scores),
                'recent_avg_reward': sum(recent_rewards) / len(recent_rewards),
                'recent_avg_max_tile': sum(recent_max_tiles) / len(recent_max_tiles),
                'total_episodes': len(self.training_stats['episode_scores'])
            })
        
        return stats
    
    def print_current_stats(self):
        """打印当前训练统计信息"""
        stats = self.get_training_stats()
        if not stats:
            print("[INFO] No training statistics available")
            return
        
        print(f"\n{'='*50}")
        print(f"[CURRENT STATS]")
        print(f"  Episode: {stats['episode']}")
        print(f"  Steps: {stats['steps']}")
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  Best Score: {stats['best_score']}")
        print(f"  Best Max Tile: {stats['best_max_tile']}")
        print(f"  Buffer Size: {stats['buffer_size']}")
        print(f"  Avg Loss: {stats['avg_loss']:.6f}")
        
        if 'recent_avg_score' in stats:
            print(f"  Recent Avg Score: {stats['recent_avg_score']:.0f}")
            print(f"  Recent Avg Reward: {stats['recent_avg_reward']:.2f}")
            print(f"  Recent Avg Max Tile: {stats['recent_avg_max_tile']:.1f}")
            print(f"  Total Episodes: {stats['total_episodes']}")
        print(f"{'='*50}") 
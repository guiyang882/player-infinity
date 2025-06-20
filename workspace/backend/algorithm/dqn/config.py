"""
DQN强化学习算法配置文件
"""

import os
from dotenv import load_dotenv

# 加载.env文件
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# 强化学习DQN配置
RL_CONFIG = {
    'learning_rate': 0.001,      # 学习率
    'epsilon': 0.1,              # 探索率
    'epsilon_min': 0.01,         # 最小探索率
    'epsilon_decay': 0.995,      # 探索率衰减
    'gamma': 0.95,               # 折扣因子
    'batch_size': 32,            # 批处理大小
    'buffer_size': 10000,        # 经验回放缓冲区大小
    'update_target_freq': 100,   # 目标网络更新频率
    'hidden_size': 256,          # 隐藏层大小
    'description': '基于深度Q网络的强化学习算法'
}

# 网络结构配置
NETWORK_CONFIG = {
    'input_size': 16,            # 输入层大小 (4x4网格)
    'hidden_size': 256,          # 隐藏层大小
    'output_size': 4,            # 输出层大小 (上下左右)
    'dropout_rate': 0.2,         # Dropout比率
    'num_layers': 3              # 隐藏层数量
}

# 奖励函数权重配置
REWARD_WEIGHTS = {
    'score_weight': 1.0,         # 分数奖励权重
    'empty_weight': 2.0,         # 空格奖励权重
    'max_tile_weight': 10.0,     # 最大数字奖励权重
    'monotonicity_weight': 0.1,  # 单调性奖励权重
    'game_over_penalty': -100,   # 游戏结束惩罚
    'win_reward': 1000,          # 获胜奖励
    'invalid_move_penalty': -10  # 无效移动惩罚
}

# 训练配置
TRAINING_CONFIG = {
    'max_episodes': 1000,        # 最大训练回合数
    'max_moves_per_episode': 1000,  # 每回合最大移动数
    'save_frequency': 10,       # 保存模型频率
    'eval_frequency': 50,        # 评估频率
    'early_stopping_patience': 200  # 早停耐心值
}

# 模型保存配置，优先读取环境变量
MODEL_CONFIG = {
    'default_path': os.getenv('MODEL_PATH', '/workspaces/player-infinity/workspace/models/rl_model.pth'),
    'backup_path': os.getenv('BACKUP_PATH', '/workspaces/player-infinity/workspace/models/rl_model_backup.pth'),
    'checkpoint_dir': os.getenv('CHECKPOINT_DIR', '/workspaces/player-infinity/workspace/models/checkpoints')
} 
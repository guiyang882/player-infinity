"""
算法配置文件
集中管理各种AI算法的参数和权重配置
"""

# Expectimax算法配置
EXPECTIMAX_CONFIG = {
    'max_depth': 4,  # 搜索深度
    'weights': {
        'empty_cells': 2.7,     # 空格数量权重
        'max_tile': 1.0,        # 最大数字权重
        'monotonicity': 1.0,    # 单调性权重
        'smoothness': 0.1,      # 平滑性权重
        'score': 0.001,         # 游戏分数权重
        'corner_bonus': 1.5     # 角落奖励权重
    }
}

# 角落策略配置
CORNER_CONFIG = {
    'priority_moves': ['down', 'left', 'up', 'right'],  # 移动优先级
    'description': '优先级：左下 > 左上 > 右下 > 右上'
}

# 算法元信息
ALGORITHM_INFO = {
    'expectimax': {
        'name': 'Expectimax AI',
        'description': 'Expectimax算法 - 基于期望值最大化的高级AI',
        'complexity': 'High',
        'performance': 'Excellent'
    },
    'greedy': {
        'name': 'Greedy AI',
        'description': '贪心算法 - 选择能获得最高即时分数的移动',
        'complexity': 'Low',
        'performance': 'Fair'
    },
    'random': {
        'name': 'Random AI',
        'description': '随机算法 - 随机选择有效移动',
        'complexity': 'Very Low',
        'performance': 'Poor'
    },
    'corner': {
        'name': 'Corner AI',
        'description': '角落策略 - 尽量把大数字推向角落',
        'complexity': 'Low',
        'performance': 'Fair'
    }
} 
"""
2048游戏AI求解器模块

此模块提供多种AI算法的统一接口，所有算法已重构到独立的文件中
以便于维护和扩展。保持向后兼容性。
"""

from algorithm import ExpectimaxAI, GreedyAI, RandomAI, CornerAI, MCTSAI, config

# 为了保持向后兼容性，重新导出原有的类名
AI2048Solver = ExpectimaxAI

# 导出所有算法类
__all__ = [
    'AI2048Solver',     # 向后兼容的主AI类
    'ExpectimaxAI',     # Expectimax算法
    'GreedyAI',         # 贪心算法 
    'RandomAI',         # 随机算法
    'CornerAI'          # 角落策略算法
]

# 算法工厂函数，便于动态创建算法实例
def create_ai(algorithm_name: str, **kwargs):
    """
    根据算法名称创建AI实例
    
    Args:
        algorithm_name: 算法名称 ('expectimax', 'greedy', 'random', 'corner')
        **kwargs: 算法初始化参数
        
    Returns:
        对应的AI算法实例
        
    Raises:
        ValueError: 如果算法名称不被支持
    """
    algorithms = {
        'expectimax': ExpectimaxAI,
        'greedy': GreedyAI,
        'random': RandomAI,
        'corner': CornerAI,
        'mcts': MCTSAI
    }
    
    if algorithm_name.lower() not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}. "
                        f"Supported algorithms: {list(algorithms.keys())}")
    
    return algorithms[algorithm_name.lower()](**kwargs)

# 获取所有可用算法的信息
def get_available_algorithms():
    """
    获取所有可用算法的信息
    
    Returns:
        dict: 算法名称到详细信息的映射
    """
    return config.ALGORITHM_INFO.copy()

# 获取算法的简单描述信息（向后兼容）
def get_algorithm_descriptions():
    """
    获取算法的简单描述信息
    
    Returns:
        dict: 算法名称到简单描述的映射
    """
    return {name: info['description'] for name, info in config.ALGORITHM_INFO.items()} 
from abc import ABC, abstractmethod
from typing import Optional
import sys
import os

# 添加父目录到path以便导入game_2048
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_2048 import Game2048


class BaseAI(ABC):
    """2048游戏AI算法的基础抽象类"""
    
    def __init__(self, name: str = "BaseAI"):
        self.name = name
    
    @abstractmethod
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """
        获取最佳移动方向
        
        Args:
            game: 当前游戏状态
            
        Returns:
            最佳移动方向字符串 ('up', 'down', 'left', 'right') 或 None
        """
        pass
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')" 
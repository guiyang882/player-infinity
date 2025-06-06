import random
from typing import Optional
from .base_ai import BaseAI
import sys
import os

# 添加父目录到path以便导入game_2048
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_2048 import Game2048


class RandomAI(BaseAI):
    """随机算法AI：随机选择有效移动"""
    
    def __init__(self):
        super().__init__("Random AI")
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """随机选择一个有效的移动方向"""
        if game.game_over or not game.can_move():
            return None
        
        valid_moves = game.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None 
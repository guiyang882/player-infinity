import random
from typing import Optional
from .base_ai import BaseAI
from .config import CORNER_CONFIG
import sys
import os

# 添加父目录到path以便导入game_2048
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_2048 import Game2048


class CornerAI(BaseAI):
    """角落策略AI：尽量把大数字推向角落"""
    
    def __init__(self):
        super().__init__("Corner AI")
        self.priority_moves = CORNER_CONFIG['priority_moves'].copy()
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """使用角落策略选择移动方向"""
        if game.game_over or not game.can_move():
            return None
        
        valid_moves = game.get_valid_moves()
        for move in self.priority_moves:
            if move in valid_moves:
                return move
        
        return random.choice(valid_moves) if valid_moves else None 
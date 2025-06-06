import random
from typing import Optional
from .base_ai import BaseAI
import sys
import os

# 添加父目录到path以便导入game_2048
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_2048 import Game2048


class GreedyAI(BaseAI):
    """贪心算法AI：选择能获得最高即时分数的移动"""
    
    def __init__(self):
        super().__init__("Greedy AI")
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """获取能获得最高即时分数的移动方向"""
        if game.game_over or not game.can_move():
            return None
        
        best_move = None
        best_score = -1
        
        valid_moves = game.get_valid_moves()
        for direction in valid_moves:
            test_game = game.clone()
            old_score = test_game.score
            if test_game.move(direction):
                score_gain = test_game.score - old_score
                if score_gain > best_score:
                    best_score = score_gain
                    best_move = direction
        
        return best_move if best_move else random.choice(valid_moves) 
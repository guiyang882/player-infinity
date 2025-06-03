import math
import random
from typing import List, Tuple, Optional, Dict
from game_2048 import Game2048

class AI2048Solver:
    """2048游戏AI求解器，使用启发式搜索算法"""
    
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.direction_map = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
        self.direction_names = ['up', 'right', 'down', 'left']
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """获取最佳移动方向"""
        if game.game_over or not game.can_move():
            return None
        
        best_move = None
        best_score = -math.inf
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        for direction in valid_moves:
            # 创建游戏副本并执行移动
            test_game = game.clone()
            if test_game.move(direction):
                # 使用expectimax算法评估这个移动
                score = self.expectimax(test_game, self.max_depth - 1, False)
                
                if score > best_score:
                    best_score = score
                    best_move = direction
        
        return best_move
    
    def expectimax(self, game: Game2048, depth: int, is_max_player: bool) -> float:
        """Expectimax算法核心"""
        if depth == 0 or game.game_over:
            return self.evaluate_state(game)
        
        if is_max_player:
            # 玩家回合：选择最大化期望值的移动
            max_score = -math.inf
            valid_moves = game.get_valid_moves()
            
            if not valid_moves:
                return self.evaluate_state(game)
            
            for direction in valid_moves:
                test_game = game.clone()
                if test_game.move(direction):
                    score = self.expectimax(test_game, depth - 1, False)
                    max_score = max(max_score, score)
            
            return max_score
        else:
            # 随机回合：计算所有可能结果的期望值
            empty_cells = game.get_empty_cells()
            if not empty_cells:
                return self.evaluate_state(game)
            
            expected_score = 0.0
            total_prob = 0.0
            
            for row, col in empty_cells:
                # 90%概率放置2
                test_game_2 = game.clone()
                test_game_2.grid[row][col] = 2
                score_2 = self.expectimax(test_game_2, depth - 1, True)
                expected_score += 0.9 * score_2
                total_prob += 0.9
                
                # 10%概率放置4
                test_game_4 = game.clone()
                test_game_4.grid[row][col] = 4
                score_4 = self.expectimax(test_game_4, depth - 1, True)
                expected_score += 0.1 * score_4
                total_prob += 0.1
            
            return expected_score / len(empty_cells) if empty_cells else 0
    
    def evaluate_state(self, game: Game2048) -> float:
        """状态评估函数，综合多个启发式指标"""
        if game.game_over:
            return -math.inf
        
        # 权重配置
        weights = {
            'empty_cells': 2.7,
            'max_tile': 1.0,
            'monotonicity': 1.0,
            'smoothness': 0.1,
            'score': 0.001,
            'corner_bonus': 1.5
        }
        
        score = 0.0
        
        # 1. 空格数量奖励
        empty_cells = len(game.get_empty_cells())
        score += weights['empty_cells'] * empty_cells
        
        # 2. 最大数字奖励
        max_tile = game.get_max_tile()
        score += weights['max_tile'] * math.log2(max_tile) if max_tile > 0 else 0
        
        # 3. 单调性奖励
        monotonicity = self.calculate_monotonicity(game.grid)
        score += weights['monotonicity'] * monotonicity
        
        # 4. 平滑性奖励
        smoothness = self.calculate_smoothness(game.grid)
        score += weights['smoothness'] * smoothness
        
        # 5. 游戏分数
        score += weights['score'] * game.score
        
        # 6. 角落奖励
        corner_bonus = self.calculate_corner_bonus(game.grid)
        score += weights['corner_bonus'] * corner_bonus
        
        return score
    
    def calculate_monotonicity(self, grid: List[List[int]]) -> float:
        """计算单调性：相邻格子的值应该单调递增或递减"""
        totals = [0, 0, 0, 0]  # up, right, down, left
        
        # 水平方向
        for i in range(4):
            current = 0
            next_val = 1
            while next_val < 4:
                while next_val < 4 and grid[i][next_val] == 0:
                    next_val += 1
                if next_val >= 4:
                    next_val -= 1
                
                current_value = math.log2(grid[i][current]) if grid[i][current] != 0 else 0
                next_value = math.log2(grid[i][next_val]) if grid[i][next_val] != 0 else 0
                
                if current_value > next_value:
                    totals[1] += next_value - current_value
                elif next_value > current_value:
                    totals[3] += current_value - next_value
                
                current = next_val
                next_val += 1
        
        # 垂直方向
        for j in range(4):
            current = 0
            next_val = 1
            while next_val < 4:
                while next_val < 4 and grid[next_val][j] == 0:
                    next_val += 1
                if next_val >= 4:
                    next_val -= 1
                
                current_value = math.log2(grid[current][j]) if grid[current][j] != 0 else 0
                next_value = math.log2(grid[next_val][j]) if grid[next_val][j] != 0 else 0
                
                if current_value > next_value:
                    totals[2] += next_value - current_value
                elif next_value > current_value:
                    totals[0] += current_value - next_value
                
                current = next_val
                next_val += 1
        
        return max(totals[0], totals[1]) + max(totals[2], totals[3])
    
    def calculate_smoothness(self, grid: List[List[int]]) -> float:
        """计算平滑性：相邻非零格子的值应该相近"""
        smoothness = 0
        
        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    value = math.log2(grid[i][j])
                    
                    # 检查右邻居
                    if j < 3:
                        target_j = j + 1
                        while target_j < 4 and grid[i][target_j] == 0:
                            target_j += 1
                        if target_j < 4:
                            target_value = math.log2(grid[i][target_j])
                            smoothness -= abs(value - target_value)
                    
                    # 检查下邻居
                    if i < 3:
                        target_i = i + 1
                        while target_i < 4 and grid[target_i][j] == 0:
                            target_i += 1
                        if target_i < 4:
                            target_value = math.log2(grid[target_i][j])
                            smoothness -= abs(value - target_value)
        
        return smoothness
    
    def calculate_corner_bonus(self, grid: List[List[int]]) -> float:
        """计算角落奖励：最大值应该在角落"""
        max_tile = 0
        for i in range(4):
            for j in range(4):
                max_tile = max(max_tile, grid[i][j])
        
        if max_tile == 0:
            return 0
        
        corners = [grid[0][0], grid[0][3], grid[3][0], grid[3][3]]
        if max_tile in corners:
            return math.log2(max_tile)
        
        # 检查边缘
        edges = []
        for i in range(4):
            edges.extend([grid[i][0], grid[i][3]])  # 左右边缘
        for j in range(4):
            edges.extend([grid[0][j], grid[3][j]])  # 上下边缘
        
        if max_tile in edges:
            return math.log2(max_tile) * 0.5
        
        return 0


class GreedyAI:
    """贪心算法AI：选择能获得最高即时分数的移动"""
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
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


class RandomAI:
    """随机算法AI：随机选择有效移动"""
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        if game.game_over or not game.can_move():
            return None
        
        valid_moves = game.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None


class CornerAI:
    """角落策略AI：尽量把大数字推向角落"""
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        if game.game_over or not game.can_move():
            return None
        
        # 优先级：左下 > 左上 > 右下 > 右上
        priority_moves = ['down', 'left', 'up', 'right']
        
        valid_moves = game.get_valid_moves()
        for move in priority_moves:
            if move in valid_moves:
                return move
        
        return random.choice(valid_moves) if valid_moves else None 
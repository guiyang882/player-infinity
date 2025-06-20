import math
import random
import copy
from typing import List, Optional, Dict, Tuple
import sys
import os
import numpy as np
try:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../cpython')))
    from cython_mcts import rollout_simulation
    print("[MCTS] Cython加速模块已加载。")
except ImportError as e:
    print("[MCTS] Cython加速模块未加载，回退到纯Python实现。")
    def rollout_simulation(grid, score, max_depth, rollout_params):
         return _simulate_once_for_pool((grid, score, max_depth, rollout_params))

# 添加父目录到path以便导入game_2048
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_2048 import Game2048
from .base_ai import BaseAI

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    
    def __init__(self, game_state: Game2048, parent=None, move: Optional[str] = None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # 到达这个节点的移动
        self.children: Dict[str, 'MCTSNode'] = {}  # 子节点，键为移动方向
        self.visits = 0  # 访问次数
        self.value = 0.0  # 累计价值
        self.untried_moves = game_state.get_valid_moves()  # 未尝试的移动
    
    def ucb1(self, exploration_weight: float = None) -> float:
        """计算UCB1值"""
        if exploration_weight is None:
            exploration_weight = getattr(self, 'ucb1_c', 1.8)
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self) -> Tuple[str, 'MCTSNode']:
        """选择最佳子节点"""
        return max(self.children.items(), key=lambda x: x[1].ucb1())
    
    def expand(self) -> Optional[Tuple[str, 'MCTSNode']]:
        """扩展一个未尝试的移动"""
        if not self.untried_moves:
            return None
        
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        
        # 创建新的游戏状态
        new_game = self.game_state.clone()
        new_game.move(move)
        
        # 创建新的子节点
        child = MCTSNode(new_game, parent=self, move=move)
        self.children[move] = child
        return move, child
    
    def update(self, value: float):
        """更新节点统计信息"""
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.update(value)

class MCTSAI(BaseAI):
    """基于蒙特卡洛树搜索的2048 AI"""
    
    def __init__(self, name: str = "MCTSAI", simulation_count: int = 800, max_depth: int = 40, ucb1_c: float = 1.8,
                 snake_weight: int = 200, merge_weight: int = 100, corner_weight: int = 600,
                 uct_top_k: int = None, ucb1_threshold: float = None):
        super().__init__(name)
        self.simulation_count = simulation_count
        self.max_depth = max_depth
        self.ucb1_c = ucb1_c
        self.snake_weight = snake_weight
        self.merge_weight = merge_weight
        self.corner_weight = corner_weight
        self.uct_top_k = uct_top_k
        self.ucb1_threshold = ucb1_threshold
        self.last_root = None  # 持久化树根
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """串行MCTS+树持久化+UCT剪枝预留，使用 cython_mcts.rollout_simulation 加速模拟"""
        if not game.get_valid_moves():
            return None
        # 树持久化：如果有last_root且能匹配当前局面，则复用
        root = None
        if self.last_root is not None:
            for move, child in self.last_root.children.items():
                if child.game_state.grid == game.grid and child.game_state.score == game.score:
                    root = child
                    root.parent = None
                    break
        if root is None:
            root = MCTSNode(game)
        rollout_params = {
            'snake_weight': self.snake_weight,
            'merge_weight': self.merge_weight,
            'corner_weight': self.corner_weight
        }
        batch_size = min(32, self.simulation_count)  # 每批数（现在只用于分组，实际串行）
        sims_left = self.simulation_count
        while sims_left > 0:
            batch = min(batch_size, sims_left)
            nodes_to_rollout = []
            # Selection+Expansion
            for _ in range(batch):
                node = root
                while not node.untried_moves and node.children:
                    # UCT剪枝预留：可在此处加top_k或阈值筛选
                    children = list(node.children.items())
                    if self.uct_top_k:
                        children = (sorted(children, key=lambda x: x[1].ucb1(self.ucb1_c), reverse=True)[:self.uct_top_k])
                    if self.ucb1_threshold:
                        children = [c for c in children if c[1].ucb1(self.ucb1_c) >= self.ucb1_threshold]
                    if not children:
                        break
                    _, node = max(children, key=lambda x: x[1].ucb1(self.ucb1_c))
                if node.untried_moves:
                    expanded = node.expand()
                    if expanded:
                        _, node = expanded
                nodes_to_rollout.append(node)
            # 串行rollout，调用 cython_mcts.rollout_simulation 加速
            for node in nodes_to_rollout:
                value = rollout_simulation(node.game_state.grid, node.game_state.score, self.max_depth, rollout_params)
                node.update(value)
            sims_left -= batch
        # 树持久化：保存本轮root
        self.last_root = root
        if root.children:
            return max(root.children.items(), key=lambda x: x[1].visits)[0]
        valid_moves = game.get_valid_moves()
        print("[MCTS] 没有子节点，返回随机合法移动。")
        return random.choice(valid_moves) if valid_moves else None

    def _simulate(self, game: Game2048) -> float:
        """第1步用贪心+角落优先，其余全随机，提升速度"""
        sim_game = game.clone()
        depth = 0
        while depth < self.max_depth and not sim_game.game_over:
            valid_moves = sim_game.get_valid_moves()
            if not valid_moves:
                break
            if depth == 0:
                # 第一步用贪心+角落
                best_move = None
                best_score = -float('inf')
                for move in valid_moves:
                    test_game = sim_game.clone()
                    test_game.move(move)
                    score = self._evaluate(test_game)
                    if self._is_max_in_corner(test_game):
                        score += 1000
                    if score > best_score:
                        best_score = score
                        best_move = move
                sim_game.move(best_move)
            else:
                move = random.choice(valid_moves)
                sim_game.move(move)
            depth += 1
        return self._evaluate(sim_game)

    def _evaluate(self, game: Game2048) -> float:
        """更智能的评估函数：分数+最大方块+空格+单调性+角落+平滑性"""
        grid = game.grid
        score = game.score
        max_tile = game.get_max_tile()
        empty_cells = len(game.get_empty_cells())
        monotonicity = self._calc_monotonicity(grid)
        smoothness = self._calc_smoothness(grid)
        corner_bonus = 500 if self._is_max_in_corner(game) else 0
        return score + max_tile * 100 + empty_cells * 20 + monotonicity * 10 + smoothness * 2 + corner_bonus

    def _is_max_in_corner(self, game: Game2048) -> bool:
        max_tile = game.get_max_tile()
        corners = [game.grid[0][0], game.grid[0][3], game.grid[3][0], game.grid[3][3]]
        return max_tile in corners

    def _calc_monotonicity(self, grid):
        mono = 0
        # 行
        for row in grid:
            mono += self._mono_line(row)
        # 列
        for col in zip(*grid):
            mono += self._mono_line(col)
        return mono

    def _mono_line(self, line):
        inc = dec = 0
        for i in range(3):
            if line[i] > line[i+1]:
                dec += line[i] - line[i+1]
            else:
                inc += line[i+1] - line[i]
        return -min(inc, dec)

    def _calc_smoothness(self, grid):
        smooth = 0
        for i in range(4):
            for j in range(4):
                if grid[i][j] == 0:
                    continue
                v = math.log2(grid[i][j]) if grid[i][j] > 0 else 0
                for dx, dy in [(0,1),(1,0)]:
                    ni, nj = i+dx, j+dy
                    if 0<=ni<4 and 0<=nj<4 and grid[ni][nj]>0:
                        nv = math.log2(grid[ni][nj])
                        smooth -= abs(v-nv)
        return smooth
    
    def __str__(self):
        return f"{self.name}(sims={self.simulation_count}, depth={self.max_depth})" 
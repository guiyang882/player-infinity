import math
import random
import copy
from typing import List, Optional, Dict, Tuple
import sys
import os
import numpy as np

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
        """串行MCTS+树持久化+UCT剪枝预留"""
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
                        children = sorted(children, key=lambda x: x[1].ucb1(self.ucb1_c), reverse=True)[:self.uct_top_k]
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
            # 串行rollout
            for node in nodes_to_rollout:
                value = _simulate_once_for_pool((node.game_state.grid, node.game_state.score, self.max_depth, rollout_params))
                node.update(value)
            sims_left -= batch
        # 树持久化：保存本轮root
        self.last_root = root
        if root.children:
            return max(root.children.items(), key=lambda x: x[1].visits)[0]
        valid_moves = game.get_valid_moves()
        print("[MCTS] 没有子节点，返回随机合法移动。")
        return random.choice(valid_moves) if valid_moves else None

def _simulate_once_for_pool(args):
    import math
    import random
    import traceback
    try:
        if len(args) == 4:
            grid, score, max_depth, rollout_params = args
        else:
            grid, score, max_depth = args
            rollout_params = {'snake_weight': 200, 'merge_weight': 100, 'corner_weight': 600}
        np_grid = np.array(grid, dtype=np.int32)
        cur_score = score
        depth = 0
        while depth < max_depth:
            valid_moves = _get_valid_moves_np(np_grid)
            if not valid_moves:
                break
            if depth < 5:
                best_move = None
                best_score = -float('inf')
                for move in valid_moves:
                    test_grid, test_score = _move_np(np_grid, cur_score, move)
                    score_eval = _evaluate_np(test_grid, test_score, rollout_params)
                    if score_eval > best_score:
                        best_score = score_eval
                        best_move = move
                if best_move is None:
                    best_move = random.choice(valid_moves)
                np_grid, cur_score = _move_np(np_grid, cur_score, best_move)
            else:
                move = random.choice(valid_moves)
                np_grid, cur_score = _move_np(np_grid, cur_score, move)
            if _is_game_over_np(np_grid):
                break
            depth += 1
        return _evaluate_np(np_grid, cur_score, rollout_params)
    except Exception as e:
        print(f"[MCTS-Worker-NP] 模拟异常: {e}\n{traceback.format_exc()}")
        return -1

def _evaluate_np(grid, score, rollout_params):
    max_tile = np.max(grid)
    empty_cells = np.sum(grid == 0)
    monotonicity = _calc_monotonicity_np(grid)
    smoothness = _calc_smoothness_np(grid)
    corner_bonus = 500 if _is_max_in_corner_np(grid) else 0
    corner_bonus += _corner_weight_np(grid, rollout_params)
    snake_bonus = _snake_bonus_np(grid, rollout_params)
    merge_potential = _merge_potential_np(grid, rollout_params)
    return (score + max_tile * 100 + empty_cells * 20 + monotonicity * 10 + smoothness * 2 + corner_bonus
            + snake_bonus + merge_potential)

def _get_valid_moves_np(grid):
    moves = []
    for move in ['up', 'down', 'left', 'right']:
        new_grid, _ = _move_np(grid, 0, move)
        if not np.array_equal(new_grid, grid):
            moves.append(move)
    return moves

def _move_np(grid, score, direction):
    g = grid.copy()
    s = score
    if direction == 'left':
        for i in range(4):
            row = g[i][g[i] != 0]
            merged = []
            skip = False
            j = 0
            while j < len(row):
                if not skip and j+1 < len(row) and row[j] == row[j+1]:
                    merged.append(row[j]*2)
                    s += row[j]*2
                    skip = True
                else:
                    if not skip:
                        merged.append(row[j])
                    skip = False
                j += 1 if not skip else 2
            merged += [0]*(4-len(merged))
            g[i] = merged
    elif direction == 'right':
        g = np.fliplr(g)
        g, s = _move_np(g, s, 'left')
        g = np.fliplr(g)
    elif direction == 'up':
        g = g.T
        g, s = _move_np(g, s, 'left')
        g = g.T
    elif direction == 'down':
        g = g.T
        g, s = _move_np(g, s, 'right')
        g = g.T
    return g, s

def _is_game_over_np(grid):
    if np.any(grid == 0):
        return False
    for i in range(4):
        for j in range(3):
            if grid[i, j] == grid[i, j+1]:
                return False
    for j in range(4):
        for i in range(3):
            if grid[i, j] == grid[i+1, j]:
                return False
    return True

def _is_max_in_corner_np(grid):
    max_tile = np.max(grid)
    corners = [grid[0,0], grid[0,3], grid[3,0], grid[3,3]]
    return max_tile in corners

def _corner_weight_np(grid, rollout_params):
    max_tile = np.max(grid)
    corner_weight = rollout_params.get('corner_weight', 600)
    bonus = 0
    if grid[3,0] == max_tile:
        bonus += corner_weight
    if grid[3,3] == max_tile:
        bonus += corner_weight
    return bonus

def _calc_monotonicity_np(grid):
    mono = 0
    for row in grid:
        mono += _mono_line_np(row)
    for col in grid.T:
        mono += _mono_line_np(col)
    return mono

def _mono_line_np(line):
    inc = dec = 0
    for i in range(3):
        if line[i] > line[i+1]:
            dec += line[i] - line[i+1]
        else:
            inc += line[i+1] - line[i]
    return -min(inc, dec)

def _calc_smoothness_np(grid):
    smooth = 0
    for i in range(4):
        for j in range(4):
            if grid[i,j] == 0:
                continue
            v = math.log2(grid[i,j]) if grid[i,j] > 0 else 0
            for dx, dy in [(0,1),(1,0)]:
                ni, nj = i+dx, j+dy
                if 0<=ni<4 and 0<=nj<4 and grid[ni,nj]>0:
                    nv = math.log2(grid[ni,nj])
                    smooth -= abs(v-nv)
    return smooth

def _snake_bonus_np(grid, rollout_params):
    snake_weight = rollout_params.get('snake_weight', 200)
    snake_order = [
        (3,0),(3,1),(3,2),(3,3),
        (2,3),(2,2),(2,1),(2,0),
        (1,0),(1,1),(1,2),(1,3),
        (0,3),(0,2),(0,1),(0,0)
    ]
    bonus = 0
    for idx, (i,j) in enumerate(snake_order):
        if grid[i,j] > 0:
            bonus += grid[i,j] * (len(snake_order)-idx) * snake_weight // (2**idx)
    return bonus // 1000

def _merge_potential_np(grid, rollout_params):
    merge_weight = rollout_params.get('merge_weight', 100)
    potential = 0
    for i in range(4):
        for j in range(3):
            if grid[i,j] == grid[i,j+1] and grid[i,j] != 0:
                potential += math.log2(grid[i,j])
    for j in range(4):
        for i in range(3):
            if grid[i,j] == grid[i+1,j] and grid[i,j] != 0:
                potential += math.log2(grid[i,j])
    return int(potential * merge_weight)

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
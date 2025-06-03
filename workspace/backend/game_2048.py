import random
import copy
from typing import List, Tuple, Optional

class Game2048:
    """2048游戏核心逻辑类"""
    
    def __init__(self):
        self.grid = [[0 for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.reset()
    
    def reset(self):
        """重置游戏"""
        self.grid = [[0 for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.add_random_tile()
        self.add_random_tile()
    
    def add_random_tile(self):
        """添加随机数字方块(2或4)"""
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.grid[i][j] == 0]
        if empty_cells:
            row, col = random.choice(empty_cells)
            # 90%概率是2，10%概率是4
            self.grid[row][col] = 2 if random.random() < 0.9 else 4
    
    def can_move(self) -> bool:
        """检查是否还能移动"""
        # 检查是否有空格
        for i in range(4):
            for j in range(4):
                if self.grid[i][j] == 0:
                    return True
        
        # 检查是否可以合并
        for i in range(4):
            for j in range(3):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True
        
        for j in range(4):
            for i in range(3):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return True
        
        return False
    
    def is_won(self) -> bool:
        """检查是否获胜(达到2048)"""
        for i in range(4):
            for j in range(4):
                if self.grid[i][j] == 2048:
                    return True
        return False
    
    def move_left(self) -> bool:
        """向左移动"""
        moved = False
        new_score = 0
        
        for i in range(4):
            # 移除零值
            row = [val for val in self.grid[i] if val != 0]
            
            # 合并相同数字
            merged_row = []
            skip_next = False
            
            for j in range(len(row)):
                if skip_next:
                    skip_next = False
                    continue
                
                if j < len(row) - 1 and row[j] == row[j + 1]:
                    # 合并
                    merged_value = row[j] * 2
                    merged_row.append(merged_value)
                    new_score += merged_value
                    skip_next = True
                else:
                    merged_row.append(row[j])
            
            # 补充零值
            while len(merged_row) < 4:
                merged_row.append(0)
            
            # 检查是否有变化
            if self.grid[i] != merged_row:
                moved = True
                self.grid[i] = merged_row
        
        self.score += new_score
        return moved
    
    def move_right(self) -> bool:
        """向右移动"""
        # 翻转，向左移动，再翻转回来
        for i in range(4):
            self.grid[i].reverse()
        
        moved = self.move_left()
        
        for i in range(4):
            self.grid[i].reverse()
        
        return moved
    
    def move_up(self) -> bool:
        """向上移动"""
        # 转置，向左移动，再转置回来
        self.transpose()
        moved = self.move_left()
        self.transpose()
        return moved
    
    def move_down(self) -> bool:
        """向下移动"""
        # 转置，向右移动，再转置回来
        self.transpose()
        moved = self.move_right()
        self.transpose()
        return moved
    
    def transpose(self):
        """转置矩阵"""
        self.grid = [[self.grid[j][i] for j in range(4)] for i in range(4)]
    
    def move(self, direction: str) -> bool:
        """执行移动操作
        
        Args:
            direction: 移动方向 'up', 'down', 'left', 'right'
        
        Returns:
            bool: 是否成功移动
        """
        if self.game_over:
            return False
        
        moved = False
        
        if direction == 'left':
            moved = self.move_left()
        elif direction == 'right':
            moved = self.move_right()
        elif direction == 'up':
            moved = self.move_up()
        elif direction == 'down':
            moved = self.move_down()
        
        if moved:
            self.add_random_tile()
            
            if self.is_won():
                self.game_won = True
            
            if not self.can_move():
                self.game_over = True
        
        return moved
    
    def get_valid_moves(self) -> List[str]:
        """获取所有有效的移动方向"""
        valid_moves = []
        directions = ['up', 'down', 'left', 'right']
        
        for direction in directions:
            # 创建游戏状态副本进行测试
            test_game = copy.deepcopy(self)
            if test_game.move(direction):
                valid_moves.append(direction)
        
        return valid_moves
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """获取所有空格位置"""
        return [(i, j) for i in range(4) for j in range(4) if self.grid[i][j] == 0]
    
    def get_max_tile(self) -> int:
        """获取最大数字方块"""
        max_tile = 0
        for i in range(4):
            for j in range(4):
                max_tile = max(max_tile, self.grid[i][j])
        return max_tile
    
    def clone(self) -> 'Game2048':
        """创建游戏状态的深拷贝"""
        return copy.deepcopy(self)
    
    def set_grid(self, grid: List[List[int]], score: int = 0):
        """设置游戏状态"""
        self.grid = copy.deepcopy(grid)
        self.score = score
        self.game_over = not self.can_move()
        self.game_won = self.is_won()
    
    def __str__(self) -> str:
        """打印游戏状态"""
        result = f"Score: {self.score}\n"
        result += "-" * 17 + "\n"
        for row in self.grid:
            result += "|"
            for cell in row:
                if cell == 0:
                    result += "    "
                else:
                    result += f"{cell:4d}"
            result += "|\n"
        result += "-" * 17
        return result 
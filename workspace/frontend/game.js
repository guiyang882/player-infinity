class Game2048 {
    constructor() {
        this.grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ];
        this.score = 0;
        this.best = localStorage.getItem('best2048') || 0;
        this.gameWon = false;
        this.gameOver = false;
        this.autoPlayInterval = null;
        this.currentAlgorithm = 'random';
        this.init();
    }

    init() {
        this.setupGrid();
        this.addEventListeners();
        this.setupAlgorithmSelection();
        this.restart();
    }

    setupGrid() {
        const gridContainer = document.getElementById('grid-container');
        gridContainer.innerHTML = '';
        
        for (let i = 0; i < 4; i++) {
            const row = document.createElement('div');
            row.className = 'grid-row';
            for (let j = 0; j < 4; j++) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                row.appendChild(cell);
            }
            gridContainer.appendChild(row);
        }
    }

    setupAlgorithmSelection() {
        const options = document.querySelectorAll('.algorithm-option');
        options.forEach(option => {
            option.addEventListener('click', () => {
                // 移除所有选中状态
                options.forEach(opt => opt.classList.remove('selected'));
                // 添加选中状态
                option.classList.add('selected');
                // 设置当前算法
                this.currentAlgorithm = option.dataset.algorithm;
            });
        });
        
        // 默认选中随机算法
        const defaultOption = document.querySelector('[data-algorithm="random"]');
        if (defaultOption) {
            defaultOption.classList.add('selected');
        }
    }

    restart() {
        console.log('重新开始游戏...');
        this.grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ];
        this.score = 0;
        this.gameWon = false;
        this.gameOver = false;
        this.stopAutoPlay();
        
        console.log('添加初始方块...');
        this.addRandomTile();
        this.addRandomTile();
        console.log('初始网格状态:', this.grid);
        
        this.updateDisplay();
        this.hideMessage();
        
        console.log('游戏重新开始完成');
    }

    addRandomTile() {
        console.log('开始添加随机方块...');
        const emptyCells = [];
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                if (this.grid[i][j] === 0) {
                    emptyCells.push({row: i, col: j});
                }
            }
        }
        
        console.log('找到空格子数量:', emptyCells.length);
        
        if (emptyCells.length > 0) {
            const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
            const newValue = Math.random() < 0.9 ? 2 : 4;
            this.grid[randomCell.row][randomCell.col] = newValue;
            console.log(`在位置 (${randomCell.row}, ${randomCell.col}) 添加了方块 ${newValue}`);
        } else {
            console.log('没有空格子可以添加方块');
        }
    }

    updateDisplay() {
        console.log('开始更新显示...', this.grid);
        
        // 安全检查：确保grid是正确的4x4数组
        if (!this.grid || !Array.isArray(this.grid) || this.grid.length !== 4) {
            console.error('Grid未正确初始化:', this.grid);
            return;
        }
        
        // 检查每一行是否是正确的数组
        for (let i = 0; i < 4; i++) {
            if (!Array.isArray(this.grid[i]) || this.grid[i].length !== 4) {
                console.error(`Grid第${i}行未正确初始化:`, this.grid[i]);
                return;
            }
        }
        
        const tileContainer = document.getElementById('tile-container');
        
        if (!tileContainer) {
            console.error('找不到tile-container元素！');
            return;
        }
        
        tileContainer.innerHTML = '';
        let tileCount = 0;
        
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                if (this.grid[i][j] !== 0) {
                    const tile = document.createElement('div');
                    tile.className = `tile tile-${this.grid[i][j]}`;
                    tile.textContent = this.grid[i][j];
                    tile.style.left = `${j * (107 + 15)}px`;
                    tile.style.top = `${i * (107 + 15)}px`;
                    tileContainer.appendChild(tile);
                    tileCount++;
                    console.log(`显示方块: ${this.grid[i][j]} 在位置 (${i}, ${j}), CSS位置: (${tile.style.left}, ${tile.style.top})`);
                }
            }
        }
        
        console.log(`总共显示了 ${tileCount} 个方块`);
        
        // 确保分数元素存在再更新
        const scoreElement = document.getElementById('score');
        const bestElement = document.getElementById('best');
        if (scoreElement) {
            scoreElement.textContent = this.score;
            console.log('更新分数:', this.score);
        } else {
            console.error('找不到score元素！');
        }
        
        if (bestElement) {
            bestElement.textContent = this.best;
            console.log('更新最高分:', this.best);
        } else {
            console.error('找不到best元素！');
        }
    }

    move(direction) {
        if (this.gameOver) return false;
        
        let moved = false;
        const newGrid = this.grid.map(row => [...row]);
        
        switch (direction) {
            case 'ArrowLeft':
                moved = this.moveLeft(newGrid);
                break;
            case 'ArrowRight':
                moved = this.moveRight(newGrid);
                break;
            case 'ArrowUp':
                moved = this.moveUp(newGrid);
                break;
            case 'ArrowDown':
                moved = this.moveDown(newGrid);
                break;
        }
        
        if (moved) {
            this.grid = newGrid;
            this.addRandomTile();
            this.updateDisplay();
            this.checkGameState();
        }
        
        return moved;
    }

    moveLeft(grid) {
        let moved = false;
        for (let i = 0; i < 4; i++) {
            const row = grid[i].filter(val => val !== 0);
            for (let j = 0; j < row.length - 1; j++) {
                if (row[j] === row[j + 1]) {
                    row[j] *= 2;
                    this.score += row[j];
                    row.splice(j + 1, 1);
                }
            }
            while (row.length < 4) {
                row.push(0);
            }
            
            for (let j = 0; j < 4; j++) {
                if (grid[i][j] !== row[j]) {
                    moved = true;
                }
                grid[i][j] = row[j];
            }
        }
        return moved;
    }

    moveRight(grid) {
        let moved = false;
        for (let i = 0; i < 4; i++) {
            const row = grid[i].filter(val => val !== 0);
            for (let j = row.length - 1; j > 0; j--) {
                if (row[j] === row[j - 1]) {
                    row[j] *= 2;
                    this.score += row[j];
                    row.splice(j - 1, 1);
                    j--;
                }
            }
            while (row.length < 4) {
                row.unshift(0);
            }
            
            for (let j = 0; j < 4; j++) {
                if (grid[i][j] !== row[j]) {
                    moved = true;
                }
                grid[i][j] = row[j];
            }
        }
        return moved;
    }

    moveUp(grid) {
        let moved = false;
        for (let j = 0; j < 4; j++) {
            const column = [];
            for (let i = 0; i < 4; i++) {
                if (grid[i][j] !== 0) {
                    column.push(grid[i][j]);
                }
            }
            
            for (let i = 0; i < column.length - 1; i++) {
                if (column[i] === column[i + 1]) {
                    column[i] *= 2;
                    this.score += column[i];
                    column.splice(i + 1, 1);
                }
            }
            
            while (column.length < 4) {
                column.push(0);
            }
            
            for (let i = 0; i < 4; i++) {
                if (grid[i][j] !== column[i]) {
                    moved = true;
                }
                grid[i][j] = column[i];
            }
        }
        return moved;
    }

    moveDown(grid) {
        let moved = false;
        for (let j = 0; j < 4; j++) {
            const column = [];
            for (let i = 0; i < 4; i++) {
                if (grid[i][j] !== 0) {
                    column.push(grid[i][j]);
                }
            }
            
            for (let i = column.length - 1; i > 0; i--) {
                if (column[i] === column[i - 1]) {
                    column[i] *= 2;
                    this.score += column[i];
                    column.splice(i - 1, 1);
                    i--;
                }
            }
            
            while (column.length < 4) {
                column.unshift(0);
            }
            
            for (let i = 0; i < 4; i++) {
                if (grid[i][j] !== column[i]) {
                    moved = true;
                }
                grid[i][j] = column[i];
            }
        }
        return moved;
    }

    checkGameState() {
        // 检查是否达到2048
        if (!this.gameWon) {
            for (let i = 0; i < 4; i++) {
                for (let j = 0; j < 4; j++) {
                    if (this.grid[i][j] === 2048) {
                        this.gameWon = true;
                        this.showMessage('你赢了!', 'game-won');
                        return;
                    }
                }
            }
        }
        
        // 检查游戏是否结束
        if (this.isGameOver()) {
            this.gameOver = true;
            this.showMessage('游戏结束!', 'game-over');
        }
        
        // 更新最高分
        if (this.score > this.best) {
            this.best = this.score;
            localStorage.setItem('best2048', this.best);
        }
    }

    isGameOver() {
        // 检查是否有空格
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                if (this.grid[i][j] === 0) {
                    return false;
                }
            }
        }
        
        // 检查是否可以合并
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 3; j++) {
                if (this.grid[i][j] === this.grid[i][j + 1]) {
                    return false;
                }
            }
        }
        
        for (let j = 0; j < 4; j++) {
            for (let i = 0; i < 3; i++) {
                if (this.grid[i][j] === this.grid[i + 1][j]) {
                    return false;
                }
            }
        }
        
        return true;
    }

    showMessage(text, className) {
        const messageElement = document.getElementById('game-message');
        const textElement = document.getElementById('message-text');
        textElement.textContent = text;
        messageElement.className = `game-message ${className}`;
        messageElement.style.display = 'flex';
    }

    hideMessage() {
        const messageElement = document.getElementById('game-message');
        messageElement.style.display = 'none';
    }

    addEventListeners() {
        document.addEventListener('keydown', (e) => {
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
                e.preventDefault();
                this.move(e.key);
            }
        });
    }

    // 算法选择相关方法
    showAlgorithmSelection() {
        const modal = document.getElementById('algorithm-modal');
        modal.style.display = 'flex';
    }

    hideAlgorithmSelection() {
        const modal = document.getElementById('algorithm-modal');
        modal.style.display = 'none';
    }

    startAutoPlayWithAlgorithm() {
        this.hideAlgorithmSelection();
        this.updateCurrentAlgorithmDisplay();
        this.autoPlay();
    }

    updateCurrentAlgorithmDisplay() {
        const algorithmNames = {
            'random': '随机算法',
            'greedy': '贪心算法',
            'corner': '角落策略',
            'expectimax': '期望值算法'
        };
        const currentAlgorithmElement = document.getElementById('current-algorithm');
        if (currentAlgorithmElement) {
            currentAlgorithmElement.textContent = `当前算法: ${algorithmNames[this.currentAlgorithm]}`;
        }
    }

    // AI相关方法
    async makeAIMove() {
        const direction = await this.getAIMove();
        if (direction) {
            this.move(direction);
        }
    }

    async getAIMove() {
        // 调用后端Python算法
        try {
            const response = await fetch(`/api/ai_move/${this.currentAlgorithm}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    grid: this.grid,
                    score: this.score
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Backend AI response:', data);
            return data.direction;
        } catch (error) {
            console.error('Error calling backend AI:', error);
            // 如果后端调用失败，回退到前端算法
            return this.getFallbackAIMove();
        }
    }

    // 备用的前端算法（当后端不可用时使用）
    getFallbackAIMove() {
        console.log('Using fallback frontend AI algorithms');
        switch (this.currentAlgorithm) {
            case 'random':
                return this.getRandomMove();
            case 'greedy':
                return this.getGreedyMove();
            case 'corner':
                return this.getCornerMove();
            case 'expectimax':
                return this.getExpectimaxMove();
            default:
                return this.getRandomMove();
        }
    }

    // 随机算法
    getRandomMove() {
        const directions = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
        const validMoves = directions.filter(dir => {
            const testGrid = this.grid.map(row => [...row]);
            return this.testMove(dir, testGrid);
        });
        
        if (validMoves.length > 0) {
            return validMoves[Math.floor(Math.random() * validMoves.length)];
        }
        return null;
    }

    // 贪心算法
    getGreedyMove() {
        const directions = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
        let bestMove = null;
        let bestScore = -1;
        
        for (const direction of directions) {
            const testGrid = this.grid.map(row => [...row]);
            let tempScore = 0; // 临时分数变量来模拟移动
            
            // 模拟移动并计算实际得分
            const scoreGain = this.simulateMove(direction, testGrid);
            
            if (scoreGain > bestScore) {
                bestScore = scoreGain;
                bestMove = direction;
            }
        }
        
        // 如果没有找到有分数增益的移动，随便选一个有效移动
        if (!bestMove) {
            const validMoves = directions.filter(dir => {
                const testGrid = this.grid.map(row => [...row]);
                return this.testMove(dir, testGrid);
            });
            if (validMoves.length > 0) {
                bestMove = validMoves[0];
            }
        }
        
        return bestMove;
    }

    // 角落策略
    getCornerMove() {
        const directions = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
        let bestMove = null;
        let bestScore = -Infinity;
        
        for (const direction of directions) {
            const testGrid = this.grid.map(row => [...row]);
            
            if (this.testMove(direction, testGrid)) {
                const score = this.evaluateCornerStrategy(testGrid);
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = direction;
                }
            }
        }
        
        return bestMove;
    }

    // 期望值算法（简化版）
    getExpectimaxMove() {
        const directions = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
        let bestMove = null;
        let bestScore = -Infinity;
        
        for (const direction of directions) {
            const testGrid = this.grid.map(row => [...row]);
            
            if (this.testMove(direction, testGrid)) {
                const score = this.expectimax(testGrid, 2, false);
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = direction;
                }
            }
        }
        
        return bestMove;
    }

    // 辅助方法
    simulateMove(direction, grid) {
        let scoreGain = 0;
        let moved = false;
        
        switch (direction) {
            case 'ArrowLeft':
                moved = this.simulateMoveLeft(grid, (points) => { scoreGain += points; });
                break;
            case 'ArrowRight':
                moved = this.simulateMoveRight(grid, (points) => { scoreGain += points; });
                break;
            case 'ArrowUp':
                moved = this.simulateMoveUp(grid, (points) => { scoreGain += points; });
                break;
            case 'ArrowDown':
                moved = this.simulateMoveDown(grid, (points) => { scoreGain += points; });
                break;
        }
        
        return moved ? scoreGain : -1;
    }

    simulateMoveLeft(grid, addScore) {
        let moved = false;
        for (let i = 0; i < 4; i++) {
            const row = grid[i].filter(val => val !== 0);
            for (let j = 0; j < row.length - 1; j++) {
                if (row[j] === row[j + 1]) {
                    row[j] *= 2;
                    addScore(row[j]);
                    row.splice(j + 1, 1);
                }
            }
            while (row.length < 4) {
                row.push(0);
            }
            
            for (let j = 0; j < 4; j++) {
                if (grid[i][j] !== row[j]) {
                    moved = true;
                }
                grid[i][j] = row[j];
            }
        }
        return moved;
    }

    simulateMoveRight(grid, addScore) {
        let moved = false;
        for (let i = 0; i < 4; i++) {
            const row = grid[i].filter(val => val !== 0);
            for (let j = row.length - 1; j > 0; j--) {
                if (row[j] === row[j - 1]) {
                    row[j] *= 2;
                    addScore(row[j]);
                    row.splice(j - 1, 1);
                    j--;
                }
            }
            while (row.length < 4) {
                row.unshift(0);
            }
            
            for (let j = 0; j < 4; j++) {
                if (grid[i][j] !== row[j]) {
                    moved = true;
                }
                grid[i][j] = row[j];
            }
        }
        return moved;
    }

    simulateMoveUp(grid, addScore) {
        let moved = false;
        for (let j = 0; j < 4; j++) {
            const column = [];
            for (let i = 0; i < 4; i++) {
                if (grid[i][j] !== 0) {
                    column.push(grid[i][j]);
                }
            }
            
            for (let i = 0; i < column.length - 1; i++) {
                if (column[i] === column[i + 1]) {
                    column[i] *= 2;
                    addScore(column[i]);
                    column.splice(i + 1, 1);
                }
            }
            
            while (column.length < 4) {
                column.push(0);
            }
            
            for (let i = 0; i < 4; i++) {
                if (grid[i][j] !== column[i]) {
                    moved = true;
                }
                grid[i][j] = column[i];
            }
        }
        return moved;
    }

    simulateMoveDown(grid, addScore) {
        let moved = false;
        for (let j = 0; j < 4; j++) {
            const column = [];
            for (let i = 0; i < 4; i++) {
                if (grid[i][j] !== 0) {
                    column.push(grid[i][j]);
                }
            }
            
            for (let i = column.length - 1; i > 0; i--) {
                if (column[i] === column[i - 1]) {
                    column[i] *= 2;
                    addScore(column[i]);
                    column.splice(i - 1, 1);
                    i--;
                }
            }
            
            while (column.length < 4) {
                column.unshift(0);
            }
            
            for (let i = 0; i < 4; i++) {
                if (grid[i][j] !== column[i]) {
                    moved = true;
                }
                grid[i][j] = column[i];
            }
        }
        return moved;
    }

    calculateScore(grid) {
        let score = 0;
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                if (grid[i][j] > 0) {
                    score += grid[i][j];
                }
            }
        }
        return score;
    }

    evaluateCornerStrategy(grid) {
        let score = 0;
        const maxValue = Math.max(...grid.flat());
        
        // 检查最大值是否在角落
        const corners = [grid[0][0], grid[0][3], grid[3][0], grid[3][3]];
        if (corners.includes(maxValue)) {
            score += maxValue * 10;
        }
        
        // 单调性奖励
        score += this.calculateMonotonicity(grid);
        
        // 平滑性奖励
        score += this.calculateSmoothness(grid);
        
        // 空格奖励
        score += this.countEmptyTiles(grid) * 100;
        
        return score;
    }

    calculateMonotonicity(grid) {
        let score = 0;
        
        // 行单调性
        for (let i = 0; i < 4; i++) {
            let increasing = 0, decreasing = 0;
            for (let j = 0; j < 3; j++) {
                if (grid[i][j] <= grid[i][j + 1]) increasing++;
                if (grid[i][j] >= grid[i][j + 1]) decreasing++;
            }
            score += Math.max(increasing, decreasing);
        }
        
        // 列单调性
        for (let j = 0; j < 4; j++) {
            let increasing = 0, decreasing = 0;
            for (let i = 0; i < 3; i++) {
                if (grid[i][j] <= grid[i + 1][j]) increasing++;
                if (grid[i][j] >= grid[i + 1][j]) decreasing++;
            }
            score += Math.max(increasing, decreasing);
        }
        
        return score * 10;
    }

    calculateSmoothness(grid) {
        let score = 0;
        
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                if (grid[i][j] !== 0) {
                    // 检查右邻居
                    if (j < 3 && grid[i][j + 1] !== 0) {
                        score -= Math.abs(Math.log2(grid[i][j]) - Math.log2(grid[i][j + 1]));
                    }
                    // 检查下邻居
                    if (i < 3 && grid[i + 1][j] !== 0) {
                        score -= Math.abs(Math.log2(grid[i][j]) - Math.log2(grid[i + 1][j]));
                    }
                }
            }
        }
        
        return score;
    }

    countEmptyTiles(grid) {
        let count = 0;
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                if (grid[i][j] === 0) count++;
            }
        }
        return count;
    }

    expectimax(grid, depth, isPlayerTurn) {
        if (depth === 0) {
            return this.evaluateCornerStrategy(grid);
        }
        
        if (isPlayerTurn) {
            let maxScore = -Infinity;
            const directions = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
            
            for (const direction of directions) {
                const testGrid = grid.map(row => [...row]);
                if (this.testMove(direction, testGrid)) {
                    const score = this.expectimax(testGrid, depth - 1, false);
                    maxScore = Math.max(maxScore, score);
                }
            }
            
            return maxScore === -Infinity ? 0 : maxScore;
        } else {
            let totalScore = 0;
            let count = 0;
            
            for (let i = 0; i < 4; i++) {
                for (let j = 0; j < 4; j++) {
                    if (grid[i][j] === 0) {
                        // 尝试放置2
                        const testGrid2 = grid.map(row => [...row]);
                        testGrid2[i][j] = 2;
                        totalScore += this.expectimax(testGrid2, depth - 1, true) * 0.9;
                        count++;
                        
                        // 尝试放置4
                        const testGrid4 = grid.map(row => [...row]);
                        testGrid4[i][j] = 4;
                        totalScore += this.expectimax(testGrid4, depth - 1, true) * 0.1;
                    }
                }
            }
            
            return count > 0 ? totalScore / count : 0;
        }
    }

    testMove(direction, grid) {
        switch (direction) {
            case 'ArrowLeft':
                return this.moveLeft(grid);
            case 'ArrowRight':
                return this.moveRight(grid);
            case 'ArrowUp':
                return this.moveUp(grid);
            case 'ArrowDown':
                return this.moveDown(grid);
        }
        return false;
    }

    autoPlay() {
        if (this.autoPlayInterval) return;
        
        this.autoPlayInterval = setInterval(() => {
            if (this.gameOver) {
                this.stopAutoPlay();
                return;
            }
            this.makeAIMove();
        }, 200);
    }

    stopAutoPlay() {
        if (this.autoPlayInterval) {
            clearInterval(this.autoPlayInterval);
            this.autoPlayInterval = null;
        }
    }
}

// 游戏初始化
let game = null;

// 添加调试信息
console.log('脚本开始执行');

// 确保DOM完全加载后再初始化游戏
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM已加载，开始初始化游戏...');
    
    // 检查必要的DOM元素是否存在
    const gridContainer = document.getElementById('grid-container');
    const tileContainer = document.getElementById('tile-container');
    const scoreElement = document.getElementById('score');
    const bestElement = document.getElementById('best');
    
    console.log('DOM元素检查:', {
        gridContainer: !!gridContainer,
        tileContainer: !!tileContainer,
        scoreElement: !!scoreElement,
        bestElement: !!bestElement
    });
    
    if (gridContainer && tileContainer && scoreElement && bestElement) {
        // 创建游戏实例
        game = new Game2048();
        console.log('游戏实例创建成功:', game);
        
        // 手动触发一次重新开始来确保初始化
        setTimeout(() => {
            console.log('执行游戏重新开始...');
            game.restart();
        }, 100);
    } else {
        console.error('必要的DOM元素缺失，无法初始化游戏');
    }
});

// 备用初始化 - 如果DOMContentLoaded已经触发
if (document.readyState === 'loading') {
    console.log('文档仍在加载中，等待DOMContentLoaded事件');
} else {
    console.log('文档已加载完成，立即初始化游戏');
    setTimeout(() => {
        if (!game) {
            game = new Game2048();
            console.log('备用初始化完成:', game);
        }
    }, 50);
}

// 确保所有元素都已正确创建
window.addEventListener('load', function() {
    console.log('页面完全加载');
    if (game) {
        console.log('游戏状态:', {
            grid: game.grid,
            score: game.score,
            gameOver: game.gameOver
        });
        
        // 再次确保显示正确
        game.updateDisplay();
    }
});
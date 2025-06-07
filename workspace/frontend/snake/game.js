// 游戏变量
let canvas;
let ctx;
let snake = [{ x: 10, y: 10 }];
let food = {};
let dx = 0;
let dy = 0;
let score = 0;
let highScore = 0;
let gameRunning = false;
let gamePaused = false;
let gameLoop;
let gameSpeed = 150; // 毫秒

// 游戏配置
const GRID_SIZE = 20;
const TILE_COUNT = 20;

// DOM 元素
let scoreElement;
let highScoreElement;
let lengthElement;
let gameOverElement;
let finalScoreElement;
let startBtn;
let pauseBtn;
let resumeBtn;
let speedSlider;
let speedValue;

// 初始化游戏
function init() {
    canvas = document.getElementById('gameCanvas');
    ctx = canvas.getContext('2d');
    
    // 获取DOM元素
    scoreElement = document.getElementById('score');
    highScoreElement = document.getElementById('high-score');
    lengthElement = document.getElementById('length');
    gameOverElement = document.getElementById('gameOver');
    finalScoreElement = document.getElementById('finalScore');
    startBtn = document.getElementById('startBtn');
    pauseBtn = document.getElementById('pauseBtn');
    resumeBtn = document.getElementById('resumeBtn');
    speedSlider = document.getElementById('speedSlider');
    speedValue = document.getElementById('speedValue');
    
    // 加载最高分
    highScore = localStorage.getItem('snakeHighScore') || 0;
    highScoreElement.textContent = highScore;
    
    // 设置事件监听器
    document.addEventListener('keydown', handleKeyPress);
    speedSlider.addEventListener('input', updateSpeed);
    
    // 移动端控制按钮
    const controlBtns = document.querySelectorAll('.control-btn');
    controlBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const direction = e.target.dataset.direction;
            handleDirectionChange(direction);
        });
    });
    
    // 生成初食物
    generateFood();
    drawGame();
}

// 开始游戏
function startGame() {
    if (gameRunning) return;
    
    // 重置游戏状态
    snake = [{ x: 10, y: 10 }];
    dx = 0;
    dy = 0;
    score = 0;
    gameRunning = true;
    gamePaused = false;
    
    // 更新UI
    updateScore();
    updateLength();
    gameOverElement.classList.remove('show');
    startBtn.disabled = true;
    pauseBtn.disabled = false;
    
    // 生成新食物
    generateFood();
    
    // 开始游戏循环
    gameLoop = setInterval(update, gameSpeed);
}

// 暂停游戏
function pauseGame() {
    if (!gameRunning || gamePaused) return;
    
    gamePaused = true;
    clearInterval(gameLoop);
    pauseBtn.style.display = 'none';
    resumeBtn.style.display = 'inline-block';
}

// 继续游戏
function resumeGame() {
    if (!gameRunning || !gamePaused) return;
    
    gamePaused = false;
    gameLoop = setInterval(update, gameSpeed);
    pauseBtn.style.display = 'inline-block';
    resumeBtn.style.display = 'none';
}

// 结束游戏
function endGame() {
    gameRunning = false;
    gamePaused = false;
    clearInterval(gameLoop);
    
    // 更新最高分
    if (score > highScore) {
        highScore = score;
        localStorage.setItem('snakeHighScore', highScore);
        highScoreElement.textContent = highScore;
    }
    
    // 显示游戏结束画面
    finalScoreElement.textContent = score;
    gameOverElement.classList.add('show');
    
    // 更新按钮状态
    startBtn.disabled = false;
    pauseBtn.disabled = true;
    pauseBtn.style.display = 'inline-block';
    resumeBtn.style.display = 'none';
}

// 游戏主循环
function update() {
    if (!gameRunning || gamePaused) return;
    
    // 如果蛇还没有开始移动，不执行更新
    if (dx === 0 && dy === 0) return;
    
    // 移动蛇头
    const head = { x: snake[0].x + dx, y: snake[0].y + dy };
    
    // 检查碰撞
    if (checkCollision(head)) {
        endGame();
        return;
    }
    
    snake.unshift(head);
    
    // 检查是否吃到食物
    if (head.x === food.x && head.y === food.y) {
        score += 10;
        updateScore();
        updateLength();
        generateFood();
    } else {
        snake.pop();
    }
    
    drawGame();
}

// 检查碰撞
function checkCollision(head) {
    // 检查墙壁碰撞
    if (head.x < 0 || head.x >= TILE_COUNT || head.y < 0 || head.y >= TILE_COUNT) {
        return true;
    }
    
    // 检查自身碰撞
    for (let segment of snake) {
        if (head.x === segment.x && head.y === segment.y) {
            return true;
        }
    }
    
    return false;
}

// 生成食物
function generateFood() {
    do {
        food = {
            x: Math.floor(Math.random() * TILE_COUNT),
            y: Math.floor(Math.random() * TILE_COUNT)
        };
    } while (snake.some(segment => segment.x === food.x && segment.y === food.y));
}

// 绘制游戏画面
function drawGame() {
    // 清空画布
    ctx.fillStyle = '#f9f9f9';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 绘制网格
    drawGrid();
    
    // 绘制蛇
    drawSnake();
    
    // 绘制食物
    drawFood();
    
    // 如果游戏正在运行但蛇还没有开始移动，显示提示
    if (gameRunning && dx === 0 && dy === 0) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = 'white';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('按方向键开始游戏', canvas.width / 2, canvas.height / 2 - 10);
        ctx.font = '14px Arial';
        ctx.fillText('或点击下方按钮控制', canvas.width / 2, canvas.height / 2 + 15);
    }
}

// 绘制网格
function drawGrid() {
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= TILE_COUNT; i++) {
        // 垂直线
        ctx.beginPath();
        ctx.moveTo(i * GRID_SIZE, 0);
        ctx.lineTo(i * GRID_SIZE, canvas.height);
        ctx.stroke();
        
        // 水平线
        ctx.beginPath();
        ctx.moveTo(0, i * GRID_SIZE);
        ctx.lineTo(canvas.width, i * GRID_SIZE);
        ctx.stroke();
    }
}

// 绘制蛇
function drawSnake() {
    snake.forEach((segment, index) => {
        ctx.fillStyle = index === 0 ? '#2E7D32' : '#4CAF50'; // 蛇头颜色稍深
        ctx.fillRect(
            segment.x * GRID_SIZE + 1,
            segment.y * GRID_SIZE + 1,
            GRID_SIZE - 2,
            GRID_SIZE - 2
        );
        
        // 蛇头添加眼睛
        if (index === 0) {
            ctx.fillStyle = 'white';
            const eyeSize = 3;
            const eyeOffset = 6;
            
            // 根据移动方向确定眼睛位置
            let eye1X, eye1Y, eye2X, eye2Y;
            const centerX = segment.x * GRID_SIZE + GRID_SIZE / 2;
            const centerY = segment.y * GRID_SIZE + GRID_SIZE / 2;
            
            if (dx === 1) { // 向右
                eye1X = centerX + 2; eye1Y = centerY - 3;
                eye2X = centerX + 2; eye2Y = centerY + 3;
            } else if (dx === -1) { // 向左
                eye1X = centerX - 2; eye1Y = centerY - 3;
                eye2X = centerX - 2; eye2Y = centerY + 3;
            } else if (dy === 1) { // 向下
                eye1X = centerX - 3; eye1Y = centerY + 2;
                eye2X = centerX + 3; eye2Y = centerY + 2;
            } else if (dy === -1) { // 向上
                eye1X = centerX - 3; eye1Y = centerY - 2;
                eye2X = centerX + 3; eye2Y = centerY - 2;
            } else { // 静止时默认眼睛
                eye1X = centerX - 3; eye1Y = centerY - 2;
                eye2X = centerX + 3; eye2Y = centerY - 2;
            }
            
            ctx.beginPath();
            ctx.arc(eye1X, eye1Y, eyeSize, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(eye2X, eye2Y, eyeSize, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}

// 绘制食物
function drawFood() {
    ctx.fillStyle = '#FF5722';
    ctx.fillRect(
        food.x * GRID_SIZE + 2,
        food.y * GRID_SIZE + 2,
        GRID_SIZE - 4,
        GRID_SIZE - 4
    );
    
    // 为食物添加一个小的闪光效果
    ctx.fillStyle = '#FF8A65';
    ctx.fillRect(
        food.x * GRID_SIZE + 4,
        food.y * GRID_SIZE + 4,
        GRID_SIZE - 12,
        GRID_SIZE - 12
    );
}

// 处理键盘输入
function handleKeyPress(e) {
    if (!gameRunning) return;
    
    const keyPressed = e.key;
    
    switch (keyPressed) {
        case 'ArrowUp':
            if (dy !== 1) handleDirectionChange('up');
            break;
        case 'ArrowDown':
            if (dy !== -1) handleDirectionChange('down');
            break;
        case 'ArrowLeft':
            if (dx !== 1) handleDirectionChange('left');
            break;
        case 'ArrowRight':
            if (dx !== -1) handleDirectionChange('right');
            break;
        case ' ':
            e.preventDefault();
            if (gamePaused) {
                resumeGame();
            } else {
                pauseGame();
            }
            break;
    }
}

// 处理方向改变
function handleDirectionChange(direction) {
    if (!gameRunning) return;
    
    switch (direction) {
        case 'up':
            if (dy !== 1) { dx = 0; dy = -1; }
            break;
        case 'down':
            if (dy !== -1) { dx = 0; dy = 1; }
            break;
        case 'left':
            if (dx !== 1) { dx = -1; dy = 0; }
            break;
        case 'right':
            if (dx !== -1) { dx = 1; dy = 0; }
            break;
    }
}

// 更新游戏速度
function updateSpeed() {
    const speed = parseInt(speedSlider.value);
    speedValue.textContent = speed;
    gameSpeed = 300 - (speed * 25); // 速度越高，间隔越短
    
    // 如果游戏正在运行，重新设置间隔
    if (gameRunning && !gamePaused) {
        clearInterval(gameLoop);
        gameLoop = setInterval(update, gameSpeed);
    }
}

// 更新分数显示
function updateScore() {
    scoreElement.textContent = score;
}

// 更新长度显示
function updateLength() {
    lengthElement.textContent = snake.length;
}

// 防止方向键滚动页面
document.addEventListener('keydown', function(e) {
    if(['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].indexOf(e.code) > -1) {
        e.preventDefault();
    }
}, false);

// 页面加载完成后初始化游戏
document.addEventListener('DOMContentLoaded', init); 
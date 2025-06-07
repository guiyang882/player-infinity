// 游戏变量
let canvas;
let ctx;
let gameRunning = false;
let gamePaused = false;
let gameLoop;

// 游戏对象
let paddle;
let ball;
let bricks = [];
let particles = [];

// 游戏状态
let score = 0;
let lives = 3;
let level = 1;
let currentDifficulty = 'easy';

// 游戏配置
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const PADDLE_SPEED = 8;
const BALL_SPEED_BASE = 4;

// 难度设置
const DIFFICULTY_SETTINGS = {
    easy: {
        ballSpeed: 4,
        paddleWidth: 120,
        brickRows: 4,
        lives: 5
    },
    normal: {
        ballSpeed: 6,
        paddleWidth: 100,
        brickRows: 5,
        lives: 3
    },
    hard: {
        ballSpeed: 8,
        paddleWidth: 80,
        brickRows: 6,
        lives: 2
    }
};

// DOM 元素
let scoreElement, livesElement, levelElement, bricksElement;
let gameMessage, messageTitle, messageText;

// 控制变量
let keys = {};
let mouseX = 0;
let paddleMoving = '';

// 初始化游戏
function init() {
    canvas = document.getElementById('gameCanvas');
    ctx = canvas.getContext('2d');
    
    // 获取DOM元素
    scoreElement = document.getElementById('score');
    livesElement = document.getElementById('lives');
    levelElement = document.getElementById('level');
    bricksElement = document.getElementById('bricks');
    gameMessage = document.getElementById('gameMessage');
    messageTitle = document.getElementById('messageTitle');
    messageText = document.getElementById('messageText');
    
    // 设置事件监听器
    setupEventListeners();
    
    // 初始化游戏对象
    initializeGame();
    
    // 显示开始界面
    showMessage('准备开始', '点击开始游戏按钮或按空格键开始');
    
    updateUI();
}

// 设置事件监听器
function setupEventListeners() {
    // 键盘事件
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);
    
    // 鼠标事件
    canvas.addEventListener('mousemove', handleMouseMove);
    
    // 触摸事件（移动端）
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    
    // 防止页面滚动
    document.addEventListener('keydown', function(e) {
        if(['Space', 'ArrowLeft', 'ArrowRight'].indexOf(e.code) > -1) {
            e.preventDefault();
        }
    }, false);
}

// 初始化游戏
function initializeGame() {
    const settings = DIFFICULTY_SETTINGS[currentDifficulty];
    
    // 创建挡板
    paddle = {
        x: CANVAS_WIDTH / 2 - settings.paddleWidth / 2,
        y: CANVAS_HEIGHT - 30,
        width: settings.paddleWidth,
        height: 15,
        speed: PADDLE_SPEED,
        color: '#4ECDC4'
    };
    
    // 创建小球
    ball = {
        x: CANVAS_WIDTH / 2,
        y: CANVAS_HEIGHT - 50,
        radius: 8,
        dx: 0,
        dy: 0,
        speed: settings.ballSpeed,
        color: '#FF6B6B',
        stuck: true
    };
    
    // 创建砖块
    createBricks();
    
    // 重置变量
    particles = [];
    lives = settings.lives;
}

// 创建砖块
function createBricks() {
    bricks = [];
    const settings = DIFFICULTY_SETTINGS[currentDifficulty];
    const rows = settings.brickRows;
    const cols = 10;
    const brickWidth = 70;
    const brickHeight = 20;
    const padding = 5;
    const offsetTop = 60;
    const offsetLeft = (CANVAS_WIDTH - (cols * (brickWidth + padding) - padding)) / 2;
    
    // 颜色数组
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'];
    
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            bricks.push({
                x: offsetLeft + c * (brickWidth + padding),
                y: offsetTop + r * (brickHeight + padding),
                width: brickWidth,
                height: brickHeight,
                color: colors[r % colors.length],
                visible: true,
                points: (rows - r) * 10 // 上层砖块分数更高
            });
        }
    }
}

// 开始游戏
function startGame() {
    if (gameRunning) return;
    
    gameRunning = true;
    gamePaused = false;
    hideMessage();
    
    // 如果小球卡在挡板上，给它初始速度
    if (ball.stuck) {
        const angle = (Math.random() - 0.5) * Math.PI / 3; // -30° 到 30°
        ball.dx = ball.speed * Math.sin(angle);
        ball.dy = -ball.speed * Math.cos(angle);
        ball.stuck = false;
    }
    
    updateUI();
    gameLoop = requestAnimationFrame(update);
}

// 暂停游戏
function pauseGame() {
    if (!gameRunning || gamePaused) return;
    
    gamePaused = true;
    cancelAnimationFrame(gameLoop);
    showMessage('游戏暂停', '按空格键继续游戏');
}

// 继续游戏
function resumeGame() {
    if (!gameRunning || !gamePaused) return;
    
    gamePaused = false;
    hideMessage();
    
    // 如果小球卡在挡板上，给它初始速度
    if (ball.stuck) {
        const angle = (Math.random() - 0.5) * Math.PI / 3; // -30° 到 30°
        ball.dx = ball.speed * Math.sin(angle);
        ball.dy = -ball.speed * Math.cos(angle);
        ball.stuck = false;
    }
    
    gameLoop = requestAnimationFrame(update);
}

// 重新开始游戏
function restartGame() {
    gameRunning = false;
    gamePaused = false;
    cancelAnimationFrame(gameLoop);
    
    score = 0;
    level = 1;
    initializeGame();
    showMessage('准备开始', '点击开始游戏按钮或按空格键开始');
    updateUI();
}

// 设置难度
function setDifficulty(difficulty) {
    if (gameRunning) return;
    
    currentDifficulty = difficulty;
    
    // 更新按钮状态
    document.querySelectorAll('.difficulty-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-level="${difficulty}"]`).classList.add('active');
    
    initializeGame();
    updateUI();
}

// 游戏主循环
function update() {
    if (!gameRunning || gamePaused) return;
    
    // 更新游戏对象
    updatePaddle();
    updateBall();
    updateParticles();
    
    // 检测碰撞
    checkCollisions();
    
    // 绘制游戏
    draw();
    
    // 检查游戏状态
    checkGameState();
    
    // 更新UI
    updateUI();
    
    gameLoop = requestAnimationFrame(update);
}

// 更新挡板
function updatePaddle() {
    // 键盘控制
    if (keys['ArrowLeft'] || paddleMoving === 'left') {
        paddle.x -= paddle.speed;
    }
    if (keys['ArrowRight'] || paddleMoving === 'right') {
        paddle.x += paddle.speed;
    }
    
    // 鼠标控制
    if (mouseX > 0) {
        paddle.x = mouseX - paddle.width / 2;
    }
    
    // 限制挡板在画布内
    paddle.x = Math.max(0, Math.min(CANVAS_WIDTH - paddle.width, paddle.x));
    
    // 如果小球卡在挡板上，跟随挡板移动
    if (ball.stuck) {
        ball.x = paddle.x + paddle.width / 2;
    }
}

// 更新小球
function updateBall() {
    if (ball.stuck) return;
    
    ball.x += ball.dx;
    ball.y += ball.dy;
    
    // 左右墙壁碰撞
    if (ball.x <= ball.radius || ball.x >= CANVAS_WIDTH - ball.radius) {
        ball.dx = -ball.dx;
        createParticles(ball.x, ball.y, '#45B7D1');
    }
    
    // 上墙壁碰撞
    if (ball.y <= ball.radius) {
        ball.dy = -ball.dy;
        createParticles(ball.x, ball.y, '#45B7D1');
    }
    
    // 小球落底
    if (ball.y >= CANVAS_HEIGHT + ball.radius) {
        loseLife();
    }
}

// 更新粒子
function updateParticles() {
    for (let i = particles.length - 1; i >= 0; i--) {
        const particle = particles[i];
        particle.x += particle.dx;
        particle.y += particle.dy;
        particle.dy += 0.2; // 重力
        particle.life--;
        particle.alpha = particle.life / particle.maxLife;
        
        if (particle.life <= 0) {
            particles.splice(i, 1);
        }
    }
}

// 检测碰撞
function checkCollisions() {
    // 挡板碰撞
    if (ball.y + ball.radius >= paddle.y &&
        ball.x >= paddle.x &&
        ball.x <= paddle.x + paddle.width &&
        ball.dy > 0) {
        
        // 计算碰撞点相对于挡板中心的位置
        const hitPos = (ball.x - (paddle.x + paddle.width / 2)) / (paddle.width / 2);
        const angle = hitPos * Math.PI / 3; // 最大60度角
        
        ball.dx = ball.speed * Math.sin(angle);
        ball.dy = -ball.speed * Math.cos(angle);
        
        createParticles(ball.x, ball.y, paddle.color);
    }
    
    // 砖块碰撞
    for (let i = 0; i < bricks.length; i++) {
        const brick = bricks[i];
        if (!brick.visible) continue;
        
        if (ball.x + ball.radius >= brick.x &&
            ball.x - ball.radius <= brick.x + brick.width &&
            ball.y + ball.radius >= brick.y &&
            ball.y - ball.radius <= brick.y + brick.height) {
            
            // 砖块被击中
            brick.visible = false;
            score += brick.points;
            
            // 创建粒子效果
            createParticles(brick.x + brick.width / 2, brick.y + brick.height / 2, brick.color);
            
            // 简单的反弹逻辑
            ball.dy = -ball.dy;
            
            break;
        }
    }
}

// 创建粒子效果
function createParticles(x, y, color) {
    for (let i = 0; i < 8; i++) {
        particles.push({
            x: x,
            y: y,
            dx: (Math.random() - 0.5) * 6,
            dy: (Math.random() - 0.5) * 6,
            color: color,
            life: 30,
            maxLife: 30,
            alpha: 1,
            size: Math.random() * 3 + 1
        });
    }
}

// 失去生命
function loseLife() {
    lives--;
    
    if (lives <= 0) {
        gameOver();
    } else {
        resetBall();
        showMessage('失去一条生命', `剩余生命: ${lives}，按空格键继续`);
        gamePaused = true;
    }
}

// 重置小球
function resetBall() {
    ball.x = paddle.x + paddle.width / 2;
    ball.y = CANVAS_HEIGHT - 50;
    ball.dx = 0;
    ball.dy = 0;
    ball.stuck = true;
}

// 检查游戏状态
function checkGameState() {
    const remainingBricks = bricks.filter(brick => brick.visible).length;
    
    if (remainingBricks === 0) {
        nextLevel();
    }
}

// 下一关
function nextLevel() {
    level++;
    gamePaused = true;
    
    // 增加小球速度
    ball.speed += 0.5;
    
    createBricks();
    resetBall();
    
    showMessage(`第 ${level} 关`, '按空格键开始下一关');
}

// 游戏结束
function gameOver() {
    gameRunning = false;
    gamePaused = false;
    cancelAnimationFrame(gameLoop);
    
    showMessage('游戏结束', `最终分数: ${score}`);
}

// 绘制游戏
function draw() {
    // 清空画布
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // 绘制背景网格
    drawGrid();
    
    // 绘制砖块
    drawBricks();
    
    // 绘制挡板
    drawPaddle();
    
    // 绘制小球
    drawBall();
    
    // 绘制粒子
    drawParticles();
}

// 绘制网格背景
function drawGrid() {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    
    for (let x = 0; x < CANVAS_WIDTH; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, CANVAS_HEIGHT);
        ctx.stroke();
    }
    
    for (let y = 0; y < CANVAS_HEIGHT; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(CANVAS_WIDTH, y);
        ctx.stroke();
    }
}

// 绘制砖块
function drawBricks() {
    bricks.forEach(brick => {
        if (!brick.visible) return;
        
        // 砖块主体
        ctx.fillStyle = brick.color;
        ctx.fillRect(brick.x, brick.y, brick.width, brick.height);
        
        // 砖块边框
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.strokeRect(brick.x, brick.y, brick.width, brick.height);
        
        // 砖块高光
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.fillRect(brick.x, brick.y, brick.width, 5);
    });
}

// 绘制挡板
function drawPaddle() {
    // 渐变效果
    const gradient = ctx.createLinearGradient(paddle.x, paddle.y, paddle.x, paddle.y + paddle.height);
    gradient.addColorStop(0, paddle.color);
    gradient.addColorStop(1, '#2E8B8B');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(paddle.x, paddle.y, paddle.width, paddle.height);
    
    // 挡板边框
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.strokeRect(paddle.x, paddle.y, paddle.width, paddle.height);
}

// 绘制小球
function drawBall() {
    // 小球阴影
    ctx.beginPath();
    ctx.arc(ball.x + 2, ball.y + 2, ball.radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fill();
    
    // 小球主体
    const gradient = ctx.createRadialGradient(ball.x - 3, ball.y - 3, 0, ball.x, ball.y, ball.radius);
    gradient.addColorStop(0, '#FF9999');
    gradient.addColorStop(1, ball.color);
    
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // 小球高光
    ctx.beginPath();
    ctx.arc(ball.x - 2, ball.y - 2, 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.fill();
}

// 绘制粒子
function drawParticles() {
    particles.forEach(particle => {
        ctx.save();
        ctx.globalAlpha = particle.alpha;
        ctx.fillStyle = particle.color;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    });
}

// 显示消息
function showMessage(title, text) {
    messageTitle.textContent = title;
    messageText.textContent = text;
    gameMessage.classList.remove('hidden');
    
    // 更新按钮状态和文本
    updateMessageButtons();
}

// 隐藏消息
function hideMessage() {
    gameMessage.classList.add('hidden');
}

// 更新UI
function updateUI() {
    scoreElement.textContent = score;
    livesElement.textContent = lives;
    levelElement.textContent = level;
    bricksElement.textContent = bricks.filter(brick => brick.visible).length;
}

// 更新消息按钮状态和文本
function updateMessageButtons() {
    const startBtn = document.getElementById('startBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const resumeBtn = document.getElementById('resumeBtn');
    const restartBtn = document.getElementById('restartBtn');
    
    // 隐藏所有按钮
    startBtn.style.display = 'none';
    pauseBtn.style.display = 'none';
    resumeBtn.style.display = 'none';
    restartBtn.style.display = 'none';
    
    if (!gameRunning) {
        // 游戏未开始状态
        startBtn.style.display = 'inline-block';
        startBtn.textContent = '开始游戏';
        startBtn.onclick = startGame;
    } else if (gamePaused) {
        // 游戏暂停状态（包括失去生命）
        resumeBtn.style.display = 'inline-block';
        resumeBtn.textContent = '继续游戏';
        resumeBtn.onclick = resumeGame;
        restartBtn.style.display = 'inline-block';
    } else {
        // 游戏运行状态
        pauseBtn.style.display = 'inline-block';
        restartBtn.style.display = 'inline-block';
    }
}

// 事件处理函数
function handleKeyDown(e) {
    keys[e.code] = true;
    
    if (e.code === 'Space') {
        e.preventDefault();
        if (!gameRunning) {
            startGame();
        } else if (gamePaused) {
            resumeGame();
        } else {
            pauseGame();
        }
    }
}

function handleKeyUp(e) {
    keys[e.code] = false;
}

function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
}

function handleTouchMove(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    mouseX = touch.clientX - rect.left;
}

// 移动端控制
function startMoving(direction) {
    paddleMoving = direction;
}

function stopMoving() {
    paddleMoving = '';
}

// 页面加载完成后初始化游戏
document.addEventListener('DOMContentLoaded', init); 
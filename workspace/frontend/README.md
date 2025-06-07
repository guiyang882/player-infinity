# 游戏中心 Frontend

这是一个多游戏的前端项目，采用模块化的目录结构，便于添加和管理新游戏。

## 📁 目录结构

```
frontend/
├── index.html          # 游戏中心主页
├── main.css           # 主页样式
├── README.md          # 项目说明文档
├── 2048/              # 2048 数字游戏
│   ├── index.html     # 游戏页面
│   ├── styles.css     # 游戏样式
│   └── game.js        # 游戏逻辑
└── snake/             # 贪吃蛇游戏
    ├── index.html     # 游戏页面
    ├── styles.css     # 游戏样式
    └── game.js        # 游戏逻辑
```

## 🎮 现有游戏

### 1. 2048
- **路径**: `/2048/index.html`
- **特性**: 数字合并益智游戏，支持AI辅助和多种算法
- **技术**: HTML5 + CSS3 + JavaScript

### 2. 贪吃蛇
- **路径**: `/snake/index.html`
- **特性**: 经典街机游戏，支持键盘和触屏控制
- **技术**: HTML5 Canvas + CSS3 + JavaScript

## ➕ 添加新游戏

要添加新游戏，请按照以下步骤：

### 1. 创建游戏目录
```bash
mkdir frontend/新游戏名称
```

### 2. 创建标准文件结构
在游戏目录中创建以下文件：
- `index.html` - 游戏页面
- `styles.css` - 游戏样式
- `game.js` - 游戏逻辑

### 3. HTML 模板结构
```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>游戏名称</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="back-to-home">
        <a href="../index.html">← 返回游戏中心</a>
    </div>
    
    <div class="container">
        <!-- 游戏内容 -->
    </div>

    <script src="game.js"></script>
</body>
</html>
```

### 4. 更新主页游戏列表
在 `index.html` 的 `.games-grid` 中添加新游戏卡片：

```html
<div class="game-card" onclick="window.location.href='新游戏名称/index.html'">
    <div class="game-icon">🎮</div>
    <h3>游戏名称</h3>
    <p>游戏描述</p>
    <div class="game-features">
        <span class="feature">特性1</span>
        <span class="feature">特性2</span>
    </div>
</div>
```

## 🎨 样式规范

### 返回按钮样式
每个游戏都应该包含返回游戏中心的按钮，使用以下CSS类：

```css
.back-to-home {
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 1000;
}

.back-to-home a {
    display: inline-block;
    padding: 10px 15px;
    background-color: #主题色;
    color: white;
    text-decoration: none;
    border-radius: 5px;
    font-size: 14px;
    transition: background-color 0.2s;
}
```

### 响应式设计
确保游戏在各种设备上都能正常显示：
- 桌面端（>768px）
- 平板端（768px-480px）
- 手机端（<480px）

## 🚀 启动项目

在项目根目录运行：
```bash
cd workspace && python3 -m http.server 8000
```

然后访问：`http://localhost:8000/frontend/`

## 📝 开发建议

1. **代码规范**: 使用一致的代码风格和命名规范
2. **注释**: 为关键函数和复杂逻辑添加注释
3. **错误处理**: 添加适当的错误处理机制
4. **性能优化**: 注意游戏循环的性能，避免内存泄漏
5. **用户体验**: 提供清晰的操作说明和反馈

## 🔧 技术栈

- **HTML5**: 语义化标签，Canvas API
- **CSS3**: Flexbox/Grid布局，动画效果，响应式设计
- **JavaScript ES6+**: 模块化编程，现代语法
- **本地存储**: localStorage 保存游戏数据 
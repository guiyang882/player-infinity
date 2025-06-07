# 🎮 Player Infinity - 游戏中心

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![HTML5](https://img.shields.io/badge/HTML5-Canvas-orange.svg)](#游戲特性)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg)](#技術棧)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个现代化的Web游戏中心，集成了多款经典游戏。采用前后端分离架构，包含AI算法、现代UI设计和跨平台兼容性。

## ✨ 游戏特性

### 🧩 2048 - AI智能版
- 🤖 **多种AI算法** - Expectimax、贪心、角落策略、随机算法
- 📊 **算法性能对比** - 内置算法验证和性能测试工具
- 🎯 **自动游戏模式** - AI自动运行或手动辅助
- 🔧 **RESTful API** - 完整的后端AI服务

### 🐍 Snake - 经典贪吃蛇
- 🎮 **流畅控制** - 键盘方向键 + 触摸控制
- 🏆 **分数系统** - 实时分数 + 历史最高分记录
- ⚡ **速度设置** - 多档速度调节
- 📱 **移动适配** - 完整的触摸屏支持

### 🎯 Breakout - 弹珠台
- 🎲 **物理引擎** - 真实的碰撞检测和球体运动
- 🎨 **粒子效果** - 华丽的视觉反馈
- 🏅 **难度系统** - 简单/普通/困难三档难度
- 🎮 **多控制方式** - 鼠标、键盘、触摸三合一

## 🏗️ 项目结构

```
player-infinity/
├── 📁 workspace/                    # 主工作区
│   ├── 📄 index.html               # 游戏中心入口
│   ├── 📁 frontend/                # 前端游戏代码
│   │   ├── 📄 index.html           # 游戏选择页面
│   │   ├── 📄 main.css             # 主页样式
│   │   ├── 📁 2048/               # 2048游戏
│   │   │   ├── 📄 index.html      # 游戏界面
│   │   │   ├── 📄 styles.css      # 游戏样式
│   │   │   └── 📄 game.js         # 游戏逻辑 (28KB)
│   │   ├── 📁 snake/              # 贪吃蛇游戏
│   │   │   ├── 📄 index.html      # 游戏界面
│   │   │   ├── 📄 styles.css      # 游戏样式
│   │   │   └── 📄 game.js         # 游戏逻辑 (10KB)
│   │   ├── 📁 breakout/           # 弹珠台游戏
│   │   │   ├── 📄 index.html      # 游戏界面
│   │   │   ├── 📄 styles.css      # 游戏样式 (7KB)
│   │   │   └── 📄 game.js         # 游戏逻辑 (16KB)
│   │   └── 📄 README.md           # 前端开发文档
│   ├── 📁 backend/                 # 后端服务
│   │   ├── 🌐 server.py           # Flask服务器 (6.8KB)
│   │   ├── 🎯 game_2048.py        # 2048游戏逻辑
│   │   ├── 🤖 ai_solver.py        # AI算法实现
│   │   ├── 📊 algorithm_validator.py # 算法性能验证
│   │   ├── 📁 algorithm/          # AI算法模块
│   │   └── 📄 requirements.txt    # Python依赖
│   └── 🚀 start_server.sh         # 快速启动脚本
├── 📖 README.md                    # 项目说明 (当前文件)
├── 📝 LICENSE                      # 开源协议
└── 📄 .gitignore                   # Git忽略文件
```

## 🚀 快速开始

### 方式一：使用启动脚本（推荐）

```bash
# 克隆项目
git clone https://github.com/yourusername/player-infinity.git
cd player-infinity

# 运行启动脚本
cd workspace
./start_server.sh
```

### 方式二：手动启动

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/player-infinity.git
cd player-infinity/workspace

# 2. 安装Python依赖
cd backend
pip install -r requirements.txt

# 3. 启动服务器
python server.py
```

### 方式三：直接访问（无AI功能）

直接在浏览器中打开 `workspace/index.html` 或 `workspace/frontend/index.html`

### 访问游戏

启动成功后，在浏览器中访问：
```
http://localhost:5000          # 游戏中心入口
http://localhost:5000/2048     # 直接访问2048（含AI）
http://localhost:5000/snake    # 直接访问贪吃蛇
http://localhost:5000/breakout # 直接访问弹珠台
```

## 🎮 游戏操作

### 🧩 2048游戏
| 操作 | 说明 |
|------|------|
| ⬅️⬆️⬇️➡️ | 方向键移动数字方块 |
| 🤖 **AI 走一步** | AI智能提示下一步 |
| 🔄 **自动游戏** | 开启AI全自动模式 |
| ⏹️ **停止自动** | 停止AI自动游戏 |
| 🔄 **重新开始** | 重置游戏状态 |

### 🐍 贪吃蛇游戏
| 操作 | 说明 |
|------|------|
| ⬅️⬆️⬇️➡️ | 方向键控制蛇的移动 |
| 📱 **触摸滑动** | 移动端滑动控制 |
| ⚡ **速度按钮** | 调节游戏速度 |
| 🔄 **重新开始** | 重置游戏和分数 |

### 🎯 弹珠台游戏
| 操作 | 说明 |
|------|------|
| 🖱️ **鼠标移动** | 控制挡板左右移动 |
| ⬅️➡️ **方向键** | 键盘控制挡板 |
| 📱 **触摸滑动** | 移动端触摸控制 |
| ⏸️ **空格键** | 暂停/继续游戏 |
| 🎚️ **难度按钮** | 简单/普通/困难 |

## 🤖 AI算法详解 (2048专用)

### 1. Expectimax算法 ⭐⭐⭐⭐⭐
**核心推荐算法**，处理游戏随机性的最优解：
- 多层深度搜索（4层）
- 期望值计算处理随机方块
- 综合启发式评估函数
- **典型表现**：平均分数 25,000+，100%达成2048

### 2. 贪心算法 ⭐⭐⭐
选择即时最高分数的移动，快速但缺乏规划。

### 3. 角落策略 ⭐⭐
优先将大数字推向角落的传统策略。

### 4. 随机算法 ⭐
随机选择，作为算法性能基准。

## 🔌 API接口文档

### 2048 AI服务

```http
POST /api/ai_move
Content-Type: application/json

{
  "grid": [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
  "score": 0
}
```

### 指定AI算法类型
```http
POST /api/ai_move/{ai_type}
```
支持：`expectimax` | `greedy` | `corner` | `random`

### 游戏状态API
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/game/new` | POST | 创建新游戏 |
| `/api/game/move` | POST | 执行移动 |
| `/api/evaluate` | POST | 评估当前状态 |

## 💻 技术栈

### 前端技术
- **HTML5 Canvas** - 游戏图形渲染
- **原生JavaScript (ES6+)** - 游戏逻辑实现
- **CSS3** - 现代UI设计和动画
- **响应式设计** - 跨设备兼容

### 后端技术
- **Python 3.8+** - 主要开发语言
- **Flask 2.3+** - Web框架
- **RESTful API** - 前后端通信

### 游戏引擎特性
- **Canvas 2D API** - 高性能图形渲染
- **事件驱动架构** - 流畅的用户交互
- **物理引擎** - 真实的碰撞检测
- **粒子系统** - 华丽的视觉效果

## 📊 性能测试

运行AI算法性能测试（仅限2048）：

```bash
cd workspace/backend

# 快速测试
python algorithm_validator.py

# 详细测试
python test_ai.py
```

### AI算法性能对比

| 算法 | 平均分数 | 平均最大方块 | 2048达成率 | 推荐度 |
|------|----------|--------------|------------|--------|
| Expectimax | 25,472 | 2048+ | 100% | ⭐⭐⭐⭐⭐ |
| Greedy | 4,052 | 341 | 12% | ⭐⭐⭐ |
| Corner | 1,467 | 107 | 3% | ⭐⭐ |
| Random | 940 | 85 | 0% | ⭐ |

## 🎯 游戏特色

### 🧩 2048特色
- **AI辅助模式**：可选择让AI提示或全自动运行
- **算法可视化**：观察不同AI的决策过程
- **性能统计**：详细的算法表现数据

### 🐍 Snake特色
- **智能避撞**：改进的碰撞检测算法
- **视觉反馈**：蛇头方向指示、食物动画
- **分数系统**：本地存储最高分记录

### 🎯 Breakout特色
- **物理引擎**：真实的球体运动和碰撞
- **粒子效果**：击中砖块时的华丽特效
- **关卡系统**：无限关卡，难度递增
- **多控制方式**：支持各种输入设备

## 🤝 贡献指南

欢迎贡献代码！您可以：

### 新功能开发
- 添加新的游戏
- 实现更多AI算法
- 改进现有游戏机制

### 优化改进
- 性能优化
- UI/UX改进
- 移动端适配增强

### 提交规范
1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 🎯 开发路线图

### 近期计划
- [ ] 添加更多经典游戏（俄罗斯方块、扫雷等）
- [ ] 实现用户账户系统
- [ ] 添加在线排行榜
- [ ] 改进移动端体验

### 长期规划
- [ ] 多人游戏支持
- [ ] 神经网络AI算法
- [ ] 游戏录像回放功能
- [ ] 自定义游戏难度

## 📄 开源协议

本项目基于 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

- 感谢所有贡献者的努力
- 感谢开源社区的支持
- 特别感谢经典游戏的创作者们

---

⭐ **如果这个项目对您有帮助，请给我们一个星标！**

🎮 **开始游戏：** [http://localhost:5000](http://localhost:5000)
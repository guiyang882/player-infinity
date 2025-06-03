# 🎮 2048 AI Game - 启发式搜索算法实现

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AI](https://img.shields.io/badge/AI-Expectimax-red.svg)](#ai算法)

一个完整的2048游戏实现，集成了多种AI启发式搜索算法。项目采用前后端分离架构，前端使用原生HTML/CSS/JavaScript，后端使用Python Flask提供AI算法服务。

![2048 Game Demo](https://via.placeholder.com/600x400/f0f0f0/333333?text=2048+AI+Game+Demo)

## ✨ 主要特性

- 🎯 **完整的2048游戏实现** - 支持所有标准游戏规则
- 🤖 **多种AI算法** - Expectimax、贪心、角落策略、随机算法
- 🌐 **前后端分离** - RESTful API架构
- 📊 **算法性能对比** - 内置算法验证和性能测试工具
- 🎨 **现代UI设计** - 响应式界面，流畅动画效果
- 🔧 **易于扩展** - 模块化设计，便于添加新算法

## 🏗️ 项目结构

```
2048-ai-game/
├── 📁 frontend/                 # 前端代码
│   ├── 📄 index.html           # 游戏界面
│   └── 📄 README.md            # 前端说明
├── 📁 backend/                  # 后端代码
│   ├── 🎯 game_2048.py         # 游戏核心逻辑
│   ├── 🤖 ai_solver.py         # AI算法实现
│   ├── 🌐 server.py            # Flask API服务器
│   ├── 🧪 test_ai.py           # AI算法测试
│   ├── 📊 algorithm_validator.py # 算法性能验证
│   ├── 📄 requirements.txt     # Python依赖
│   └── 📄 README.md            # 后端说明
├── 🚀 start_server.sh          # 启动脚本
├── 📋 .gitignore               # Git忽略文件
├── 📖 README.md                # 项目说明 (当前文件)
└── 📝 instruction.md           # 开发指令
```

## 🚀 快速开始

### 方式一：使用启动脚本（推荐）

```bash
# 克隆项目
git clone https://github.com/yourusername/2048-ai-game.git
cd 2048-ai-game

# 运行启动脚本
./start_server.sh
```

### 方式二：手动启动

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/2048-ai-game.git
cd 2048-ai-game

# 2. 安装Python依赖
cd backend
pip install -r requirements.txt

# 3. 启动服务器
python3 server.py
```

### 访问游戏

启动成功后，在浏览器中访问：
```
http://localhost:5000
```

## 🎮 游戏操作

| 操作 | 说明 |
|------|------|
| ⬅️⬆️⬇️➡️ | 使用方向键移动数字方块 |
| 🤖 **AI 走一步** | 让AI帮你走一步 |
| 🔄 **自动游戏** | 开启AI自动游戏模式 |
| ⏹️ **停止自动** | 停止AI自动游戏 |
| 🔄 **重新开始** | 重新开始游戏 |

## 🤖 AI算法详解

### 1. Expectimax算法 ⭐

**核心算法**，使用期望最大化处理随机性：

- **算法特点**：
  - 多层深度搜索（默认4层）
  - 期望值计算处理随机方块
  - 综合启发式评估函数

- **评估指标**：
  - 空格数量奖励 (权重: 2.7)
  - 最大数字奖励 (权重: 1.0)
  - 单调性评估 (权重: 1.0)
  - 平滑性评估 (权重: 0.1)
  - 角落位置奖励 (权重: 1.5)

### 2. 贪心算法

选择能获得最高即时分数的移动，简单高效但缺乏长远规划。

### 3. 角落策略

优先将大数字推向角落，基于"大数字应该在角落"的策略。

### 4. 随机算法

随机选择有效移动，作为对照组评估其他算法的性能。

## 📊 性能测试

运行算法性能测试：

```bash
cd backend

# 快速测试（3局游戏）
python3 test_ai.py

# 详细验证（可调整游戏数量）
python3 algorithm_validator.py
```

### 典型测试结果

| 算法 | 平均分数 | 平均最大方块 | 胜率 | 性能 |
|------|----------|--------------|------|------|
| Expectimax | 25,472 | 2048 | 100% | ⭐⭐⭐⭐⭐ |
| Greedy | 4,052 | 341 | 0% | ⭐⭐⭐ |
| Corner | 1,467 | 107 | 0% | ⭐⭐ |
| Random | 940 | 85 | 0% | ⭐ |

## 🔌 API文档

### 获取AI建议

```http
POST /api/ai_move
Content-Type: application/json

{
  "grid": [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
  "score": 0
}
```

### 指定AI类型

```http
POST /api/ai_move/{ai_type}
```

支持的AI类型：`expectimax` | `greedy` | `corner` | `random`

### 游戏操作

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/game/new` | POST | 创建新游戏 |
| `/api/game/move` | POST | 执行移动 |
| `/api/evaluate` | POST | 评估状态 |

## 🛠️ 技术栈

### 前端
- **HTML5** - 页面结构
- **CSS3** - 样式设计，动画效果
- **JavaScript** - 游戏逻辑，API交互

### 后端
- **Python 3.8+** - 主要开发语言
- **Flask 2.3+** - Web框架
- **NumPy** - 数值计算
- **Matplotlib** - 数据可视化（可选）

### 开发工具
- **Flask-CORS** - 跨域支持
- **Git** - 版本控制

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. **Fork** 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 **Pull Request**

### 开发建议

- 添加新的AI算法
- 改进现有算法性能
- 增强前端用户体验
- 完善测试覆盖率
- 优化代码文档

## 📝 开发日志

- **v1.0** - 基础游戏实现
- **v1.1** - 添加Expectimax算法
- **v1.2** - 前后端分离架构
- **v1.3** - 多算法支持和性能测试

## 🎯 未来计划

- [ ] 添加蒙特卡洛树搜索算法
- [ ] 实现神经网络AI
- [ ] 添加游戏回放功能
- [ ] 支持不同网格大小
- [ ] 添加在线排行榜
- [ ] 移动端适配

## 📄 许可证

本项目使用 [MIT 许可证](LICENSE)

## 🙏 致谢

- 感谢 [2048 Game](https://github.com/gabrielecirulli/2048) 提供的游戏灵感
- 感谢开源社区的算法实现参考

## 📞 联系方式

- **项目地址**: [GitHub Repository](https://github.com/yourusername/2048-ai-game)
- **问题反馈**: [Issues](https://github.com/yourusername/2048-ai-game/issues)
- **讨论交流**: [Discussions](https://github.com/yourusername/2048-ai-game/discussions)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！ 
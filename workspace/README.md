# 2048游戏 - AI启发式搜索算法

这是一个完整的2048游戏实现，包含HTML前端界面和Python后端AI算法。项目采用前后端分离的架构。

## 项目结构

```
workspace/
├── frontend/               # 前端代码
│   └── index.html         # HTML游戏界面
├── backend/               # 后端代码
│   ├── game_2048.py      # 游戏核心逻辑类
│   ├── ai_solver.py      # AI算法实现
│   ├── server.py         # Flask服务器
│   ├── test_ai.py        # AI算法测试脚本
│   ├── algorithm_validator.py  # 算法验证器
│   └── requirements.txt  # Python依赖
├── README.md             # 项目说明
└── instruction.md        # 任务指令
```

## 安装和运行

1. 进入后端目录并安装依赖：
```bash
cd backend
pip install -r requirements.txt
```

2. 启动后端服务器：
```bash
python server.py
```

3. 访问游戏：打开浏览器访问 http://localhost:5000

4. 测试AI算法：
```bash
python test_ai.py
```

## 技术架构

### 前端 (Frontend)
- **技术栈**: HTML5, CSS3, JavaScript
- **文件**: `frontend/index.html`
- **功能**: 
  - 游戏界面渲染
  - 用户交互处理
  - 与后端API通信
  - AI自动游戏控制

### 后端 (Backend)
- **技术栈**: Python 3, Flask
- **主要模块**:
  - `game_2048.py` - 游戏核心逻辑
  - `ai_solver.py` - AI算法实现
  - `server.py` - RESTful API服务器
  - `test_ai.py` - 算法测试工具
  - `algorithm_validator.py` - 算法性能验证

## AI算法

1. **Expectimax算法** - 主要的启发式搜索算法
2. **贪心算法** - 选择即时收益最大的移动
3. **角落策略** - 优先将大数字推向角落
4. **随机算法** - 随机选择有效移动

## 使用说明

- 使用方向键移动数字方块
- 点击"AI 走一步"让AI帮你走一步
- 点击"自动游戏"开始AI自动游戏
- 相同数字的方块相碰时会合并
- 目标是创造出2048方块 
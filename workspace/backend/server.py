from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from game_2048 import Game2048
from ai_solver import AI2048Solver, GreedyAI, RandomAI, CornerAI

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局AI实例
ai_solver = AI2048Solver(max_depth=4)
greedy_ai = GreedyAI()
random_ai = RandomAI()
corner_ai = CornerAI()

@app.route('/')
def index():
    """提供HTML页面"""
    return send_file('../frontend/index.html')

@app.route('/styles.css')
def styles():
    """提供CSS文件"""
    return send_file('../frontend/styles.css', mimetype='text/css')

@app.route('/game.js')
def game_js():
    """提供JavaScript文件"""
    return send_file('../frontend/game.js', mimetype='application/javascript')

@app.route('/static/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory('../frontend', filename)

@app.route('/test')
def test_page():
    """提供测试页面"""
    return send_file('../simple_test.html')

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """获取AI建议的移动方向"""
    try:
        data = request.get_json()
        grid = data.get('grid')
        score = data.get('score', 0)
        
        if not grid:
            return jsonify({'error': '缺少游戏状态'}), 400
        
        # 创建游戏实例并设置状态
        game = Game2048()
        game.set_grid(grid, score)
        
        # 获取AI建议
        direction = ai_solver.get_best_move(game)
        
        # 转换方向格式
        direction_map = {
            'up': 'ArrowUp',
            'down': 'ArrowDown',
            'left': 'ArrowLeft',
            'right': 'ArrowRight'
        }
        
        frontend_direction = direction_map.get(direction) if direction else None
        
        return jsonify({
            'direction': frontend_direction,
            'ai_type': 'expectimax'
        })
        
    except Exception as e:
        print(f"AI move error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_move/<ai_type>', methods=['POST'])
def ai_move_by_type(ai_type):
    """根据指定AI类型获取移动建议"""
    try:
        data = request.get_json()
        grid = data.get('grid')
        score = data.get('score', 0)
        
        if not grid:
            return jsonify({'error': '缺少游戏状态'}), 400
        
        # 创建游戏实例并设置状态
        game = Game2048()
        game.set_grid(grid, score)
        
        # 选择AI类型
        direction = None
        if ai_type == 'expectimax':
            direction = ai_solver.get_best_move(game)
        elif ai_type == 'greedy':
            direction = greedy_ai.get_best_move(game)
        elif ai_type == 'random':
            direction = random_ai.get_best_move(game)
        elif ai_type == 'corner':
            direction = corner_ai.get_best_move(game)
        else:
            return jsonify({'error': f'未知的AI类型: {ai_type}'}), 400
        
        # 转换方向格式
        direction_map = {
            'up': 'ArrowUp',
            'down': 'ArrowDown',
            'left': 'ArrowLeft',
            'right': 'ArrowRight'
        }
        
        frontend_direction = direction_map.get(direction) if direction else None
        
        return jsonify({
            'direction': frontend_direction,
            'ai_type': ai_type
        })
        
    except Exception as e:
        print(f"AI move error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_state():
    """评估当前游戏状态"""
    try:
        data = request.get_json()
        grid = data.get('grid')
        score = data.get('score', 0)
        
        if not grid:
            return jsonify({'error': '缺少游戏状态'}), 400
        
        # 创建游戏实例并设置状态
        game = Game2048()
        game.set_grid(grid, score)
        
        # 评估状态
        evaluation_score = ai_solver.evaluate_state(game)
        
        # 获取各项指标
        empty_cells = len(game.get_empty_cells())
        max_tile = game.get_max_tile()
        monotonicity = ai_solver.calculate_monotonicity(game.grid)
        smoothness = ai_solver.calculate_smoothness(game.grid)
        corner_bonus = ai_solver.calculate_corner_bonus(game.grid)
        
        return jsonify({
            'evaluation_score': evaluation_score,
            'metrics': {
                'empty_cells': empty_cells,
                'max_tile': max_tile,
                'monotonicity': monotonicity,
                'smoothness': smoothness,
                'corner_bonus': corner_bonus,
                'game_score': score
            }
        })
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/game/new', methods=['POST'])
def new_game():
    """创建新游戏"""
    try:
        game = Game2048()
        return jsonify({
            'grid': game.grid,
            'score': game.score,
            'game_over': game.game_over,
            'game_won': game.game_won
        })
    except Exception as e:
        print(f"New game error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/game/move', methods=['POST'])
def make_move():
    """执行移动"""
    try:
        data = request.get_json()
        grid = data.get('grid')
        score = data.get('score', 0)
        direction = data.get('direction')
        
        if not grid or not direction:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 转换方向格式
        direction_map = {
            'ArrowUp': 'up',
            'ArrowDown': 'down',
            'ArrowLeft': 'left',
            'ArrowRight': 'right'
        }
        
        backend_direction = direction_map.get(direction)
        if not backend_direction:
            return jsonify({'error': f'无效的方向: {direction}'}), 400
        
        # 创建游戏实例并执行移动
        game = Game2048()
        game.set_grid(grid, score)
        
        moved = game.move(backend_direction)
        
        return jsonify({
            'grid': game.grid,
            'score': game.score,
            'game_over': game.game_over,
            'game_won': game.game_won,
            'moved': moved
        })
        
    except Exception as e:
        print(f"Move error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting 2048 Game Server...")
    print("访问 http://127.0.0.1:5000 来玩游戏")
    print("API端点:")
    print("  POST /api/ai_move - 获取AI建议")
    print("  POST /api/ai_move/<ai_type> - 获取指定AI类型的建议")
    print("  POST /api/evaluate - 评估游戏状态")
    print("  POST /api/game/new - 创建新游戏")
    print("  POST /api/game/move - 执行移动")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

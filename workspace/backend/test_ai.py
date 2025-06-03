#!/usr/bin/env python3
"""简单的AI测试脚本"""

from game_2048 import Game2048
from ai_solver import AI2048Solver, GreedyAI, RandomAI, CornerAI

def test_ai(ai, ai_name, num_games=3):
    """测试AI算法"""
    print(f"\n测试 {ai_name} AI...")
    
    scores = []
    max_tiles = []
    wins = 0
    
    for i in range(num_games):
        print(f"  第 {i+1} 局游戏...")
        game = Game2048()
        moves = 0
        
        while not game.game_over and moves < 2000:
            direction = ai.get_best_move(game)
            if not direction:
                break
            
            if not game.move(direction):
                break
            
            moves += 1
        
        scores.append(game.score)
        max_tiles.append(game.get_max_tile())
        if game.game_won:
            wins += 1
        
        print(f"    分数: {game.score}, 最大方块: {game.get_max_tile()}, 移动次数: {moves}")
    
    avg_score = sum(scores) / len(scores)
    avg_max_tile = sum(max_tiles) / len(max_tiles)
    win_rate = wins / num_games * 100
    
    print(f"  结果总结:")
    print(f"    平均分数: {avg_score:.0f}")
    print(f"    平均最大方块: {avg_max_tile:.0f}")
    print(f"    胜率: {win_rate:.1f}%")
    
    return {
        'avg_score': avg_score,
        'avg_max_tile': avg_max_tile,
        'win_rate': win_rate
    }

def main():
    """主函数"""
    print("2048 AI算法测试")
    print("="*50)
    
    # 创建AI实例
    ais = {
        'Expectimax': AI2048Solver(max_depth=3),
        'Greedy': GreedyAI(),
        'Corner': CornerAI(),
        'Random': RandomAI()
    }
    
    results = {}
    
    # 测试每个AI
    for name, ai in ais.items():
        results[name] = test_ai(ai, name, num_games=3)
    
    # 显示对比结果
    print("\n" + "="*50)
    print("算法对比结果:")
    print("="*50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        print(f"{i}. {name}")
        print(f"   平均分数: {result['avg_score']:.0f}")
        print(f"   平均最大方块: {result['avg_max_tile']:.0f}")
        print(f"   胜率: {result['win_rate']:.1f}%")
        print()

if __name__ == '__main__':
    main() 
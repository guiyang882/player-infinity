import time
import statistics
from typing import List, Dict
from game_2048 import Game2048
from ai_solver import AI2048Solver, GreedyAI, RandomAI, CornerAI

class AlgorithmValidator:
    """算法验证器：测试和比较不同AI算法的表现"""
    
    def __init__(self):
        self.algorithms = {
            'expectimax': AI2048Solver(max_depth=4),
            'greedy': GreedyAI(),
            'random': RandomAI(),
            'corner': CornerAI()
        }
    
    def run_single_game(self, ai_name: str, max_moves: int = 5000) -> Dict:
        """运行单次游戏并记录结果"""
        ai = self.algorithms[ai_name]
        game = Game2048()
        
        moves = 0
        start_time = time.time()
        
        while not game.game_over and moves < max_moves:
            direction = ai.get_best_move(game)
            if direction is None:
                break
            
            moved = game.move(direction)
            if not moved:
                break
            
            moves += 1
        
        end_time = time.time()
        
        return {
            'ai_name': ai_name,
            'final_score': game.score,
            'max_tile': game.get_max_tile(),
            'moves': moves,
            'game_won': game.game_won,
            'time_taken': end_time - start_time
        }
    
    def run_benchmark(self, ai_names: List[str] = None, num_games: int = 5) -> Dict:
        """运行基准测试，比较多个算法的性能"""
        if ai_names is None:
            ai_names = list(self.algorithms.keys())
        
        results = {}
        
        for ai_name in ai_names:
            print(f"\nTesting {ai_name}...")
            game_results = []
            
            for game_num in range(num_games):
                print(f"  Game {game_num + 1}/{num_games}")
                result = self.run_single_game(ai_name)
                game_results.append(result)
                print(f"    Score: {result['final_score']}, Max tile: {result['max_tile']}")
            
            # 计算统计数据
            scores = [r['final_score'] for r in game_results]
            max_tiles = [r['max_tile'] for r in game_results]
            moves = [r['moves'] for r in game_results]
            win_rate = sum(1 for r in game_results if r['game_won']) / num_games
            avg_time = statistics.mean([r['time_taken'] for r in game_results])
            
            results[ai_name] = {
                'games': game_results,
                'stats': {
                    'avg_score': statistics.mean(scores),
                    'max_score': max(scores),
                    'avg_max_tile': statistics.mean(max_tiles),
                    'max_max_tile': max(max_tiles),
                    'avg_moves': statistics.mean(moves),
                    'win_rate': win_rate,
                    'avg_time': avg_time
                }
            }
        
        return results
    
    def print_results(self, results: Dict):
        """打印测试结果"""
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON RESULTS")
        print("="*60)
        
        # 按平均分数排序
        sorted_algos = sorted(results.items(), 
                            key=lambda x: x[1]['stats']['avg_score'], 
                            reverse=True)
        
        for rank, (ai_name, data) in enumerate(sorted_algos, 1):
            stats = data['stats']
            print(f"\n{rank}. {ai_name.upper()}")
            print(f"   Average Score: {stats['avg_score']:.0f}")
            print(f"   Max Score: {stats['max_score']}")
            print(f"   Average Max Tile: {stats['avg_max_tile']:.0f}")
            print(f"   Highest Tile: {stats['max_max_tile']}")
            print(f"   Win Rate: {stats['win_rate']:.1%}")
            print(f"   Average Moves: {stats['avg_moves']:.0f}")
            print(f"   Average Time: {stats['avg_time']:.2f}s")


def main():
    """主函数：运行算法验证"""
    validator = AlgorithmValidator()
    
    print("Starting Algorithm Comparison...")
    results = validator.run_benchmark(num_games=3)
    validator.print_results(results)

if __name__ == '__main__':
    main() 
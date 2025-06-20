#!/usr/bin/env python3
"""
强化学习AI训练脚本
用于训练DQN模型玩2048游戏
"""

import sys
import os
import argparse
import time

# 添加父目录到path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithm.dqn.rl_ai import RLAI
from game_2048 import Game2048


def train_rl_model(episodes=1000, model_path=None, resume=False):
    """训练强化学习模型"""
    print("=" * 50)
    print("2048 强化学习AI训练程序")
    print("=" * 50)
    
    # 检查PyTorch
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
    except ImportError:
        print("错误: 需要安装PyTorch才能训练RL模型")
        print("请运行: pip install torch numpy")
        return
    
    # 创建AI实例
    ai = RLAI(model_path=model_path, training_mode=True)
    
    if ai.use_fallback:
        print("错误: PyTorch不可用，无法训练模型")
        return
    
    print(f"模型路径: {ai.model_path}")
    print(f"训练回合数: {episodes}")
    
    if resume and os.path.exists(ai.model_path):
        print("从现有模型继续训练...")
    else:
        print("开始全新训练...")
    
    # 开始训练
    start_time = time.time()
    
    try:
        ai.train_episodes(episodes)
        
        # 保存最终模型
        ai.save_model()
        
        # 保存最终检查点
        ai.save_checkpoint(episodes, "final")
        
        training_time = time.time() - start_time
        print(f"\n训练完成! 用时: {training_time/60:.2f} 分钟")
        
        # 测试训练后的模型
        print("\n开始测试训练后的模型...")
        test_model(ai, num_games=10)
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("保存当前模型...")
        ai.save_model()
        ai.save_checkpoint(ai.episode, "interrupted")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        ai.save_model()
        ai.save_checkpoint(ai.episode, "error")


def test_model(ai, num_games=10):
    """测试训练后的模型"""
    ai.training_mode = False  # 关闭训练模式
    
    scores = []
    max_tiles = []
    
    print(f"测试 {num_games} 局游戏...")
    
    for i in range(num_games):
        game = Game2048()
        moves = 0
        
        while not game.game_over and moves < 1000:
            move = ai.get_best_move(game)
            if move is None:
                break
            game.move(move)
            moves += 1
        
        scores.append(game.score)
        max_tiles.append(game.get_max_tile())
        
        print(f"游戏 {i+1}: 分数={game.score}, 最大方块={game.get_max_tile()}, 移动次数={moves}")
    
    avg_score = sum(scores) / len(scores)
    avg_max_tile = sum(max_tiles) / len(max_tiles)
    max_score = max(scores)
    max_tile = max(max_tiles)
    
    print(f"\n测试结果统计:")
    print(f"平均分数: {avg_score:.0f}")
    print(f"最高分数: {max_score}")
    print(f"平均最大方块: {avg_max_tile:.0f}")
    print(f"最大方块: {max_tile}")
    
    # 统计达到不同数字的次数
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    print(f"\n最大方块分布:")
    for tile in sorted(tile_counts.keys(), reverse=True):
        count = tile_counts[tile]
        percentage = count / num_games * 100
        print(f"  {tile}: {count} 次 ({percentage:.1f}%)")


def demo_play(ai=None):
    """演示强化学习AI游戏"""
    print("=" * 50)
    print("强化学习AI演示")
    print("=" * 50)
    
    if ai is None:
        ai = RLAI(training_mode=False)
    
    if ai.use_fallback:
        print("警告: 使用fallback策略（PyTorch不可用）")
    
    game = Game2048()
    moves = 0
    
    print(f"初始状态:")
    print(game)
    
    while not game.game_over and moves < 200:
        move = ai.get_best_move(game)
        if move is None:
            break
        
        old_score = game.score
        game.move(move)
        score_gain = game.score - old_score
        
        print(f"\n移动 {moves + 1}: {move}")
        print(f"得分: +{score_gain} (总分: {game.score})")
        print(game)
        
        moves += 1
        
        # 暂停显示
        if moves % 5 == 0:
            input("按回车键继续...")
    
    print(f"\n游戏结束!")
    print(f"总移动次数: {moves}")
    print(f"最终分数: {game.score}")
    print(f"最大方块: {game.get_max_tile()}")


def list_checkpoints():
    """列出可用的检查点"""
    checkpoint_dir = "/workspaces/player-infinity/workspace/models/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("检查点目录不存在")
        return
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_') and file.endswith('.pth'):
            file_path = os.path.join(checkpoint_dir, file)
            checkpoints.append((file_path, os.path.getmtime(file_path)))
    
    if not checkpoints:
        print("没有找到检查点文件")
        return
    
    print("可用的检查点:")
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    for i, (file_path, mtime) in enumerate(checkpoints):
        file_name = os.path.basename(file_path)
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        print(f"  {i+1}. {file_name} ({mtime_str})")


def load_checkpoint(checkpoint_path):
    """从检查点加载模型"""
    ai = RLAI(training_mode=False)
    if ai.load_checkpoint(checkpoint_path):
        print("检查点加载成功")
        ai.print_current_stats()
        return ai
    else:
        print("检查点加载失败")
        return None


def main():
    parser = argparse.ArgumentParser(description="2048强化学习AI训练和测试工具")
    parser.add_argument('--mode', choices=['train', 'test', 'demo', 'list-checkpoints', 'load-checkpoint'], 
                        default='train', help='运行模式')
    parser.add_argument('--episodes', type=int, default=1000, help='训练回合数 (默认: 1000)')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径')
    parser.add_argument('--resume', action='store_true', help='从现有模型继续训练')
    parser.add_argument('--test-games', type=int, default=10, help='测试游戏局数 (默认: 10)')
    parser.add_argument('--checkpoint', type=str, help='检查点文件路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_rl_model(args.episodes, args.model, args.resume)
    elif args.mode == 'test':
        ai = RLAI(model_path=args.model, training_mode=False)
        test_model(ai, args.test_games)
    elif args.mode == 'demo':
        demo_play()
    elif args.mode == 'list-checkpoints':
        list_checkpoints()
    elif args.mode == 'load-checkpoint':
        if not args.checkpoint:
            print("错误: 需要指定检查点路径 --checkpoint")
            return
        ai = load_checkpoint(args.checkpoint)
        if ai:
            demo_play(ai)


if __name__ == "__main__":
    main() 
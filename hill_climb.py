"""
hill_climb.py  —  Hill-Climbing Training
=========================================
Algorithm:
  1. Evaluate current checkpoint
  2. Run N episodes of training
  3. Evaluate result
  4. If better: keep it, print improvement, repeat from 2
  5. If not better: revert to previous best, repeat from 2
  6. Stop after max_attempts total attempts, or max_no_improve consecutive failures

Usage:
  python hill_climb.py --episodes 3200 --eval-games 500
  python hill_climb.py --episodes 3200 --eval-games 500 --train-net bid --lr 1e-5
"""

import os, sys, json, shutil, argparse, time, io, contextlib
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import BidNet, PlayNet, infer_config
from train import load_checkpoint, save_checkpoint, quick_eval, train, CKPT_CONFIG


def _ts():
    return datetime.now().strftime('%H:%M:%S')


def hill_climb(
    episodes:       int   = 3200,
    eval_games:     int   = 500,
    train_net:      str   = 'both',
    opponent:       str   = 'heuristic',
    lr:             float = 1e-4,
    hidden:         int   = 256,
    n_res:          int   = 4,
    entropy_coef:   float = 0.0,
    save_path:      str   = 'checkpoints',
    max_attempts:   int   = 200,
    max_no_improve: int   = 20,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_path = os.path.join(save_path, 'hill_best.pt')

    print(f"Hill-climb training  [{device}]")
    print(f"  {episodes} episodes/attempt  eval_games={eval_games}")
    print(f"  train_net={train_net}  lr={lr}  max_attempts={max_attempts}")
    print(f"  Stop after {max_no_improve} consecutive non-improvements")
    print()

    # Load starting checkpoint and evaluate
    bid_net, play_net, start_ep, stats = load_checkpoint(
        save_path, device, hidden, n_res)
    best_wr = quick_eval(bid_net, play_net, eval_games, device)
    best_ep = start_ep

    print(f"[{_ts()}] Starting EvalVsHeur={best_wr:.3f}  ep={best_ep}")

    # Save best weights separately so we can always revert
    torch.save({'bid_net':  bid_net.state_dict(),
                'play_net': play_net.state_dict()}, best_path)

    attempt    = 0
    no_improve = 0
    history    = []   # (attempt, wr, improved)

    while attempt < max_attempts and no_improve < max_no_improve:
        attempt += 1

        # Always start each attempt from the current best
        work_path = os.path.join(save_path, 'hill_work')
        os.makedirs(work_path, exist_ok=True)
        work_ckpt = os.path.join(work_path, 'latest.pt')
        work_meta = work_ckpt + '.json'
        shutil.copy(best_path, work_ckpt)
        with open(work_meta, 'w') as f:
            json.dump({'episode': best_ep, 'stats': stats,
                       'config': {'hidden': hidden, 'n_res': n_res,
                                   'BID_STATE_DIM': 95, 'PLAY_STATE_DIM': 136}}, f)

        # Train
        t0 = time.time()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train(n_episodes=episodes, batch_size=128, save_path=work_path,
                  opponent=opponent, resume=True, hidden=hidden, n_res=n_res,
                  lr=lr, train_net=train_net, entropy_coef=entropy_coef,
                  eval_games=0)

        # Evaluate result
        cand_bid, cand_play, cand_ep, cand_stats = load_checkpoint(
            work_path, device, hidden, n_res)
        wr = quick_eval(cand_bid, cand_play, eval_games, device)
        elapsed = time.time() - t0

        improved = wr > best_wr
        history.append((attempt, wr, improved))

        if improved:
            best_wr    = wr
            best_ep    = cand_ep
            stats      = cand_stats
            no_improve = 0
            torch.save({'bid_net':  cand_bid.state_dict(),
                        'play_net': cand_play.state_dict()}, best_path)
            save_checkpoint(cand_bid, cand_play, cand_ep, stats,
                            save_path, tag=cand_ep)
            marker = '✓ IMPROVED'
        else:
            no_improve += 1
            marker = f'✗ no improvement  ({no_improve}/{max_no_improve})'

        print(f"[{_ts()}] Attempt {attempt:3d}  EvalVsHeur={wr:.3f}  "
              f"best={best_wr:.3f}  {marker}  ({elapsed:.0f}s)")

    # Clean up work directory
    if os.path.exists(work_path):
        shutil.rmtree(work_path)

    # Final summary
    improvements = [(a, w) for a, w, imp in history if imp]
    print(f"\n{'═'*60}")
    print(f"Hill-climb complete after {attempt} attempts.")
    print(f"Final best EvalVsHeur = {best_wr:.3f}")
    print(f"Improvements found: {len(improvements)}")
    if improvements:
        for a, w in improvements:
            print(f"  Attempt {a:3d}: {w:.3f}")
    if no_improve >= max_no_improve:
        print(f"Stopped: {max_no_improve} consecutive non-improvements.")
    print(f"{'═'*60}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hill-climbing training')
    p.add_argument('--episodes',       type=int,   default=3200)
    p.add_argument('--eval-games',     type=int,   default=500)
    p.add_argument('--train-net',      default='both',
                   choices=['bid', 'play', 'both'])
    p.add_argument('--opponent',       default='heuristic',
                   choices=['heuristic', 'self'])
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--hidden',         type=int,   default=256)
    p.add_argument('--n-res',          type=int,   default=4)
    p.add_argument('--entropy-coef',   type=float, default=0.0)
    p.add_argument('--save-path',      default='checkpoints')
    p.add_argument('--max-attempts',   type=int,   default=200)
    p.add_argument('--max-no-improve', type=int,   default=20)
    a = p.parse_args()

    hill_climb(
        episodes=a.episodes,
        eval_games=a.eval_games,
        train_net=a.train_net,
        opponent=a.opponent,
        lr=a.lr,
        hidden=a.hidden,
        n_res=a.n_res,
        entropy_coef=a.entropy_coef,
        save_path=a.save_path,
        max_attempts=a.max_attempts,
        max_no_improve=a.max_no_improve,
    )

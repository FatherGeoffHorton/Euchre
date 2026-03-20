"""
beam_train.py  —  Beam Search over Training Trajectories
=========================================================
Algorithm:
  1. From current best checkpoint, run K independent arms of N episodes each
  2. Evaluate each arm vs heuristic (eval_games games)
  3. Keep the arm with the highest EvalVsHeur
  4. If best arm beats current best, it becomes the new starting point
  5. Repeat for R rounds

This exploits the noisy nature of REINFORCE: instead of hoping the gradient
walk goes uphill, we sample K trajectories and keep the best one.

Usage:
  python beam_train.py --rounds 20 --arms 5 --episodes 3200 --eval-games 200
  python beam_train.py --rounds 20 --arms 5 --episodes 3200 --train-net bid
"""

import os, sys, json, shutil, argparse, time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import BidNet, PlayNet, infer_config
from agent import EuchreAgent, HeuristicAgent
from train import (load_checkpoint, save_checkpoint, quick_eval,
                   train, CKPT_CONFIG)


def _timestamp():
    return datetime.now().strftime('%H:%M:%S')


def beam_train(
    rounds:      int   = 20,
    arms:        int   = 5,
    episodes:    int   = 3200,
    eval_games:  int   = 200,
    train_net:   str   = 'both',
    opponent:    str   = 'heuristic',
    lr:          float = 1e-5,
    hidden:      int   = 256,
    n_res:       int   = 4,
    entropy_coef: float = 0.0,
    save_path:   str   = 'checkpoints',
    beam_path:   str   = 'checkpoints/beam',
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(beam_path, exist_ok=True)

    print(f"Beam training  [{device}]")
    print(f"  {rounds} rounds × {arms} arms × {episodes} episodes")
    print(f"  eval_games={eval_games}  train_net={train_net}  lr={lr}")
    print()

    # ── Establish starting eval ────────────────────────────────────
    bid_net, play_net, start_ep, stats = load_checkpoint(
        save_path, device, hidden, n_res)

    best_wr = quick_eval(bid_net, play_net, eval_games, device)
    print(f"[{_timestamp()}] Starting EvalVsHeur = {best_wr:.3f}  (ep {start_ep})")

    # Save the starting point as beam/best.pt
    best_path = os.path.join(beam_path, 'best.pt')
    torch.save({'bid_net': bid_net.state_dict(),
                'play_net': play_net.state_dict()}, best_path)
    best_ep   = start_ep
    best_stats = stats

    history = []   # (round, arm, wr) for each arm tried

    for rnd in range(1, rounds + 1):
        print(f"\n{'─'*60}")
        print(f"Round {rnd}/{rounds}  (current best={best_wr:.3f})")
        print(f"{'─'*60}")

        arm_results = []   # (wr, arm_ckpt_path)

        for arm in range(1, arms + 1):
            arm_path = os.path.join(beam_path, f'arm_{arm}')
            os.makedirs(arm_path, exist_ok=True)

            # Copy best checkpoint into arm directory
            arm_ckpt = os.path.join(arm_path, 'latest.pt')
            arm_meta = arm_ckpt + '.json'
            shutil.copy(best_path, arm_ckpt)
            with open(arm_meta, 'w') as f:
                json.dump({
                    'episode': best_ep,
                    'stats': best_stats,
                    'config': {
                        'hidden': hidden, 'n_res': n_res,
                        **CKPT_CONFIG
                    }
                }, f)

            t0 = time.time()
            # Run training silently
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                train(
                    n_episodes=episodes,
                    batch_size=128,
                    save_path=arm_path,
                    opponent=opponent,
                    resume=True,
                    hidden=hidden,
                    n_res=n_res,
                    lr=lr,
                    train_net=train_net,
                    entropy_coef=entropy_coef,
                    eval_games=0,   # no inline eval during arm training
                )

            # Evaluate this arm
            arm_bid, arm_play, arm_ep, arm_stats = load_checkpoint(
                arm_path, device, hidden, n_res)
            wr = quick_eval(arm_bid, arm_play, eval_games, device)
            elapsed = time.time() - t0

            print(f"  [{_timestamp()}] Arm {arm}/{arms}  EvalVsHeur={wr:.3f}"
                  f"  ({elapsed:.0f}s)")
            arm_results.append((wr, arm_bid, arm_play, arm_ep, arm_stats))
            history.append((rnd, arm, wr))

        # Pick best arm
        arm_results.sort(key=lambda x: x[0], reverse=True)
        top_wr, top_bid, top_play, top_ep, top_stats = arm_results[0]

        if top_wr > best_wr:
            best_wr    = top_wr
            best_ep    = top_ep
            best_stats = top_stats
            torch.save({'bid_net': top_bid.state_dict(),
                        'play_net': top_play.state_dict()}, best_path)
            # Also update the main checkpoint
            save_checkpoint(top_bid, top_play, top_ep, top_stats,
                            save_path, tag=top_ep)
            print(f"\n  ✓ New best: {best_wr:.3f}  (saved to {save_path})")
        else:
            print(f"\n  ✗ No improvement  (best arm={top_wr:.3f} ≤ current={best_wr:.3f})")
            print(f"    Staying at current best.")

        # Print round summary
        arm_wrs = [r[0] for r in arm_results]
        print(f"  Arm results: {[f'{w:.3f}' for w in sorted(arm_wrs, reverse=True)]}")

    # ── Final summary ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"Beam training complete.")
    print(f"Final best EvalVsHeur = {best_wr:.3f}")

    # Show best per round
    print(f"\nBest per round:")
    for rnd in range(1, rounds + 1):
        round_results = [(wr) for r, _, wr in history if r == rnd]
        if round_results:
            print(f"  Round {rnd:2d}: best={max(round_results):.3f}"
                  f"  mean={sum(round_results)/len(round_results):.3f}"
                  f"  worst={min(round_results):.3f}")

    # Clean up arm directories
    for arm in range(1, arms + 1):
        arm_path = os.path.join(beam_path, f'arm_{arm}')
        if os.path.exists(arm_path):
            shutil.rmtree(arm_path)
    print(f"{'═'*60}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Beam search over training trajectories')
    p.add_argument('--rounds',       type=int,   default=20)
    p.add_argument('--arms',         type=int,   default=5)
    p.add_argument('--episodes',     type=int,   default=3200)
    p.add_argument('--eval-games',   type=int,   default=200)
    p.add_argument('--train-net',    default='both',
                   choices=['bid','play','both'])
    p.add_argument('--opponent',     default='heuristic',
                   choices=['heuristic','self'])
    p.add_argument('--lr',           type=float, default=1e-5)
    p.add_argument('--hidden',       type=int,   default=256)
    p.add_argument('--n-res',        type=int,   default=4)
    p.add_argument('--entropy-coef', type=float, default=0.0)
    p.add_argument('--save-path',    default='checkpoints')
    p.add_argument('--beam-path',    default='checkpoints/beam')
    a = p.parse_args()

    beam_train(
        rounds=a.rounds,
        arms=a.arms,
        episodes=a.episodes,
        eval_games=a.eval_games,
        train_net=a.train_net,
        opponent=a.opponent,
        lr=a.lr,
        hidden=a.hidden,
        n_res=a.n_res,
        entropy_coef=a.entropy_coef,
        save_path=a.save_path,
        beam_path=a.beam_path,
    )

"""
evaluate.py  —  Win Rate vs Heuristic
"""
import os, sys, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import BidNet, PlayNet, infer_config
from agent import EuchreAgent, HeuristicAgent
from train import _run_hand_heuristic
from euchre_game import EuchreGame


def evaluate(checkpoint: str, n_games: int = 500, bid_bias: float = 0.0):
    device = torch.device('cpu')
    if not os.path.exists(checkpoint):
        print(f"No checkpoint at {checkpoint}"); return

    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    try:
        cfg = infer_config(weights)
    except ValueError as e:
        print(f"Incompatible: {e}"); return

    bid_net  = BidNet(hidden=cfg['hidden'],  n_res=cfg['n_res'])
    play_net = PlayNet(hidden=cfg['hidden'], n_res=cfg['n_res'])
    bid_net.load_state_dict(weights['bid_net'])
    play_net.load_state_dict(weights['play_net'])

    meta_path = checkpoint + '.json'
    episode = '?'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            episode = json.load(f).get('episode', '?')

    print(f"\nEvaluating: {checkpoint}  (ep {episode})")
    print(f"hidden={cfg['hidden']} n_res={cfg['n_res']}  bid_bias={bid_bias}")

    agent     = EuchreAgent(bid_net, play_net, bid_bias=bid_bias)
    agent.eval()
    heuristic = HeuristicAgent()

    wins = 0; score_A = 0; score_B = 0
    for i in range(n_games):
        game = EuchreGame()
        while game.game_over() is None:
            game.new_hand()
            _run_hand_heuristic(game, agent, heuristic)
        if game.game_over() == 0: wins += 1
        score_A += game.score[0]
        score_B += game.score[1]
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n_games}  win rate so far: {wins/(i+1):.1%}")

    print(f"\n  Games: {n_games}  Win rate: {wins/n_games:.1%}")
    print(f"  Avg score — AI: {score_A/n_games:.1f}  Heur: {score_B/n_games:.1f}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/latest.pt')
    p.add_argument('--games',     type=int,   default=500)
    p.add_argument('--bid-bias',  type=float, default=0.0)
    a = p.parse_args()
    evaluate(a.checkpoint, a.games, a.bid_bias)

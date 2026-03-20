"""
diagnose.py  —  Bid Calibration Diagnostic
===========================================
Runs n hands and reports:
  - AI call rate when given the opportunity
  - Euchre rate and expected value per call
  - Comparison with heuristic baseline
"""

import os, sys, json, argparse, torch
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from euchre_game import EuchreGame, card_suit
from models import BidNet, PlayNet, infer_config
from agent import EuchreAgent, HeuristicAgent


def diagnose(checkpoint: str, n_hands: int = 2000, bid_bias: float = 0.0):
    device = torch.device('cpu')

    if not os.path.exists(checkpoint):
        print(f"No checkpoint at {checkpoint}")
        return

    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    try:
        cfg = infer_config(weights)
    except ValueError as e:
        print(f"Incompatible checkpoint: {e}")
        return

    hidden, n_res = cfg['hidden'], cfg['n_res']
    bid_net  = BidNet(hidden=hidden,  n_res=n_res)
    play_net = PlayNet(hidden=hidden, n_res=n_res)
    bid_net.load_state_dict(weights['bid_net'])
    play_net.load_state_dict(weights['play_net'])
    print(f"Checkpoint: hidden={hidden} n_res={n_res}  bid_bias={bid_bias}")

    agent     = EuchreAgent(bid_net, play_net, bid_bias=bid_bias)
    agent.eval()
    heuristic = HeuristicAgent()

    ai_bids  = 0;  heur_bids = 0;  total_hands = 0
    ai_results   = defaultdict(int)
    heur_results = defaultdict(int)

    game = EuchreGame()
    while total_hands < n_hands:
        game.new_hand()
        maker_team = None
        trump_called = False

        for player in game.bid1_order():
            my_team = player % 2
            if my_team == 0:
                order, alone = agent.bid_round1(game, player)
            else:
                order, alone = heuristic.bid_round1(game, player)
            if order:
                game.accept_upcard(player, alone=alone)
                maker_team = my_team
                (ai_bids if my_team == 0 else heur_bids).__class__  # dummy
                if my_team == 0: ai_bids += 1
                else:           heur_bids += 1
                trump_called = True
                break
            else:
                if game.pass_bid(): break

        if not trump_called and game.phase == 'bid2':
            for player in game.bid2_order():
                my_team = player % 2
                if my_team == 0:
                    suit, alone = agent.bid_round2(game, player)
                else:
                    suit, alone = heuristic.bid_round2(game, player)
                if suit is not None:
                    game.call_trump(player, suit, alone=alone)
                    maker_team = my_team
                    if my_team == 0: ai_bids += 1
                    else:           heur_bids += 1
                    trump_called = True
                    break
                else:
                    if game.pass_bid(): break

        if not trump_called or game.phase != 'play':
            continue

        # Play with heuristic for both sides (isolates bid quality)
        for _ in range(5):
            active = game._active_players()
            leader = game.trick_leader
            if leader not in active: leader = active[0]
            idx   = active.index(leader)
            order = [active[(idx+i)%len(active)] for i in range(len(active))]
            for p in order:
                if not game.hands[p]: continue
                game.play_card(p, heuristic.play_card(game, p))
            if game.trick_complete():
                game.resolve_trick()
            if game.phase == 'score': break

        game.score_hand()
        total_hands += 1

        twm = game.tricks_won[game.maker_team]
        twd = game.tricks_won[1 - game.maker_team]
        outcome = 'euchre' if twd >= 3 else 'march' if twm == 5 else '1pt'

        if maker_team == 0:   ai_results[outcome]   += 1
        elif maker_team == 1: heur_results[outcome] += 1

    W = 55
    print(f"\n{'═'*W}")
    print(f"  Diagnostic: {n_hands} hands")
    print(f"{'═'*W}")
    print(f"\n  Bidding (AI = team 0, players 0 & 2):")
    print(f"    AI   calls trump: {ai_bids:4d}  ({ai_bids/total_hands*100:.1f}% of hands)")
    print(f"    Heur calls trump: {heur_bids:4d}  ({heur_bids/total_hands*100:.1f}% of hands)")
    print(f"    Note: heuristic often bids before AI — AI's true call")
    print(f"    rate when given opportunity is higher.")

    def report(label, results, n):
        if n == 0: return
        e = results['euchre']; p = results['1pt']; m = results['march']
        ev = (m*2 + p - e*2) / n
        print(f"\n  When {label} calls ({n} times):")
        print(f"    Euchred: {e:4d} ({e/n*100:.1f}%)")
        print(f"    1 point: {p:4d} ({p/n*100:.1f}%)")
        print(f"    March:   {m:4d} ({m/n*100:.1f}%)")
        print(f"    EV/call: {ev:+.3f}  (>0 = profitable)")

    report("AI",        ai_results,   ai_bids)
    report("Heuristic", heur_results, heur_bids)
    print(f"\n{'═'*W}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/latest.pt')
    p.add_argument('--hands',     type=int,   default=2000)
    p.add_argument('--bid-bias',  type=float, default=0.0)
    a = p.parse_args()
    diagnose(a.checkpoint, a.hands, a.bid_bias)

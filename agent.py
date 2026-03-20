"""
agent.py  —  EuchreAgent and HeuristicAgent
============================================
EuchreAgent wraps BidNet + PlayNet with:
  - score_aggression() scaling on bid logits
  - bid_bias for threshold tuning without retraining
  - greedy (eval) vs stochastic (train) mode
  - legal-move masking on PlayNet output

HeuristicAgent is a rule-based baseline:
  - Bid round 1: order up if 3+ effective trump
  - Bid round 2: call best non-upcard suit if 3+ cards
  - Play: lead highest trump; follow with lowest winner; dump lowest
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from euchre_game import (
    EuchreGame, card_suit, card_rank, effective_suit,
    SAME_COLOR, NUM_CARDS, WINNING_SCORE
)
from state_encoder import encode_bid_state, encode_play_state
from models import BidNet, PlayNet


# ── Score-awareness ────────────────────────────────────────────────

def score_aggression(game: EuchreGame, my_team: int) -> float:
    """
    Aggression multiplier [0.5 .. 2.0] based on score situation.
    > 1 = bid more aggressively.
    """
    my_s  = game.score[my_team]
    opp_s = game.score[1 - my_team]
    if my_s >= WINNING_SCORE - 1: return 0.6   # close to winning — play safe
    if opp_s >= WINNING_SCORE - 1: return 1.8  # opponent near win — gamble
    deficit = opp_s - my_s
    if deficit >= 4: return 1.5
    if deficit >= 2: return 1.2
    return 1.0


# ── Neural agent ───────────────────────────────────────────────────

class EuchreAgent:
    def __init__(self,
                 bid_net:     BidNet,
                 play_net:    PlayNet,
                 device:      torch.device = torch.device('cpu'),
                 temperature: float = 1.0,
                 greedy:      bool  = False,
                 bid_bias:    float = 0.0):
        self.bid_net     = bid_net
        self.play_net    = play_net
        self.device      = device
        self.temperature = temperature
        self.greedy      = greedy
        self.bid_bias    = bid_bias  # >0 = more willing to call

    def train(self):
        self.bid_net.train()
        self.play_net.train()
        self.greedy = False

    def eval(self):
        self.bid_net.eval()
        self.play_net.eval()
        self.greedy = True

    def _bid_action(self, game: EuchreGame, player: int,
                    r2_scale: float = 1.0) -> int:
        """Run bid network, return action index (0=pass, 1=bid, 2=loner)."""
        state   = encode_bid_state(game, player).to(self.device)
        with torch.no_grad():
            logits = self.bid_net(state.unsqueeze(0)).squeeze(0)
        agg    = score_aggression(game, player % 2)
        scaled = logits.clone()
        scaled[1] = scaled[1] * agg * r2_scale + self.bid_bias
        scaled[2] = scaled[2] * agg * r2_scale * 0.7 + self.bid_bias * 0.5
        probs = F.softmax(scaled / self.temperature, dim=-1)
        if self.greedy:
            return int(probs.argmax().item())
        return int(torch.multinomial(probs, 1).item())

    def bid_round1(self, game: EuchreGame, player: int) -> Tuple[bool, bool]:
        action = self._bid_action(game, player)
        return action > 0, action == 2

    def bid_round2(self, game: EuchreGame, player: int) -> Tuple[Optional[int], bool]:
        action = self._bid_action(game, player, r2_scale=0.85)
        if action == 0:
            return None, False
        alone = action == 2
        # Choose best non-upcard suit
        upcard_suit = card_suit(game.upcard)
        hand        = game.hands[player]
        suit_scores = [0.0] * 4
        for c in hand:
            s = card_suit(c)
            suit_scores[s] += 1
            suit_scores[SAME_COLOR[s]] += 0.5
        best = max((s for s in range(4) if s != upcard_suit),
                   key=lambda s: suit_scores[s])
        return best, alone

    def play_card(self, game: EuchreGame, player: int) -> int:
        legal = game.legal_plays(player)
        state = encode_play_state(game, player).to(self.device)
        with torch.no_grad():
            logits = self.play_net(state.unsqueeze(0)).squeeze(0)
        mask = torch.zeros(NUM_CARDS, dtype=torch.bool, device=self.device)
        for c in legal:
            mask[c] = True
        logits = logits.masked_fill(~mask, float('-inf'))
        probs  = F.softmax(logits / self.temperature, dim=-1)
        if self.greedy:
            card = int(probs.argmax().item())
        else:
            card = int(torch.multinomial(probs, 1).item())
        assert card in legal
        return card


# ── Heuristic agent ────────────────────────────────────────────────

class HeuristicAgent:
    """Simple rule-based agent used as baseline and training opponent."""

    def bid_round1(self, game: EuchreGame, player: int) -> Tuple[bool, bool]:
        trump = card_suit(game.upcard)
        hand  = game.hands[player]
        tc = sum(1 for c in hand if effective_suit(c, trump) == trump)
        has_right = any(card_suit(c) == trump and card_rank(c) == 2 for c in hand)
        return tc >= (2 if has_right else 3), False

    def bid_round2(self, game: EuchreGame, player: int) -> Tuple[Optional[int], bool]:
        upcard_suit = card_suit(game.upcard)
        hand        = game.hands[player]
        best_suit, best_cnt = None, 0
        for s in range(4):
            if s == upcard_suit:
                continue
            cnt = sum(1 for c in hand if effective_suit(c, s) == s)
            if cnt > best_cnt:
                best_cnt, best_suit = cnt, s
        # Screw-the-dealer: last bidder must call
        if player == game.bid2_order()[-1] and best_suit is not None:
            return best_suit, False
        return (best_suit, False) if best_cnt >= 3 else (None, False)

    def play_card(self, game: EuchreGame, player: int) -> int:
        legal = game.legal_plays(player)
        trump = game.trump

        def power(c):
            return c if game.led_suit == -1 else \
                   __import__('euchre_game').card_power(c, trump, game.led_suit)

        if game.led_suit == -1:
            # Leading: play highest trump, else highest card
            trump_cards = [c for c in legal if effective_suit(c, trump) == trump]
            pool = trump_cards if trump_cards else legal
            return max(pool, key=lambda c: power(c))
        else:
            # Following: win cheaply if possible, else dump lowest
            winners = [c for c in legal
                       if __import__('euchre_game').card_power(c, trump, game.led_suit) > 0]
            if winners:
                return min(winners,
                           key=lambda c: __import__('euchre_game').card_power(
                               c, trump, game.led_suit))
            return min(legal, key=card_rank)

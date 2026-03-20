"""
state_encoder.py  —  Game State → Float Tensors
================================================
BID state  (95 dims)
  [0:24]   hand cards (one-hot)
  [24:48]  upcard (one-hot)
  [48:50]  scores [mine/10, opp/10]
  [50:54]  position relative to dealer (one-hot)
  [54:58]  bid context [round1, round2, passes/4, upcard_suit/3]
  [58:82]  cards played this hand (one-hot)
  [82:90]  hand strength [trump_ct/6, has_right, has_left, strong,
                          best_other/6, best_other>=3, 0, 0]
  [90:94]  void suits (one bit per suit, 1=void)
  [94:95]  ace count / 2

PLAY state  (136 dims)
  [0:24]   hand cards
  [24:48]  trump suit indicator (all 6 cards of trump suit = 1)
  [48:50]  scores [mine/10, opp/10]
  [50:54]  trick position (one-hot: lead/2nd/3rd/4th)
  [54:64]  trick situation (10 features)
  [64:88]  cards played this hand (one-hot)
  [88:112] cards in current trick (one-hot)
  [112:136] partner's card this trick (one-hot)
"""

import torch
from typing import List, Optional
from euchre_game import (
    card_suit, card_rank, effective_suit, SAME_COLOR, NUM_CARDS
)

BID_STATE_DIM  = 95
PLAY_STATE_DIM = 136


def _bits(cards: List[int], size: int = NUM_CARDS) -> List[float]:
    v = [0.0] * size
    for c in cards:
        if 0 <= c < size:
            v[c] = 1.0
    return v


def _card_bits(card: Optional[int]) -> List[float]:
    return _bits([card] if card is not None and card >= 0 else [])


def encode_bid_state(game, player: int) -> torch.Tensor:
    v: List[float] = []
    hand = game.hands[player]
    my_team  = player % 2
    opp_team = 1 - my_team

    # [0:24] hand
    v += _bits(hand)

    # [24:48] upcard
    v += _card_bits(game.upcard)

    # [48:50] scores
    v += [game.score[my_team] / 10.0, game.score[opp_team] / 10.0]

    # [50:54] position relative to dealer
    pos = [0.0] * 4
    pos[(player - game.dealer) % 4] = 1.0
    v += pos

    # [54:58] bid round context
    round1 = 1.0 if game.phase == 'bid1' else 0.0
    round2 = 1.0 if game.phase == 'bid2' else 0.0
    passes = game.bid_passes / 4.0
    upcard_suit_norm = card_suit(game.upcard) / 3.0 if game.upcard >= 0 else 0.0
    v += [round1, round2, passes, upcard_suit_norm]

    # [58:82] played cards
    v += _bits(game.played_cards)

    # [82:90] hand strength
    upcard_suit = card_suit(game.upcard) if game.upcard >= 0 else -1

    def trump_info(suit):
        count = 0
        has_right = has_left = False
        for c in hand:
            s, r = card_suit(c), card_rank(c)
            if r == 2 and s == suit:              has_right = True; count += 1
            elif r == 2 and s == SAME_COLOR[suit]: has_left  = True; count += 1
            elif effective_suit(c, suit) == suit:  count += 1
        return count, has_right, has_left

    if upcard_suit >= 0:
        dealer_bonus = 1 if player == game.dealer else 0
        tc, hr, hl = trump_info(upcard_suit)
        effective_tc = tc + dealer_bonus
        v += [effective_tc / 6.0, float(hr), float(hl), float(effective_tc >= 3)]

        best_other = max(
            trump_info(s)[0] for s in range(4) if s != upcard_suit
        )
        v += [best_other / 6.0, float(best_other >= 3), 0.0, 0.0]
    else:
        v += [0.0] * 8

    # [90:94] void suits
    for suit in range(4):
        has = any(card_suit(c) == suit for c in hand)
        v.append(0.0 if has else 1.0)

    # [94:95] ace count
    aces = sum(1 for c in hand if card_rank(c) == 5)
    v.append(aces / 2.0)

    assert len(v) == BID_STATE_DIM, f"Bid state: expected {BID_STATE_DIM}, got {len(v)}"
    return torch.tensor(v, dtype=torch.float32)


def encode_play_state(game, player: int) -> torch.Tensor:
    v: List[float] = []
    my_team  = player % 2
    opp_team = 1 - my_team

    # [0:24] hand
    v += _bits(game.hands[player])

    # [24:48] trump indicator
    trump_bits = [0.0] * NUM_CARDS
    if game.trump >= 0:
        for rank in range(6):
            trump_bits[game.trump * 6 + rank] = 1.0
        # Mark left bower card as trump too
        left_suit = SAME_COLOR[game.trump]
        trump_bits[left_suit * 6 + 2] = 1.0
    v += trump_bits

    # [48:50] scores
    v += [game.score[my_team] / 10.0, game.score[opp_team] / 10.0]

    # [50:54] trick position
    active = game._active_players()
    trick_pos = active.index(player) if player in active else 0
    pos = [0.0] * 4
    pos[trick_pos] = 1.0
    v += pos

    # [54:64] trick situation
    tricks_played    = sum(game.tricks_won)
    tricks_remaining = max(5 - tricks_played, 1)
    my_tw    = game.tricks_won[my_team]
    opp_tw   = game.tricks_won[opp_team]
    maker_tw = game.tricks_won[game.maker_team]
    def_tw   = game.tricks_won[1 - game.maker_team]
    i_am_maker = 1.0 if my_team == game.maker_team else 0.0

    maker_needs  = max(0, 3 - maker_tw)
    march_needs  = max(0, 5 - maker_tw)
    euchre_needs = max(0, 3 - def_tw)
    my_needs     = maker_needs if i_am_maker else euchre_needs

    v += [
        my_tw   / 5.0,
        opp_tw  / 5.0,
        maker_tw / 5.0,
        def_tw  / 5.0,
        tricks_remaining / 5.0,
        i_am_maker,
        maker_needs  / tricks_remaining,
        march_needs  / tricks_remaining,
        euchre_needs / tricks_remaining,
        my_needs     / tricks_remaining,
    ]

    # [64:88] played cards history
    v += _bits(game.played_cards)

    # [88:112] current trick
    trick_cards = [c for c in game.current_trick if c is not None]
    v += _bits(trick_cards)

    # [112:136] partner's card this trick
    partner = (player + 2) % 4
    v += _card_bits(game.current_trick[partner])

    assert len(v) == PLAY_STATE_DIM, f"Play state: expected {PLAY_STATE_DIM}, got {len(v)}"
    return torch.tensor(v, dtype=torch.float32)

"""
euchre_game.py  —  Complete Euchre Rules Engine
================================================
Card encoding
  24 cards: 9 10 J Q K A  ×  ♣ ♦ ♥ ♠
  Suit:  0=♣  1=♦  2=♥  3=♠
  Rank:  0=9  1=10 2=J  3=Q  4=K  5=A
  Index: suit*6 + rank

Left bower rule: Jack of the SAME COLOR as trump is also trump.
  Black suits: ♣(0) and ♠(3)  →  SAME_COLOR = {0:3, 3:0}
  Red suits:   ♦(1) and ♥(2)  →  SAME_COLOR = {1:2, 2:1}
"""

import random
from typing import List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────
SUITS      = ['♣', '♦', '♥', '♠']
SUIT_NAMES = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
RANKS      = ['9', '10', 'J', 'Q', 'K', 'A']
RANK_NAMES = ['9', '10', 'Jack', 'Queen', 'King', 'Ace']
NUM_CARDS  = 24
WINNING_SCORE = 10

# Same-color pairs for bower logic: black↔black, red↔red
SAME_COLOR = {0: 3, 1: 2, 2: 1, 3: 0}


# ── Card helpers ───────────────────────────────────────────────────

def card_index(suit: int, rank: int) -> int:
    return suit * 6 + rank

def card_suit(idx: int) -> int:
    return idx // 6

def card_rank(idx: int) -> int:
    return idx % 6

def card_name(idx: int) -> str:
    return RANKS[card_rank(idx)] + SUITS[card_suit(idx)]

def card_long_name(idx: int) -> str:
    return f"{RANK_NAMES[card_rank(idx)]} of {SUIT_NAMES[card_suit(idx)]}"

def effective_suit(card: int, trump: int) -> int:
    """Effective suit of a card given trump. Left bower counts as trump."""
    s, r = card_suit(card), card_rank(card)
    if r == 2 and s == SAME_COLOR[trump]:
        return trump
    return s

def card_power(card: int, trump: int, led_suit: int) -> int:
    """Trick-winning power. 0 = cannot win. Higher = stronger."""
    s, r = card_suit(card), card_rank(card)
    if r == 2 and s == trump:                return 1000  # right bower
    if r == 2 and s == SAME_COLOR[trump]:    return 999   # left bower
    if effective_suit(card, trump) == trump: return 100 + r
    if effective_suit(card, trump) == led_suit: return r
    return 0


# ── Deck / deal ────────────────────────────────────────────────────

def build_deck() -> List[int]:
    return list(range(NUM_CARDS))

def deal_hands(deck: Optional[List[int]] = None) -> Tuple[List[List[int]], int]:
    """Returns (hands[4], upcard). Each hand has 5 sorted cards."""
    if deck is None:
        deck = build_deck()
    d = deck[:]
    random.shuffle(d)
    hands = [sorted(d[i*5:(i+1)*5]) for i in range(4)]
    return hands, d[20]


# ── Game class ─────────────────────────────────────────────────────

class EuchreGame:
    """
    Complete game state manager.
    Players 0,2 = team A;  players 1,3 = team B.
    """

    def __init__(self):
        self.score  = [0, 0]
        self.dealer = 0
        self._reset_hand()

    def _reset_hand(self):
        self.hands:       List[List[int]] = [[] for _ in range(4)]
        self.upcard:      int  = -1
        self.trump:       int  = -1
        self.maker:       int  = -1
        self.maker_team:  int  = -1
        self.loner:       bool = False
        self.tricks_won        = [0, 0]
        self.current_trick: List[Optional[int]] = [None]*4
        self.led_suit:    int  = -1
        self.trick_leader:int  = -1
        self.phase:       str  = 'deal'
        self.bid_passes:  int  = 0
        self.played_cards: List[int] = []

    def new_hand(self):
        self._reset_hand()
        self.dealer      = (self.dealer + 1) % 4
        self.hands, self.upcard = deal_hands()
        self.trick_leader = (self.dealer + 1) % 4
        self.phase        = 'bid1'

    # ── Bid order helpers ──────────────────────────────────────────

    def bid1_order(self) -> List[int]:
        start = (self.dealer + 1) % 4
        return [(start + i) % 4 for i in range(4)]

    def bid2_order(self) -> List[int]:
        return self.bid1_order()

    # ── Bidding ────────────────────────────────────────────────────

    def accept_upcard(self, player: int, alone: bool = False):
        """Player orders dealer to pick up the upcard (round 1)."""
        trump = card_suit(self.upcard)
        self.trump      = trump
        self.maker      = player
        self.maker_team = player % 2
        self.loner      = alone
        # Dealer picks up, discards worst non-trump (or lowest trump)
        hand = self.hands[self.dealer]
        hand.append(self.upcard)
        discard = self._auto_discard(hand, trump)
        hand.remove(discard)
        self.phase = 'play'
        if alone:
            self.hands[(player + 2) % 4] = []

    def _auto_discard(self, hand: List[int], trump: int) -> int:
        non_trump = [c for c in hand if effective_suit(c, trump) != trump]
        pool = non_trump if non_trump else hand
        return min(pool, key=card_rank)

    def call_trump(self, player: int, suit: int, alone: bool = False):
        """Player names trump in round 2 (must differ from upcard suit)."""
        assert suit != card_suit(self.upcard)
        self.trump      = suit
        self.maker      = player
        self.maker_team = player % 2
        self.loner      = alone
        self.phase      = 'play'
        if alone:
            self.hands[(player + 2) % 4] = []

    def pass_bid(self) -> bool:
        """
        Record a pass. Returns True if hand must be scrapped (all passed r2).
        Transitions bid1→bid2 automatically after 4 passes in round 1.
        """
        self.bid_passes += 1
        if self.phase == 'bid1' and self.bid_passes == 4:
            self.phase      = 'bid2'
            self.bid_passes = 0
            return False
        if self.phase == 'bid2' and self.bid_passes == 4:
            return True
        return False

    # ── Playing ────────────────────────────────────────────────────

    def _active_players(self) -> List[int]:
        if not self.loner:
            return [0, 1, 2, 3]
        partner = (self.maker + 2) % 4
        return [p for p in range(4) if p != partner]

    def legal_plays(self, player: int) -> List[int]:
        hand = self.hands[player]
        if self.led_suit == -1:
            return hand[:]
        must = [c for c in hand if effective_suit(c, self.trump) == self.led_suit]
        return must if must else hand[:]

    def play_card(self, player: int, card: int):
        assert card in self.hands[player]
        assert card in self.legal_plays(player)
        self.hands[player].remove(card)
        self.current_trick[player] = card
        self.played_cards.append(card)
        if self.led_suit == -1:
            self.led_suit = effective_suit(card, self.trump)

    def trick_complete(self) -> bool:
        active = self._active_players()
        return all(self.current_trick[p] is not None for p in active)

    def resolve_trick(self) -> int:
        """Determine winner, update state, return winning player index."""
        active = self._active_players()
        best_p = active[0]
        best_pw = card_power(self.current_trick[active[0]], self.trump, self.led_suit)
        for p in active[1:]:
            pw = card_power(self.current_trick[p], self.trump, self.led_suit)
            if pw > best_pw:
                best_pw, best_p = pw, p
        self.tricks_won[best_p % 2] += 1
        self.trick_leader = best_p
        self.current_trick = [None]*4
        self.led_suit = -1
        if all(len(h) == 0 for h in self.hands):
            self.phase = 'score'
        return best_p

    def score_hand(self) -> Tuple[int, int]:
        """Score completed hand. Returns (delta_A, delta_B)."""
        mk  = self.maker_team
        df  = 1 - mk
        twm = self.tricks_won[mk]
        twd = self.tricks_won[df]
        da = db = 0
        if twd >= 3:                         # euchred
            if df == 0: da = 2
            else:       db = 2
        elif twm == 5:                       # march
            pts = 4 if self.loner else 2
            if mk == 0: da = pts
            else:       db = pts
        else:                                # 3 or 4 tricks
            if mk == 0: da = 1
            else:       db = 1
        self.score[0] += da
        self.score[1] += db
        return da, db

    def game_over(self) -> Optional[int]:
        """Returns winning team index or None."""
        if self.score[0] >= WINNING_SCORE: return 0
        if self.score[1] >= WINNING_SCORE: return 1
        return None

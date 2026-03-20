"""
augment.py  —  Suit Symmetry Augmentation
==========================================
Euchre is symmetric under any permutation of suits that preserves
same-color pairing (black↔black, red↔red).  There are exactly 8
such permutations, forming the Klein four-group × Z2:

  B  = swap black suits (♣↔♠)
  R  = swap red suits   (♦↔♥)
  X  = swap colors      (♣↔♦, ♠↔♥)

All 8 = {I, B, R, BR, X, XB, XR, XBR}

Each permutation is represented as a suit map: new_suit = MAP[old_suit].
Applying a map to a card index: new_card = MAP[suit]*6 + rank.
"""

from typing import List, Tuple

# The 8 suit permutations (index = old suit, value = new suit)
# Format: (name, [♣→, ♦→, ♥→, ♠→])
AUGMENTATIONS: List[Tuple[str, List[int]]] = [
    ('I',   [0, 1, 2, 3]),  # identity
    ('B',   [3, 1, 2, 0]),  # swap black ♣↔♠
    ('R',   [0, 2, 1, 3]),  # swap red   ♦↔♥
    ('BR',  [3, 2, 1, 0]),  # swap both  ♣↔♠ and ♦↔♥
    ('X',   [1, 0, 3, 2]),  # swap colors ♣↔♦, ♠↔♥
    ('XB',  [2, 0, 3, 1]),  # X then B
    ('XR',  [1, 3, 0, 2]),  # X then R
    ('XBR', [2, 3, 0, 1]),  # X then B then R
]

# Precompute full card remapping tables (card index → new card index)
# _REMAP[aug_idx][old_card] = new_card
_REMAP: List[List[int]] = []
for _, suit_map in AUGMENTATIONS:
    table = [0] * 24
    for old_card in range(24):
        old_suit = old_card // 6
        rank     = old_card % 6
        new_suit = suit_map[old_suit]
        table[old_card] = new_suit * 6 + rank
    _REMAP.append(table)


def remap_card(card: int, aug_idx: int) -> int:
    """Remap a single card index under augmentation aug_idx."""
    return _REMAP[aug_idx][card]


def remap_cards(cards: list, aug_idx: int) -> list:
    """Remap a list of card indices."""
    t = _REMAP[aug_idx]
    return [t[c] for c in cards]


def remap_suit(suit: int, aug_idx: int) -> int:
    """Remap a suit index under augmentation aug_idx."""
    return AUGMENTATIONS[aug_idx][1][suit]


def augment_bid_state(state_vec: list, aug_idx: int) -> list:
    """
    Apply suit permutation aug_idx to a bid state vector.

    Bid state layout (95 dims):
      [0:24]   hand cards (one-hot)
      [24:48]  upcard (one-hot)
      [48:50]  scores (unchanged)
      [50:54]  position (unchanged)
      [54:58]  bid round info (unchanged — upcard_suit_norm needs remap)
      [58:82]  played cards (one-hot)
      [82:90]  hand strength features
      [90:95]  void suits + ace count
    """
    if aug_idx == 0:
        return state_vec  # identity — no copy needed

    t = _REMAP[aug_idx]
    sm = AUGMENTATIONS[aug_idx][1]
    v = list(state_vec)

    # Remap one-hot card blocks
    for block_start in (0, 24, 58):
        block = [0.0] * 24
        for old_card in range(24):
            if v[block_start + old_card]:
                block[t[old_card]] = v[block_start + old_card]
        v[block_start:block_start + 24] = block

    # Remap upcard suit norm (index 57): upcard_suit / 3
    v[57] = sm[round(v[57] * 3)] / 3.0

    # Remap void suits block [90:94]: one bit per suit ♣♦♥♠
    old_voids = v[90:94]
    new_voids  = [0.0] * 4
    for old_suit in range(4):
        new_voids[sm[old_suit]] = old_voids[old_suit]
    v[90:94] = new_voids

    # Hand strength features [82:90]: values are counts/flags, not suit-indexed
    # trump_count, has_right, has_left, strong_enough are all relative to the
    # (already remapped) upcard suit — they remain valid without change.

    return v


def augment_play_state(state_vec: list, aug_idx: int) -> list:
    """
    Apply suit permutation aug_idx to a play state vector.

    Play state layout (136 dims):
      [0:24]    hand cards (one-hot)
      [24:48]   trump suit indicator (one-hot over 24 cards)
      [48:50]   scores (unchanged)
      [50:54]   trick position (unchanged)
      [54:64]   trick situation counts (unchanged — counts are suit-independent)
      [64:88]   played cards history (one-hot)
      [88:112]  current trick (one-hot)
      [112:136] partner's card (one-hot)
    """
    if aug_idx == 0:
        return state_vec

    t = _REMAP[aug_idx]
    v = list(state_vec)

    for block_start in (0, 24, 64, 88, 112):
        block = [0.0] * 24
        for old_card in range(24):
            if v[block_start + old_card]:
                block[t[old_card]] = v[block_start + old_card]
        v[block_start:block_start + 24] = block

    return v


def all_augmented_bid_states(state_vec: list) -> list:
    """Return all 8 augmented versions of a bid state vector."""
    return [augment_bid_state(state_vec, i) for i in range(8)]


def all_augmented_play_states(state_vec: list) -> list:
    """Return all 8 augmented versions of a play state vector."""
    return [augment_play_state(state_vec, i) for i in range(8)]

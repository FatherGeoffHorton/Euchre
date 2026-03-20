"""
train.py  —  Training Loop
===========================
Modes (--train-net):
  bid-imitate   supervised imitation of heuristic bidding  (8× augmented)
  play-imitate  supervised imitation of heuristic card play (8× augmented)
  play-pure     REINFORCE on random trump hands
  bid           REINFORCE, bid net only (heuristic plays)
  play          REINFORCE, play net only
  both          REINFORCE, both networks

Recommended curriculum:
  1. bid-imitate   ~20k episodes
  2. play-imitate  ~20k episodes
  3. both vs heuristic  ~100k episodes
  4. both vs self       ~100k episodes
"""

import os, sys, json, time, random, shutil
from collections import deque
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from euchre_game import (
    EuchreGame, deal_hands, card_suit, card_rank, effective_suit,
    SAME_COLOR, NUM_CARDS, WINNING_SCORE, card_long_name
)
from state_encoder import encode_bid_state, encode_play_state, BID_STATE_DIM, PLAY_STATE_DIM
from models import BidNet, PlayNet, infer_config
from agent import EuchreAgent, HeuristicAgent, score_aggression
from augment import all_augmented_bid_states, all_augmented_play_states, AUGMENTATIONS

# ── Checkpoint helpers ─────────────────────────────────────────────

CKPT_CONFIG = {'BID_STATE_DIM': BID_STATE_DIM, 'PLAY_STATE_DIM': PLAY_STATE_DIM}

def autosave(step):
    # Copy the latest checkpoint file
    src_file = os.path.join(CHECKPOINT_SRC, "model.ckpt")
    dst_file = os.path.join(CHECKPOINT_DST, f"model_step_{step}.ckpt")
    shutil.copy(src_file, dst_file)
    print(f"Saved checkpoint to Drive at step {step}")

def save_checkpoint(bid_net, play_net, episode, stats, save_path, tag=None):
    os.makedirs(save_path, exist_ok=True)
    weights = {'bid_net': bid_net.state_dict(), 'play_net': play_net.state_dict()}
    meta    = {'episode': episode, 'stats': stats,
               'config': {'hidden': list(bid_net.parameters())[0].shape[0] if False else
                          bid_net.net.embed[0].weight.shape[0],
                          **CKPT_CONFIG}}
    # Fix hidden extraction
    meta['config']['hidden'] = bid_net.net.embed[0].weight.shape[0]
    meta['config']['n_res']  = len(bid_net.net.blocks)

    ckpt = os.path.join(save_path, 'latest.pt')
    torch.save(weights, ckpt)
    autosave(episode)
    with open(ckpt + '.json', 'w') as f:
        json.dump(meta, f)
    if tag:
        numbered = os.path.join(save_path, f'ep_{tag:06d}.pt')
        torch.save(weights, numbered)

def load_checkpoint(save_path, device, hidden=256, n_res=4, resume=True):
    """
    Load latest checkpoint. Returns (bid_net, play_net, start_episode, stats).
    If resume=False, always starts fresh (ignores existing checkpoint).
    If checkpoint is incompatible or missing, returns fresh networks at episode 0.
    """
    ckpt_path = os.path.join(save_path, 'latest.pt')
    meta_path = ckpt_path + '.json'
    stats_default = {'bid_losses':[], 'play_losses':[], 'win_rates':[],
                     'eval_win_rates':[], 'mean_rewards':[]}

    if not resume:
        print("Starting fresh (--no-resume).")
        if os.path.exists(meta_path):
            os.remove(meta_path)
        bid_net  = BidNet(hidden=hidden,  n_res=n_res).to(device)
        play_net = PlayNet(hidden=hidden, n_res=n_res).to(device)
        return bid_net, play_net, 0, {k: [] for k in stats_default}

    if os.path.exists(ckpt_path):
        weights = torch.load(ckpt_path, map_location=device, weights_only=True)
        try:
            cfg = infer_config(weights)
            hidden, n_res = cfg['hidden'], cfg['n_res']
            bid_net  = BidNet(hidden=hidden,  n_res=n_res).to(device)
            play_net = PlayNet(hidden=hidden, n_res=n_res).to(device)
            bid_net.load_state_dict(weights['bid_net'])
            play_net.load_state_dict(weights['play_net'])
            episode = 0
            stats   = stats_default
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                episode = meta.get('episode', 0)
                loaded  = meta.get('stats', {})
            for k in stats_default:
                stats[k] = loaded.get(k, stats_default[k])
            print(f"Resumed from episode {episode}  (hidden={hidden} n_res={n_res})")
            return bid_net, play_net, episode, stats
        except ValueError as e:
            print(f"  Checkpoint incompatible: {e}")
            print("  Starting fresh.")
    else:
        print("No checkpoint found. Starting fresh.")

    # Fresh networks
    if os.path.exists(meta_path):
        os.remove(meta_path)
    bid_net  = BidNet(hidden=hidden,  n_res=n_res).to(device)
    play_net = PlayNet(hidden=hidden, n_res=n_res).to(device)
    return bid_net, play_net, 0, stats_default


def _freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)

def _unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def _make_opt(net, lr):
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        dummy = nn.Parameter(torch.zeros(1))
        return optim.Adam([dummy], lr=lr)
    return optim.Adam(params, lr=lr)


# ── Quick eval vs heuristic ────────────────────────────────────────

def quick_eval(bid_net, play_net, n_games: int, device) -> float:
    """Win rate vs heuristic on CPU. Uses BytesIO to guarantee CPU tensors."""
    import io as _io
    cpu = torch.device('cpu')
    # Serialise through a buffer so map_location=cpu handles device conversion
    buf = _io.BytesIO()
    torch.save({'bid_net': bid_net.state_dict(),
                'play_net': play_net.state_dict()}, buf)
    buf.seek(0)
    weights = torch.load(buf, map_location=cpu, weights_only=True)
    eb = BidNet(hidden=bid_net.net.embed[0].weight.shape[0],
                n_res=len(bid_net.net.blocks))
    ep = PlayNet(hidden=play_net.net.embed[0].weight.shape[0],
                 n_res=len(play_net.net.blocks))
    eb.load_state_dict(weights['bid_net'])
    ep.load_state_dict(weights['play_net'])
    agent     = EuchreAgent(eb, ep, device=cpu)
    agent.eval()
    heuristic = HeuristicAgent()
    wins = 0
    with torch.no_grad():
        for _ in range(n_games):
            game = EuchreGame()
            while game.game_over() is None:
                game.new_hand()
                _run_hand_heuristic(game, agent, heuristic)
            if game.game_over() == 0:
                wins += 1
    return wins / n_games


def _run_hand_heuristic(game, agent, heuristic):
    """Play one hand: agent=team0, heuristic=team1. No gradient."""
    # Bidding
    trump_called = False
    for player in game.bid1_order():
        if player % 2 == 0:
            order, alone = agent.bid_round1(game, player)
        else:
            order, alone = heuristic.bid_round1(game, player)
        if order:
            game.accept_upcard(player, alone=alone)
            trump_called = True
            break
        else:
            if game.pass_bid():
                break
    if not trump_called and game.phase == 'bid2':
        for player in game.bid2_order():
            if player % 2 == 0:
                suit, alone = agent.bid_round2(game, player)
            else:
                suit, alone = heuristic.bid_round2(game, player)
            if suit is not None:
                game.call_trump(player, suit, alone=alone)
                trump_called = True
                break
            else:
                if game.pass_bid():
                    break
    if game.phase != 'play':
        return
    # Play
    for _ in range(5):
        active = game._active_players()
        leader = game.trick_leader
        if leader not in active:
            leader = active[0]
        idx   = active.index(leader)
        order = [active[(idx+i)%len(active)] for i in range(len(active))]
        for p in order:
            if not game.hands[p]:
                continue
            if p % 2 == 0:
                card = agent.play_card(game, p)
            else:
                card = heuristic.play_card(game, p)
            game.play_card(p, card)
        if game.trick_complete():
            game.resolve_trick()
        if game.phase == 'score':
            break
    game.score_hand()


# ── Log helper ────────────────────────────────────────────────────

def _log(ep_rel, n_episodes, ep_abs, metrics: dict, elapsed):
    ts   = datetime.now().strftime('%H:%M:%S')
    main = (f"[{ts}] Ep {ep_rel:6d}/{n_episodes} (abs {ep_abs})")
    for k, v in metrics.items():
        if isinstance(v, float):
            main += f" | {k}={v:.4f}" if abs(v) < 10 else f" | {k}={v:.3f}"
        else:
            main += f" | {k}={v}"
    main += f" | {elapsed:.1f}s"
    print(main)


# ══════════════════════════════════════════════════════════════════
# BID IMITATION
# ══════════════════════════════════════════════════════════════════

def train_bid_imitate(
    n_episodes:  int   = 20000,
    batch_size:  int   = 512,
    save_every:  int   = 2000,
    save_path:   str   = 'checkpoints',
    resume:      bool  = True,
    hidden:      int   = 256,
    n_res:       int   = 4,
    lr:          float = 1e-3,
    eval_games:  int   = 0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bid imitation  [{device}]  augmentation=8×")

    bid_net, play_net, start_ep, stats = load_checkpoint(
        save_path, device, hidden, n_res, resume=resume)

    _freeze(play_net)
    _unfreeze(bid_net)
    opt   = optim.Adam(bid_net.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, max(n_episodes,1), eta_min=1e-5)
    # Equal weight — oversampling handles the class imbalance
    ce    = nn.CrossEntropyLoss()

    heuristic    = HeuristicAgent()
    end_ep       = start_ep + n_episodes
    recent_acc   = deque(maxlen=500)
    recent_loss  = deque(maxlen=100)
    t0           = time.time()

    for ep in range(start_ep, end_ep, batch_size):
        raw_states:  List[List[float]] = []
        raw_actions: List[int]         = []

        for _ in range(batch_size):
            game = EuchreGame()
            game.new_hand()
            for player in game.bid1_order():
                state = encode_bid_state(game, player).tolist()
                order, alone = heuristic.bid_round1(game, player)
                action = (2 if alone else 1) if order else 0
                raw_states.append(state)
                raw_actions.append(action)
                if order:
                    game.accept_upcard(player, alone=alone)
                    break
                else:
                    if game.pass_bid():
                        break
            if game.phase == 'bid2':
                for player in game.bid2_order():
                    state = encode_bid_state(game, player).tolist()
                    suit, alone = heuristic.bid_round2(game, player)
                    action = (2 if alone else 1) if suit is not None else 0
                    raw_states.append(state)
                    raw_actions.append(action)
                    if suit is not None:
                        game.call_trump(player, suit, alone=alone)
                        break
                    else:
                        if game.pass_bid():
                            break

        # Balance by subsampling passes to match bid count
        bid_idx  = [i for i,a in enumerate(raw_actions) if a > 0]
        pass_idx = [i for i,a in enumerate(raw_actions) if a == 0]
        if bid_idx and pass_idx:
            keep = random.sample(pass_idx, min(len(pass_idx), len(bid_idx)))
            balanced = sorted(keep + bid_idx)
            raw_states  = [raw_states[i]  for i in balanced]
            raw_actions = [raw_actions[i] for i in balanced]

        # 8× augmentation
        aug_states:  List[List[float]] = []
        aug_actions: List[int]         = []
        for state, action in zip(raw_states, raw_actions):
            for aug_idx in range(8):
                aug_states.append(
                    all_augmented_bid_states(state)[aug_idx])
                aug_actions.append(action)  # action label is suit-invariant

        if not aug_states:
            continue

        states_t  = torch.tensor(aug_states,  dtype=torch.float32, device=device)
        actions_t = torch.tensor(aug_actions, dtype=torch.long,    device=device)

        states_t  = states_t.to(device)
        actions_t = actions_t.to(device)

        logits = bid_net(states_t)
        loss   = ce(logits, actions_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bid_net.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            pred = logits.argmax(1)
        recent_acc.append((pred == actions_t).float().mean().item())
        recent_loss.append(loss.item())

        if ep % 200 == 0:
            avg_acc  = sum(recent_acc)  / max(len(recent_acc),  1)
            avg_loss = sum(recent_loss) / max(len(recent_loss), 1)
            metrics  = {'Acc': avg_acc, 'Loss': avg_loss}
            if eval_games > 0:
                with torch.no_grad():
                    metrics['EvalVsHeur'] = quick_eval(bid_net, play_net, eval_games, device)
            _log(ep - start_ep, n_episodes, ep, metrics, time.time()-t0)
            stats['bid_losses'].append(avg_loss)

        if ep % save_every == 0 or ep + batch_size >= end_ep:
            save_checkpoint(bid_net, play_net, ep + batch_size, stats, save_path, ep)
            print(f"  ✓ Saved ep {ep}")

    print(f"\nBid imitation complete. Final accuracy: "
          f"{sum(recent_acc)/max(len(recent_acc),1):.1%}")


# ══════════════════════════════════════════════════════════════════
# PLAY IMITATION
# ══════════════════════════════════════════════════════════════════

def train_play_imitate(
    n_episodes: int   = 20000,
    batch_size: int   = 512,
    save_every: int   = 2000,
    save_path:  str   = 'checkpoints',
    resume:     bool  = True,
    hidden:     int   = 256,
    n_res:      int   = 4,
    lr:         float = 1e-3,
    eval_games: int   = 0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Play imitation  [{device}]  augmentation=8×")

    bid_net, play_net, start_ep, stats = load_checkpoint(
        save_path, device, hidden, n_res, resume=resume)

    _freeze(bid_net)
    _unfreeze(play_net)
    opt   = optim.Adam(play_net.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, max(n_episodes,1), eta_min=1e-5)
    ce    = nn.CrossEntropyLoss()

    heuristic   = HeuristicAgent()
    end_ep      = start_ep + n_episodes
    recent_acc  = deque(maxlen=500)
    recent_loss = deque(maxlen=100)
    t0          = time.time()

    for ep in range(start_ep, end_ep, batch_size):
        raw_states:  List[List[float]] = []
        raw_actions: List[int]         = []
        raw_masks:   List[List[bool]]  = []

        for _ in range(batch_size):
            game = EuchreGame()
            game.hands, game.upcard = deal_hands()
            game.trump       = random.randint(0, 3)
            game.maker       = 0
            game.maker_team  = 0
            game.loner       = False
            game.tricks_won  = [0, 0]
            game.current_trick = [None]*4
            game.led_suit    = -1
            game.trick_leader = 0
            game.played_cards = []
            game.phase       = 'play'
            game.score       = [0, 0]
            game.dealer      = 3

            for _ in range(5):
                active = game._active_players()
                leader = game.trick_leader
                if leader not in active:
                    leader = active[0]
                idx   = active.index(leader)
                order = [active[(idx+i)%len(active)] for i in range(len(active))]
                for p in order:
                    if not game.hands[p]:
                        continue
                    legal  = game.legal_plays(p)
                    state  = encode_play_state(game, p).tolist()
                    action = heuristic.play_card(game, p)
                    mask   = [c in legal for c in range(NUM_CARDS)]
                    raw_states.append(state)
                    raw_actions.append(action)
                    raw_masks.append(mask)
                    game.play_card(p, action)
                if game.trick_complete():
                    game.resolve_trick()
                if game.phase == 'score':
                    break

        # 8× augmentation
        aug_states:  List[List[float]] = []
        aug_actions: List[int]         = []
        aug_masks:   List[List[bool]]  = []
        from augment import augment_play_state, _REMAP
        for state, action, mask in zip(raw_states, raw_actions, raw_masks):
            for aug_idx in range(8):
                aug_states.append(augment_play_state(state, aug_idx))
                # Remap action card index
                aug_actions.append(_REMAP[aug_idx][action])
                # Remap legal mask
                new_mask = [False] * NUM_CARDS
                t = _REMAP[aug_idx]
                for old_c, legal in enumerate(mask):
                    if legal:
                        new_mask[t[old_c]] = True
                aug_masks.append(new_mask)

        states_t  = torch.tensor(aug_states,  dtype=torch.float32, device=device)
        actions_t = torch.tensor(aug_actions, dtype=torch.long,    device=device)
        masks_t   = torch.tensor(aug_masks,   dtype=torch.bool,    device=device)

        # Ensure all tensors are on the correct device
        states_t  = states_t.to(device)
        actions_t = actions_t.to(device)
        masks_t   = masks_t.to(device)

        logits = play_net(states_t)
        logits = logits.masked_fill(~masks_t, float('-inf'))
        loss   = ce(logits, actions_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(play_net.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            pred = logits.argmax(1)
        recent_acc.append((pred == actions_t).float().mean().item())
        recent_loss.append(loss.item())

        if ep % 200 == 0:
            avg_acc  = sum(recent_acc)  / max(len(recent_acc),  1)
            avg_loss = sum(recent_loss) / max(len(recent_loss), 1)
            metrics  = {'Acc': avg_acc, 'Loss': avg_loss}
            if eval_games > 0:
                with torch.no_grad():
                    metrics['EvalVsHeur'] = quick_eval(bid_net, play_net, eval_games, device)
            _log(ep - start_ep, n_episodes, ep, metrics, time.time()-t0)
            stats['play_losses'].append(avg_loss)

        if ep % save_every == 0 or ep + batch_size >= end_ep:
            save_checkpoint(bid_net, play_net, ep + batch_size, stats, save_path, ep)
            print(f"  ✓ Saved ep {ep}")

    print(f"\nPlay imitation complete. Final accuracy: "
          f"{sum(recent_acc)/max(len(recent_acc),1):.1%}")


# ══════════════════════════════════════════════════════════════════
# REINFORCE (bid / play / both, vs heuristic or self)
# ══════════════════════════════════════════════════════════════════

class Episode:
    def __init__(self):
        self.bid_exps:  List[Dict] = []   # {state, action, reward}
        self.play_exps: List[Dict] = []   # {state, action, mask}
        self._pending:  List[Dict] = []

    def add_bid(self, state, action):
        self._pending.append({'state': state, 'action': action})

    def close_hand(self, reward: float):
        for exp in self._pending:
            exp['reward'] = reward
            self.bid_exps.append(exp)
        self._pending = []

    def add_play(self, state, action, mask):
        self.play_exps.append({'state': state, 'action': action, 'mask': mask})


def hand_reward(game: EuchreGame, my_team: int) -> float:
    mk = game.maker_team
    df = 1 - mk
    twm, twd = game.tricks_won[mk], game.tricks_won[df]
    i_am_maker = (my_team == mk)
    if twd >= 3:          return  2.0 if not i_am_maker else -2.0
    elif twm == 5:
        pts = 4.0 if game.loner else 2.0
        return pts if i_am_maker else -pts
    else:                 return  1.0 if i_am_maker else -1.0


class SelfPlayRunner:
    def __init__(self, agent_A: EuchreAgent, agent_B,
                 train_net: str = 'both', training_team: int = 0):
        self.agent_A       = agent_A
        self.agent_B       = agent_B
        self.train_net     = train_net
        self.training_team = training_team
        self.heuristic     = HeuristicAgent()

    def _get_agent(self, player):
        return self.agent_A if player % 2 == self.training_team else self.agent_B

    def run_episode(self) -> Tuple[EuchreGame, Episode, float]:
        game    = EuchreGame()
        episode = Episode()
        while game.game_over() is None:
            game.new_hand()
            self._run_hand(game, episode)
        winner = game.game_over()
        reward = 1.0 if winner == self.training_team else -1.0
        margin = abs(game.score[0] - game.score[1])
        reward *= (1.0 + margin / WINNING_SCORE)
        return game, episode, reward

    def _neural(self, agent) -> bool:
        return hasattr(agent, 'bid_net')

    def _run_hand(self, game: EuchreGame, episode: Episode):
        record_bid  = self.train_net in ('bid',  'both')
        record_play = self.train_net in ('play', 'both')

        # ── Bidding ──
        trump_called = False
        for player in game.bid1_order():
            agent = self._get_agent(player)
            is_training = (player % 2 == self.training_team)

            if self._neural(agent):
                with torch.no_grad():
                    state = encode_bid_state(game, player).to(agent.device)
                    logits = agent.bid_net(state.unsqueeze(0)).squeeze(0)
                    agg = score_aggression(game, player % 2)
                    sc  = logits.clone()
                    sc[1] *= agg; sc[2] *= agg * 0.7
                    probs  = F.softmax(sc / agent.temperature, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                if is_training and record_bid:
                    episode.add_bid(state.cpu(), action)
            else:
                order, alone = agent.bid_round1(game, player)
                action = (2 if alone else 1) if order else 0

            if action == 0:
                if game.pass_bid():
                    episode.close_hand(0.0)
                    return
            else:
                game.accept_upcard(player, alone=(action==2))
                trump_called = True
                break

        if not trump_called and game.phase == 'bid2':
            for player in game.bid2_order():
                agent = self._get_agent(player)
                is_training = (player % 2 == self.training_team)

                if self._neural(agent):
                    with torch.no_grad():
                        state  = encode_bid_state(game, player).to(agent.device)
                        logits = agent.bid_net(state.unsqueeze(0)).squeeze(0)
                        agg = score_aggression(game, player % 2)
                        sc  = logits.clone()
                        sc[1] *= agg*0.85; sc[2] *= agg*0.6
                        probs  = F.softmax(sc / agent.temperature, dim=-1)
                        action = torch.multinomial(probs, 1).item()
                    if is_training and record_bid:
                        episode.add_bid(state.cpu(), action)
                else:
                    suit, alone = agent.bid_round2(game, player)
                    action = (2 if alone else 1) if suit is not None else 0

                if action == 0:
                    if game.pass_bid():
                        episode.close_hand(0.0)
                        return
                else:
                    # Determine suit for neural agent round 2
                    if self._neural(agent):
                        suit, alone = agent.bid_round2(game, player)
                    if suit is not None:
                        game.call_trump(player, suit, alone=alone)
                        trump_called = True
                        break
                    else:
                        if game.pass_bid():
                            episode.close_hand(0.0)
                            return

        if game.phase != 'play':
            return

        # ── Play ──
        for _ in range(5):
            active = game._active_players()
            leader = game.trick_leader
            if leader not in active:
                leader = active[0]
            idx   = active.index(leader)
            order = [active[(idx+i)%len(active)] for i in range(len(active))]

            for player in order:
                if not game.hands[player]:
                    continue
                legal = game.legal_plays(player)
                state = encode_play_state(game, player).to(self.agent_A.device)
                mask  = torch.zeros(NUM_CARDS, dtype=torch.bool)
                for c in legal:
                    mask[c] = True

                # Always use heuristic for rollout actions;
                # net is trained on stored states in update()
                card = self.heuristic.play_card(game, player)

                if player % 2 == self.training_team and record_play:
                    episode.add_play(state.cpu(), card, mask.cpu())

                game.play_card(player, card)

            if game.trick_complete():
                game.resolve_trick()
            if game.phase == 'score':
                break

        hr = hand_reward(game, self.training_team)
        game.score_hand()
        episode.close_hand(hr)


class Trainer:
    def __init__(self, bid_net, play_net, lr=3e-4, gamma=0.99,
                 entropy_coef=0.01, device=torch.device('cpu'),
                 train_net='both'):
        self.bid_net      = bid_net
        self.play_net     = play_net
        self.device       = device
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.train_net    = train_net
        self.play_net_temp = 1.0

        # Clean freeze/unfreeze
        _unfreeze(bid_net); _unfreeze(play_net)
        if train_net == 'bid':  _freeze(play_net)
        if train_net == 'play': _freeze(bid_net)

        self.bid_opt  = _make_opt(bid_net,  lr)
        self.play_opt = _make_opt(play_net, lr)
        self.bid_sched  = optim.lr_scheduler.CosineAnnealingLR(
            self.bid_opt,  10000, eta_min=1e-5)
        self.play_sched = optim.lr_scheduler.CosineAnnealingLR(
            self.play_opt, 10000, eta_min=1e-5)

        self.episode_rewards = deque(maxlen=500)
        self.win_rates       = deque(maxlen=500)

    def _returns(self, rewards):
        R, out = 0.0, []
        for r in reversed(rewards):
            R = r + self.gamma * R
            out.insert(0, R)
        t = torch.tensor(out, dtype=torch.float32)
        if len(t) > 1:
            std = t.std()
            if std > 1e-6:   # only normalise when there's actual variance
                t = (t - t.mean()) / (std + 1e-8)
        t = t.clamp(-2.0, 2.0)
        return t

    def update(self, episodes: List[Tuple[Episode, float]]):
        train_bid  = self.train_net in ('bid',  'both')
        train_play = self.train_net in ('play', 'both')
        n = len(episodes)
        bid_loss_total = play_loss_total = 0.0

        if train_bid:  self.bid_opt.zero_grad()
        if train_play: self.play_opt.zero_grad()

        for episode, final_reward in episodes:
            self.episode_rewards.append(final_reward)
            self.win_rates.append(1.0 if final_reward > 0 else 0.0)

            if train_bid and episode.bid_exps:
                states  = torch.stack([e['state'] for e in episode.bid_exps]).to(self.device)
                actions = torch.tensor([e['action'] for e in episode.bid_exps],
                                       dtype=torch.long, device=self.device)
                hand_rw = torch.tensor([e['reward'] for e in episode.bid_exps],
                                       dtype=torch.float32, device=self.device)
                blended = 0.7 * hand_rw + 0.3 * final_reward
                if len(blended) > 1:
                    blended = (blended - blended.mean()) / (blended.std() + 1e-8)

                logits   = self.bid_net(states)
                lp_all   = F.log_softmax(logits, dim=-1)
                probs    = lp_all.exp()
                lp       = lp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
                entropy  = -(probs * (probs+1e-8).log()).sum(-1).mean()
                loss     = (-(lp * blended).mean() - self.entropy_coef*entropy) / n
                loss.backward()
                bid_loss_total += loss.item() * n

            if train_play and episode.play_exps:
                # Accumulate across batch for normalisation
                for e in episode.play_exps:
                    e['_reward'] = final_reward

        # Batch-level play update — normalise returns across all episodes
        if train_play:
            all_play = [e for ep, _ in episodes for e in ep.play_exps
                        if '_reward' in e]
            if all_play:
                states  = torch.stack([e['state']  for e in all_play]).to(self.device)
                actions = torch.tensor([e['action'] for e in all_play],
                                       dtype=torch.long, device=self.device)
                masks   = torch.stack([e['mask']   for e in all_play]).to(self.device)
                rewards = torch.tensor([e['_reward'] for e in all_play],
                                       dtype=torch.float32)
                # Normalise across full batch
                if rewards.std() > 1e-6:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                rewards = rewards.clamp(-2.0, 2.0).to(self.device)

                logits  = self.play_net(states).masked_fill(~masks, float('-inf'))
                lp_all  = F.log_softmax(logits / self.play_net_temp, dim=-1)
                probs   = lp_all.exp()
                lp      = lp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
                entropy = -(probs * (probs+1e-8).log()).sum(-1).mean()
                loss    = -(lp * rewards).mean() - self.entropy_coef * entropy
                loss.backward()
                play_loss_total = loss.item()

        if train_bid:
            torch.nn.utils.clip_grad_norm_(self.bid_net.parameters(), 1.0)
            self.bid_opt.step(); self.bid_sched.step()
        if train_play:
            torch.nn.utils.clip_grad_norm_(self.play_net.parameters(), 1.0)
            self.play_opt.step(); self.play_sched.step()

        return bid_loss_total / n, play_loss_total / n

    @property
    def win_rate(self):
        return sum(self.win_rates) / max(len(self.win_rates), 1)


def train(
    n_episodes:        int   = 100000,
    batch_size:        int   = 128,
    save_every:        int   = 4000,
    save_path:         str   = 'checkpoints',
    opponent:          str   = 'heuristic',
    resume:            bool  = True,
    hidden:            int   = 256,
    n_res:             int   = 4,
    lr:                float = 1e-4,
    temperature_start: float = 1.5,
    temperature_end:   float = 0.7,
    train_net:         str   = 'both',
    entropy_coef:      float = 0.0,
    eval_games:        int   = 50,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"REINFORCE [{device}]  train_net={train_net}  opponent={opponent}")

    bid_net, play_net, start_ep, stats = load_checkpoint(
        save_path, device, hidden, n_res, resume=resume)

    trainer = Trainer(bid_net, play_net, lr=lr, device=device,
                      train_net=train_net, entropy_coef=entropy_coef)

    heuristic = HeuristicAgent()
    if opponent == 'heuristic':
        opp = heuristic
        best_eval_wr = 0.0
    else:
        print("  Self-play: EvalVsHeur is the metric to watch (WinRate≈50% is normal).")
        opp_bid  = BidNet(hidden=bid_net.net.embed[0].weight.shape[0],
                          n_res=len(bid_net.net.blocks)).to(device)
        opp_play = PlayNet(hidden=play_net.net.embed[0].weight.shape[0],
                           n_res=len(play_net.net.blocks)).to(device)
        opp_bid.load_state_dict(bid_net.state_dict())
        opp_play.load_state_dict(play_net.state_dict())
        opp = EuchreAgent(opp_bid, opp_play, device=device)
        opp.eval()
        best_eval_wr = 0.0
        OPP_UPDATE_EVERY = 5000

    end_ep = start_ep + n_episodes
    t0     = time.time()

    for ep in range(start_ep, end_ep, batch_size):
        frac  = (ep - start_ep) / n_episodes
        temp  = temperature_start + (temperature_end - temperature_start) * frac
        agent = EuchreAgent(bid_net, play_net, device=device, temperature=temp)
        agent.train()
        trainer.play_net_temp = temp

        runner = SelfPlayRunner(agent, opp, train_net=train_net)
        batch  = []
        wins   = 0
        for _ in range(batch_size):
            _, episode, reward = runner.run_episode()
            batch.append((episode, reward))
            if reward > 0:
                wins += 1

        bid_loss, play_loss = trainer.update(batch)

        # Self-play opponent update (only on improvement)
        if opponent == 'self' and ep % OPP_UPDATE_EVERY == 0 and ep > start_ep:
            ew = quick_eval(bid_net, play_net, 50, device)
            if ew >= best_eval_wr:
                opp_bid.load_state_dict(bid_net.state_dict())
                opp_play.load_state_dict(play_net.state_dict())
                best_eval_wr = ew
                print(f"  ↑ Opp updated  EvalVsHeur={ew:.3f}")
            else:
                print(f"  ✗ Opp held     EvalVsHeur={ew:.3f} < best {best_eval_wr:.3f}")

        if ep % 200 == 0:
            metrics = {
                'WinRate': trainer.win_rate,
                'BidLoss': bid_loss,
                'PlayLoss': play_loss,
                'Temp': temp,
            }
            if eval_games > 0:
                metrics['EvalVsHeur'] = quick_eval(bid_net, play_net, eval_games, device)
                stats.setdefault('eval_win_rates', []).append(metrics['EvalVsHeur'])
            _log(ep - start_ep, n_episodes, ep, metrics, time.time()-t0)
            stats['bid_losses'].append(bid_loss)
            stats['play_losses'].append(play_loss)
            stats['win_rates'].append(trainer.win_rate)
            # Save a numbered snapshot at every log line so any result is recoverable
            save_checkpoint(bid_net, play_net, ep + batch_size, stats, save_path, ep)

        if ep % save_every == 0 or ep + batch_size >= end_ep:
            save_checkpoint(bid_net, play_net, ep + batch_size, stats, save_path, ep)
            print(f"  ✓ Saved ep {ep}")

    print(f"\nTraining complete.  Final WinRate={trainer.win_rate:.3f}")


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Train Euchre AI')
    p.add_argument('--train-net', choices=['bid-imitate','play-imitate',
                   'bid','play','both'], default='both')
    p.add_argument('--episodes',  type=int,   default=20000)
    p.add_argument('--batch',     type=int,   default=128)
    p.add_argument('--opponent',  choices=['heuristic','self'], default='heuristic')
    p.add_argument('--save-path', default='checkpoints')
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--lr',        type=float, default=None)
    p.add_argument('--hidden',    type=int,   default=256)
    p.add_argument('--n-res',     type=int,   default=4)
    p.add_argument('--entropy-coef', type=float, default=None)
    p.add_argument('--eval-games',   type=int,   default=50)
    a = p.parse_args()

    if a.no_resume:
        ckpt = os.path.join(a.save_path, 'latest.pt')
        if os.path.exists(ckpt):
            print(f"WARNING: --no-resume will discard '{ckpt}'.")
            if input("Type 'yes' to confirm: ").strip().lower() != 'yes':
                print("Aborted."); sys.exit(0)
        else:
            print("--no-resume: no existing checkpoint found, starting fresh.")

    resume = not a.no_resume
    os.makedirs(CHECKPOINT_DST, exist_ok=True)

    # Default lr and entropy per mode
    lr = a.lr or {'bid-imitate':1e-3, 'play-imitate':1e-3,
                  'bid':3e-4, 'play':3e-4, 'both':1e-4}[a.train_net]
    ec = a.entropy_coef if a.entropy_coef is not None else 0.0

    if a.train_net == 'bid-imitate':
        train_bid_imitate(n_episodes=a.episodes, batch_size=a.batch,
                          save_path=a.save_path, resume=resume,
                          hidden=a.hidden, n_res=a.n_res, lr=lr,
                          eval_games=a.eval_games)
    elif a.train_net == 'play-imitate':
        train_play_imitate(n_episodes=a.episodes, batch_size=a.batch,
                           save_path=a.save_path, resume=resume,
                           hidden=a.hidden, n_res=a.n_res, lr=lr,
                           eval_games=a.eval_games)
    else:
        train(n_episodes=a.episodes, batch_size=a.batch,
              save_path=a.save_path, opponent=a.opponent,
              resume=resume, hidden=a.hidden, n_res=a.n_res,
              lr=lr, train_net=a.train_net, entropy_coef=ec,
              eval_games=a.eval_games)

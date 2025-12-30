'''
    File name: /helper/agent_ranking.py
    Author: Oliver Czerwinski
    Date created: 08/18/2025
    Date last modified: 12/27/2025
    Python Version: 3.9+
'''

import os
import time
import csv
import itertools
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import torch
from safetensors.torch import load_file
import re
import rlcard
import glicko2
import multiprocessing as mp

from rlcard.agents.bauernskat.rule_agents import (
    BauernskatRandomRuleAgent,
    BauernskatFrugalRuleAgent,
    BauernskatLookaheadRuleAgent,
    BauernskatSHOTAlphaBetaRuleAgent
)

from rlcard.agents.bauernskat.dmc_agent.config import BauernskatNetConfig as DMCConfig
from rlcard.agents.bauernskat.dmc_agent.model import BauernskatNet as DMCNet
from rlcard.agents.bauernskat.dmc_agent.agent import AgentDMC_Actor as DMCActor

from rlcard.agents.bauernskat.sac_agent.config import BauernskatNetConfig as SACConfig
from rlcard.agents.bauernskat.sac_agent.model import BauernskatNet as SACNet
from rlcard.agents.bauernskat.sac_agent.agent import AgentSAC_Actor as SACActor


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

MAX_WORKERS = 8             
BATCH_SIZE = 40
TARGET_TOTAL_GAMES = 10000
MASTER_SEED = 21000
OUTPUT_DIR = "agent_ranking_results"
ELO_INIT = 1500.0

AGENTS_REGISTRY = {
    # Rule agents
    'Rule_Random': {
        'type': 'rule', 'class': BauernskatRandomRuleAgent, 'info': 'normal'},
    'Rule_Frugal': {
        'type': 'rule', 'class': BauernskatFrugalRuleAgent, 'info': 'normal'},
    'Rule_Lookahead': {
        'type': 'rule', 'class': BauernskatLookaheadRuleAgent, 'info': 'normal'},
    'Rule_SHOT-AB': {
        'type': 'rule', 'class': BauernskatSHOTAlphaBetaRuleAgent, 'info': 'normal'},

    # RL agents
    'DMC_1024M_Regular': {
        'type': 'dmc', 'path': r'pretrained/DMC_1024M_Regular.safetensors', 'info': 'normal', 'device': 'cpu'},
    'DMC_11590M_Regular': {
        'type': 'dmc', 'path': r'pretrained/DMC_11590M_Regular.safetensors', 'info': 'normal', 'device': 'cpu'},
    'DMC_1024M_Show_Self': {
        'type': 'dmc', 'path': r'pretrained/DMC_1024M_Show_Self.safetensors', 'info': 'show_self', 'device': 'cpu'},
    'DMC_1024M_Perfect': {
        'type': 'dmc', 'path': r'pretrained/DMC_1024M_Perfect.safetensors', 'info': 'perfect', 'device': 'cpu'},
    'DMC_1024M_Binary_Reward': {
        'type': 'dmc', 'path': r'pretrained/DMC_1024M_Binary_Reward.safetensors', 'info': 'normal', 'device': 'cpu'},
    'DMC_1024M_Game_Value_Reward': {
        'type': 'dmc', 'path': r'pretrained/DMC_1024M_Game_Value_Reward.safetensors', 'info': 'normal', 'device': 'cpu'},
    'DMC_1024M_Rule_Trump': {
        'type': 'dmc', 'path': r'pretrained/DMC_1024M_Rule_Trump.safetensors', 'info': 'normal', 'device': 'cpu'},
    'DMC_1024M_Teacher': {
        'type': 'dmc', 'path': r'pretrained/DMC_1024M_Teacher.safetensors', 'info': 'normal', 'device': 'cpu'},
    'SAC_640M_Regular': {
        'type': 'sac', 'path': r'pretrained/SAC_640M_Regular.safetensors', 'info': 'normal', 'device': 'cpu'},
    'SAC_1024M_Regular': {
        'type': 'sac', 'path': r'pretrained/SAC_1024M_Regular.safetensors', 'info': 'normal', 'device': 'cpu'},
    'SAC_640M_No_Entropy': {
        'type': 'sac', 'path': r'pretrained/SAC_640M_No_Entropy.safetensors', 'info': 'normal', 'device': 'cpu'},
    'SAC_1024M_No_Entropy': {
        'type': 'sac', 'path': r'pretrained/SAC_1024M_No_Entropy.safetensors', 'info': 'normal', 'device': 'cpu'},
}

_env_cache = None
_agent_cache = {}

def get_or_create_env():
    """
    Gets or creates a cached RLCard Bauernskat environment.
    """
    
    global _env_cache
    
    if _env_cache is None:
        _env_cache = rlcard.make('bauernskat', config={'information_level': 'normal'})
    
    return _env_cache

def load_rl_agent(config):
    """
    Loads a reinforcement learning agent with a configuration.
    """
    
    path = config['path']
    device = torch.device(config.get('device', 'cpu'))
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path not found: {path}")

    if path.endswith('.safetensors'):
        state_dict = load_file(path)
    else:
        data = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = data['model_state_dict'] if isinstance(data, dict) and 'model_state_dict' in data else data

    atype = config['type']
    if atype == 'dmc':
        net = DMCNet(DMCConfig())
        AgentClass = DMCActor
    elif atype == 'sac':
        net = SACNet(SACConfig())
        AgentClass = SACActor
    else:
        raise ValueError(f"Unknown RL type: {atype}")

    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()
    return AgentClass(net, device=device)

def get_agent(name):
    """
    Gets an agent instance by name.
    """
    
    global _agent_cache
    
    if name not in _agent_cache:
        conf = AGENTS_REGISTRY[name]
        if conf['type'] == 'rule':
            instance = conf['class']()
            _agent_cache[name] = instance.agents[0] if hasattr(instance, 'agents') else instance
        else:
            _agent_cache[name] = load_rl_agent(conf)
    
    return _agent_cache[name]

def worker_init():
    """
    Initializes worker process.
    """
    
    torch.set_num_threads(1)
    get_or_create_env()

def play_matchup_batch(args):
    """
    Plays a batch of games and returns the results.
    """
    
    agent_a_name, agent_b_name, num_games, seed = args
    env = get_or_create_env()
    
    agent_a = get_agent(agent_a_name)
    agent_b = get_agent(agent_b_name)
    
    info_a = AGENTS_REGISTRY[agent_a_name]['info']
    info_b = AGENTS_REGISTRY[agent_b_name]['info']
    type_a = AGENTS_REGISTRY[agent_a_name]['type']
    type_b = AGENTS_REGISTRY[agent_b_name]['type']
    
    results = {
        'a_wins': 0, 'b_wins': 0, 'draws': 0,
        'a_total_score': 0.0,
        'a_total_pips': 0.0
    }
    
    games_per_side = num_games // 2
    
    # Initial positions
    env.game.information_level = {0: info_a, 1: info_b}
    env.set_agents([agent_a, agent_b])
    
    for i in range(games_per_side):
        env.seed(seed + i)
        state, pid = env.reset()
        
        while not env.is_over():
            curr_agent = agent_a if pid == 0 else agent_b
            curr_type = type_a if pid == 0 else type_b
            
            if curr_type == 'rule':
                action = curr_agent.eval_step(state)[0] if hasattr(curr_agent, 'eval_step') else curr_agent.step(state)
            else:
                action, _ = curr_agent.eval_step(state, env)
            state, pid = env.step(action)
            
        payoffs = env.get_payoffs() # [p0, p1]
        scores = env.get_scores()   # [p0_pips, p1_pips]
        
        results['a_total_score'] += payoffs[0]
        results['a_total_pips'] += (scores[0] - scores[1])
        
        if payoffs[0] > 0: results['a_wins'] += 1
        elif payoffs[1] > 0: results['b_wins'] += 1
        else: results['draws'] += 1

    # Swapped positions
    env.game.information_level = {0: info_b, 1: info_a}
    env.set_agents([agent_b, agent_a])
    
    for i in range(games_per_side):
        env.seed(seed + games_per_side + i)
        state, pid = env.reset()
        while not env.is_over():
            curr_agent = agent_b if pid == 0 else agent_a
            curr_type = type_b if pid == 0 else type_a
            
            if curr_type == 'rule':
                action = curr_agent.eval_step(state)[0] if hasattr(curr_agent, 'eval_step') else curr_agent.step(state)
            else:
                action, _ = curr_agent.eval_step(state, env)
            state, pid = env.step(action)

        payoffs = env.get_payoffs()
        scores = env.get_scores()
        
        results['a_total_score'] += payoffs[1]
        results['a_total_pips'] += (scores[1] - scores[0])

        if payoffs[1] > 0: results['a_wins'] += 1
        elif payoffs[0] > 0: results['b_wins'] += 1
        else: results['draws'] += 1

    return (agent_a_name, agent_b_name, results)

def get_rd_color(rd):
    """
    Colors for RD based on thresholds.
    """
    
    if rd > 225: return "\033[91m"
    elif rd > 150: return "\033[93m"
    elif rd > 100: return "\033[33m"
    elif rd > 75: return "\033[92m"
    else: return "\033[94m"

def format_change(delta):
    """
    Formats rank change with symbol and color.
    """
    
    sym = ('\033[32m▲\033[0m' if delta > 0 else ('\033[31m▼\033[0m' if delta < 0 else '\033[90m-\033[0m'))
    return f"{sym} {delta:+d}".rjust(8)

def format_delta(delta):
    """
    Formats ELO delta with color.
    """
    
    num_str = f"{int(delta):+d}"
    color = "\033[32m" if delta > 0 else ("\033[31m" if delta < 0 else "\033[90m")
    return f"{color}{num_str.rjust(5)}\033[0m"

def print_ranking_table(players, global_stats, prev_ranks, prev_ratings, round_num, total_games_played):
    """
    Prints the ranking table to console.
    """
    
    sorted_players = sorted(players.items(), key=lambda x: x[1].rating, reverse=True)
    
    header = (
        f"{'Rank':<5} {'Agent':<28} {'Change':<9} {'ELO':>6} {'Delta':>6} {'Winrate':>8} "
        f"{'W':>5} {'L':>5} {'D':>5} {'Match':>6} {'Bal':>4} {'Ø Val':>7} {'Ø Pip':>7} {'RD':>6}"
    )
    separator = "-" * 125
    
    output_lines = []
    title = f"\nStandings after Round {round_num} ({total_games_played} games per matchup)."
    
    print(title); output_lines.append(title)
    print(header); output_lines.append(header)
    print(separator); output_lines.append(separator)

    current_ranks = {}
    
    for i, (name, p) in enumerate(sorted_players):
        rank = i + 1
        current_ranks[name] = rank
        
        s = global_stats[name]
        matches = s['total_matches']
        
        # Metrics
        winrate = (s['wins'] / matches * 100) if matches > 0 else 0.0
        avg_val = s['total_value'] / matches if matches > 0 else 0.0
        avg_pip = s['total_pips'] / matches if matches > 0 else 0.0
        
        # Changes
        prev_rank = prev_ranks.get(name, rank)
        rank_change = prev_rank - rank
        change_str = format_change(rank_change)
        
        prev_rating = prev_ratings.get(name, ELO_INIT)
        rating_delta = p.rating - prev_rating
        delta_str = format_delta(rating_delta)
        
        rd_str = f"{get_rd_color(p.rd)}{p.rd:>6.0f}\033[0m"
        
        balance = s['p0_count'] - s['p1_count']
        
        line = (
            f"{rank:<5} {name:<28} {change_str:<18} {p.rating:>6.0f} {delta_str} {winrate:>7.1f}% "
            f"{s['wins']:>5} {s['losses']:>5} {s['draws']:>5} {matches:>6} {balance:>4} "
            f"{avg_val:>7.2f} {avg_pip:>7.2f} {rd_str}"
        )
        output_lines.append(line)
        print(line)
        
    print(separator + "\n"); output_lines.append(separator + "\n")
    
    return current_ranks, output_lines

def save_txt_report(lines, filename):
    """
    Saves the ranking table to a text file.
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
        for line in lines:
            f.write(ansi_escape.sub('', line) + "\n")

def save_matrices(agent_names, head_to_head, output_dir):
    """
    Save Winrate, Payoff, and Pip matrices.
    """
    
    matrices = {
        'winrate': lambda s: f"{(s['wins']/s['total'])*100:.1f}%" if s['total'] else "-",
        'payoff': lambda s: f"{s['val']/s['total']:.2f}" if s['total'] else "-",
        'pips': lambda s: f"{s['pips']/s['total']:.1f}" if s['total'] else "-"
    }
    
    for metric, func in matrices.items():
        path = os.path.join(output_dir, f"matrix_{metric}.csv")
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['v Agent'] + agent_names)
            
            for p0 in agent_names:
                row = [p0]
                for p1 in agent_names:
                    if p0 == p1:
                        row.append("-")
                    else:
                        key = tuple(sorted((p0, p1)))
                        stats = head_to_head[key]
                        
                        rel_stats = {'total': stats['total']}
                        
                        if key[0] == p0:
                            rel_stats['wins'] = stats['p0_wins']
                            rel_stats['val'] = stats['p0_val_diff']
                            rel_stats['pips'] = stats['p0_pip_diff']
                        else:
                            rel_stats['wins'] = stats['p1_wins']
                            rel_stats['val'] = -stats['p0_val_diff'] # Invert for P1
                            rel_stats['pips'] = -stats['p0_pip_diff'] # Invert for P1
                        
                        row.append(func(rel_stats))
                writer.writerow(row)

def run_tournament():
    """
    Runs the full tournament and outputs the results.
    """
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    agent_names = sorted(list(AGENTS_REGISTRY.keys()))
    print(f"Loaded {len(agent_names)} agents: {', '.join(agent_names)}")
    
    head_to_head = defaultdict(lambda: {
        'total': 0, 
        'p0_wins': 0, 'p1_wins': 0, 'draws': 0,
        'p0_val_diff': 0.0, 'p0_pip_diff': 0.0
    })
    
    global_stats = {name: {
        'wins': 0, 'losses': 0, 'draws': 0, 
        'total_matches': 0, 
        'total_value': 0.0, 'total_pips': 0.0,
        'p0_count': 0, 'p1_count': 0
    } for name in agent_names}
    
    players = {name: glicko2.Player() for name in agent_names}
    
    prev_ranks = {name: i+1 for i, name in enumerate(agent_names)} 
    prev_ratings = {name: ELO_INIT for name in agent_names}
    
    elo_history = []
    unique_pairs = list(itertools.combinations(agent_names, 2))
    total_rounds = TARGET_TOTAL_GAMES // BATCH_SIZE
    
    print(f"Starting Tournament: {TARGET_TOTAL_GAMES} games per matchup.")
    print(f"Batch Size: {BATCH_SIZE} | Rounds: {total_rounds}")
    print("="*115)
    
    start_time = time.time()

    try:
        with Pool(processes=MAX_WORKERS, initializer=worker_init) as pool:
            
            for round_idx in range(1, total_rounds + 1):
                round_seed_base = MASTER_SEED + (round_idx * 9999)
                job_args = []
                for i, (p0, p1) in enumerate(unique_pairs):
                    job_args.append((p0, p1, BATCH_SIZE, round_seed_base + i*123))
                
                results_batch = list(tqdm(
                    pool.imap_unordered(play_matchup_batch, job_args), 
                    total=len(job_args), 
                    desc=f"Round {round_idx}/{total_rounds}",
                    leave=False 
                ))
                
                round_elo_updates = defaultdict(lambda: {'ratings': [], 'rds': [], 'outcomes': []})
                
                for p_a, p_b, res in results_batch:
                    key = tuple(sorted((p_a, p_b)))
                    is_sorted_order = (p_a == key[0])
                    
                    h2h = head_to_head[key]
                    h2h['total'] += BATCH_SIZE
                    h2h['draws'] += res['draws']
                    
                    if is_sorted_order:
                        h2h['p0_wins'] += res['a_wins']
                        h2h['p1_wins'] += res['b_wins']
                        h2h['p0_val_diff'] += res['a_total_score']
                        h2h['p0_pip_diff'] += res['a_total_pips']
                    else:
                        h2h['p0_wins'] += res['b_wins']
                        h2h['p1_wins'] += res['a_wins']
                        h2h['p0_val_diff'] -= res['a_total_score'] # B is P0
                        h2h['p0_pip_diff'] -= res['a_total_pips']

                    stats_a = global_stats[p_a]
                    stats_a['total_matches'] += BATCH_SIZE
                    stats_a['wins'] += res['a_wins']
                    stats_a['losses'] += res['b_wins']
                    stats_a['draws'] += res['draws']
                    stats_a['total_value'] += res['a_total_score']
                    stats_a['total_pips'] += res['a_total_pips']
                    stats_a['p0_count'] += BATCH_SIZE // 2
                    stats_a['p1_count'] += BATCH_SIZE // 2
                    
                    stats_b = global_stats[p_b]
                    stats_b['total_matches'] += BATCH_SIZE
                    stats_b['wins'] += res['b_wins']
                    stats_b['losses'] += res['a_wins']
                    stats_b['draws'] += res['draws']
                    stats_b['total_value'] -= res['a_total_score'] # Invert
                    stats_b['total_pips'] -= res['a_total_pips']   # Invert
                    stats_b['p0_count'] += BATCH_SIZE // 2
                    stats_b['p1_count'] += BATCH_SIZE // 2

                    # Prepare Glicko updates
                    score_a = (res['a_wins'] + 0.5 * res['draws']) / BATCH_SIZE
                    round_elo_updates[p_a]['ratings'].append(players[p_b].rating)
                    round_elo_updates[p_a]['rds'].append(players[p_b].rd)
                    round_elo_updates[p_a]['outcomes'].append(score_a)
                    
                    round_elo_updates[p_b]['ratings'].append(players[p_a].rating)
                    round_elo_updates[p_b]['rds'].append(players[p_a].rd)
                    round_elo_updates[p_b]['outcomes'].append(1.0 - score_a)
                
                # Apply Glicko
                current_elo_snapshot = {}
                old_ratings_snapshot = {n: p.rating for n, p in players.items()}
                
                for name, p in players.items():
                    data = round_elo_updates[name]
                    if data['ratings']:
                        p.update_player(data['ratings'], data['rds'], data['outcomes'])
                    else:
                        p.did_not_compete()
                    
                    current_elo_snapshot[name] = round(p.rating, 1)
                
                elo_history.append(current_elo_snapshot)
                
                curr_ranks, table_lines = print_ranking_table(
                    players, global_stats, prev_ranks, prev_ratings, 
                    round_idx, round_idx * BATCH_SIZE
                )
                
                prev_ranks = curr_ranks
                prev_ratings = old_ratings_snapshot

                save_matrices(agent_names, head_to_head, OUTPUT_DIR)
                
                headers = ['Round'] + list(current_elo_snapshot.keys())
                with open(os.path.join(OUTPUT_DIR, "elo_history.csv"), 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for idx, entry in enumerate(elo_history):
                        row = {'Round': idx}
                        row.update(entry)
                        writer.writerow(row)
                        
                save_txt_report(table_lines, os.path.join(OUTPUT_DIR, "final_ranking.txt"))

    except KeyboardInterrupt:
        print("\nTournament interrupted!")
        
    end_time = time.time()
    print(f"\nTournament Complete in {(end_time - start_time)/60:.1f} minutes.")
    print(f"Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    run_tournament()
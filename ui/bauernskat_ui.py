'''
    File name: /ui/bauernskat_ui.py
    Author: Oliver Czerwinski
    Date created: 09/16/2025
    Date last modified: 12/27/2025
    Python Version: 3.9+
'''

import os
import time
import random
import base64
import streamlit as st
import sys
import torch
from safetensors.torch import load_file
import rlcard

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from rlcard.games.bauernskat.action_event import ActionEvent, PlayCardAction, DeclareTrumpAction
from rlcard.games.bauernskat.card import BauernskatCard

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

APP_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(APP_DIR, 'assets')
CARD_IMAGES_DIR = os.path.join(ASSETS_DIR, 'cards')
ICON_PATH = os.path.join(ASSETS_DIR, 'icon.png')
CARD_WIDTH = 96

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

@st.cache_data
def get_image_as_base64(path: str) -> str:
    """
    Gets base64 encoded image file.
    """
    
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@st.cache_resource
def load_agent(agent_name: str):
    """
    Loads an agent based on the registry config.
    """
    config = AGENTS_REGISTRY[agent_name]
    agent_type = config['type']
    
    # Rule agent
    if agent_type == 'rule':
        instance = config['class']()
        if hasattr(instance, 'agents'):
            return instance.agents[0]

        return instance

    # RL agent
    path = config['path']

    if not os.path.exists(path):
        potential_path = os.path.join(project_root, path)
        
        if os.path.exists(potential_path):
            path = potential_path
        else:
            st.error(f"Model file not found: {path}")
            st.stop()
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if path.endswith('.safetensors'):
            state_dict = load_file(path)
        else:
            data = torch.load(path, map_location=device, weights_only=False)
            state_dict = data['model_state_dict'] if isinstance(data, dict) and 'model_state_dict' in data else data
    except Exception as e:
        st.error(f"Failed to load model file: {e}")
        st.stop()

    if agent_type == 'dmc':
        net = DMCNet(DMCConfig())
        AgentClass = DMCActor
    elif agent_type == 'sac':
        net = SACNet(SACConfig())
        AgentClass = SACActor
    else:
        st.error(f"Unknown agent type: {agent_type}")
        st.stop()

    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()
    
    return AgentClass(net, device=device)

def apply_custom_css():
    """
    Custom CSS for UI styling.
    """
    
    st.markdown(f"""
        <style>
            .main .block-container {{
                max-width: 98%;
                padding: 1rem 1.5rem 1.5rem;
            }}
            .card-slot-container {{
                position: relative;
                width: {CARD_WIDTH}px;
                height: {int(CARD_WIDTH * 1.5)}px;
            }}
            .card-slot-container img {{
                border-radius: 5px;
                border: 1px solid #777;
                box-shadow: 2px 2px 3px rgba(0,0,0,0.3);
                width: 100%;
                height: 100%;
                object-fit: cover;
            }}
            .hidden-card-indicator {{
                position: absolute;
                top: -3px;
                left: -3px;
                width: {CARD_WIDTH+6}px;
                height: {int(CARD_WIDTH*1.5)+6}px;
                border: 3px solid #FFD700;
                border-radius: 8px;
                pointer-events: none;
            }}
            .empty-slot {{
                border: 2px dashed #555;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 5px;
                width: {CARD_WIDTH}px;
                height: {int(CARD_WIDTH * 1.5)}px;
            }}
        </style>
    """, unsafe_allow_html=True)

def render_player_layout(player_layout):
    """
    Renders a player layout.
    """
    
    b64_images = {str(card): get_image_as_base64(os.path.join(CARD_IMAGES_DIR, f"{str(card)}.png"))
                for card in BauernskatCard.get_deck()}

    def render_card_slot_html(col_data):
        """
        Renders HTML for a card slot.
        """
        
        card = col_data.open_card
        html = '<div class="card-slot-container">'
        if card:
            b64_src = f"data:image/png;base64,{b64_images.get(str(card), '')}"
            html += f'<img src="{b64_src}" width="{CARD_WIDTH}">'
            if col_data.closed_card is not None:
                html += '<div class="hidden-card-indicator"></div>'
        else:
            html += '<div class="empty-slot"></div>'
        html += '</div>'
        return html

    cols_row1 = st.columns(4)
    for i in range(4):
        with cols_row1[i]:
            st.markdown(render_card_slot_html(player_layout[i]), unsafe_allow_html=True)

    cols_row2 = st.columns(4)
    for i in range(4, 8):
        with cols_row2[i - 4]:
            st.markdown(render_card_slot_html(player_layout[i]), unsafe_allow_html=True)

def render_left_panel(human_player_id: int):
    """
    Renders the left panel game status.
    """
    
    game = st.session_state.env.game
    my_score = game.players[human_player_id].score
    agent_score = game.players[1 - human_player_id].score
    
    st.header("Game Status")
    score_cols = st.columns(2)
    score_cols[0].metric("Your Score", my_score)
    score_cols[1].metric("Agent Score", agent_score)
    st.metric("Tricks Played", f"{game.round.tricks_played if game.round else 0} / 16")
    
    trump_suit = game.round.trump_suit if game.round else None
    st.markdown(f"""
        **Trump Suit**  
        <div style="font-size: 32px; font-weight: bold; color: #FFD700; text-align: center;">
            {trump_suit or "-"}
        </div>
    """, unsafe_allow_html=True)
    
    if 'game_seed' in st.session_state:
        st.markdown("---")
        st.caption(f"Seed: {st.session_state.game_seed}")
        st.caption(f"Agent: {st.session_state.agent_name}")

def render_right_panel(state, human_player_id: int):
    """
    Renders the right panel trick info and action buttons.
    """
    
    game = st.session_state.env.game
    
    trick_layout_cols = st.columns([1, 1])
    with trick_layout_cols[0]:
        st.subheader("Previous Trick")
        if 'previous_trick' in st.session_state and st.session_state.previous_trick:
            prev = st.session_state.previous_trick
            st.markdown(f"{prev[0][1]} (P{prev[0][0]}) vs {prev[1][1]} (P{prev[1][0]})")
        else:
            st.caption("No tricks completed yet.")
    
    with trick_layout_cols[1]:
        st.subheader("Current Trick")
        trick_cols = st.columns(2)
        trick_moves = game.round.trick_moves if game.round else []
        if not trick_moves:
            trick_cols[0].caption("Waiting for lead...")
        else:
            path_0 = os.path.join(CARD_IMAGES_DIR, f"{trick_moves[0][1]}.png")
            b64_0 = get_image_as_base64(path_0)
            trick_cols[0].image(f"data:image/png;base64,{b64_0}", width=CARD_WIDTH)
            if len(trick_moves) > 1:
                path_1 = os.path.join(CARD_IMAGES_DIR, f"{trick_moves[1][1]}.png")
                b64_1 = get_image_as_base64(path_1)
                trick_cols[1].image(f"data:image/png;base64,{b64_1}", width=CARD_WIDTH)
    
    st.markdown("---")
    current_player_id = game.get_player_id()
    if current_player_id == human_player_id:
        st.header("Your Turn!")
        round_phase = state['raw_state_info']['round_phase']
        if round_phase == 'declare_trump':
            st.info("Select a trump suit.")
            action_cols = st.columns(len(DeclareTrumpAction.VALID_TRUMPS))
            for i, suit in enumerate(DeclareTrumpAction.VALID_TRUMPS):
                if action_cols[i].button(suit, key=f"declare_{suit}"):
                    handle_action(DeclareTrumpAction(suit))
        elif round_phase == 'play':
            playable = [ActionEvent.from_action_id(aid) for aid in state['legal_actions'] 
                    if isinstance(ActionEvent.from_action_id(aid), PlayCardAction)]
            if not playable:
                st.warning("You have no playable cards.")
                return
            num_cards = len(playable)
            action_cols = st.columns(min(num_cards, 4))
            for i, action in enumerate(playable):
                with action_cols[i % 4]:
                    if st.button(f"Play {action.card}", key=f"play_{action.action_id}"):
                        handle_action(action)
                    path = os.path.join(CARD_IMAGES_DIR, f"{action.card}.png")
                    b64 = get_image_as_base64(path)
                    st.image(f"data:image/png;base64,{b64}", width=CARD_WIDTH)
    else:
        st.header("Agent's Turn")
        st.info("The agent is thinking...")

def handle_action(action):
    """
    Handles an action by the human player.
    """
    
    env = st.session_state.env
    trick_will_be_completed = len(env.game.round.trick_moves) == 1 and isinstance(action, PlayCardAction)
    
    if trick_will_be_completed:
        st.session_state.previous_trick = [
            (env.game.round.trick_moves[0][0], env.game.round.trick_moves[0][1]),
            (env.game.get_player_id(), action.card)
        ]
        
    new_state, _ = env.step(action.action_id)
    st.session_state.state = new_state
    
    if trick_will_be_completed:
        st.session_state.pause_after_render = True
    st.rerun()

def setup_screen():
    """
    Renders the setup screen for game configuration.
    """
    
    st.title("Play Bauernskat vs. AI")
    st.image(ICON_PATH, width=150)
    
    agent_options = sorted(list(AGENTS_REGISTRY.keys()))
    agent_name = st.selectbox("Choose your opponent:", options=agent_options)
    
    role = st.radio("Choose your role:", ["Player 0 (Vorhand)", "Player 1 (Geber)", "Random"], index=2)
    
    seed_input = st.number_input(
        "Game Seed (0 = Random):",
        min_value=0, max_value=999999, value=0, step=1
    )
    
    if st.button("Start Game", type="primary"):
        human_id = random.choice([0, 1]) if role == "Random" else (0 if role.startswith("Player 0") else 1)
        agent_id = 1 - human_id
        
        with st.spinner(f"Loading {agent_name}..."):
            agent_instance = load_agent(agent_name)
            
        agent_config = AGENTS_REGISTRY[agent_name]
        agent_info_level = agent_config['info']
        
        game_seed = seed_input if seed_input != 0 else random.randint(1, 999999)
        
        env = rlcard.make('bauernskat', config={'seed': game_seed})
        
        env.game.information_level = {
            human_id: 'normal',
            agent_id: agent_info_level
        }
        
        st.session_state.env = env
        st.session_state.agent = agent_instance
        st.session_state.agent_type = agent_config['type']
        st.session_state.human_player_id = human_id
        st.session_state.agent_name = agent_name
        st.session_state.game_seed = game_seed
        st.session_state.game_setup = True
        st.session_state.previous_trick = []
        
        state, _ = env.reset()
        st.session_state.state = state
        st.rerun()

def game_screen():
    """
    Renders the game screen.
    """
    
    env = st.session_state.env
    human_player_id = st.session_state.human_player_id
    
    if env.game.is_over():
        st.balloons()
        st.title("Game Over!")
        my_score = env.game.players[human_player_id].score
        agent_score = env.game.players[1 - human_player_id].score
        payoff = env.get_payoffs()[human_player_id]
        
        st.header(f"Final Pips Score: You {my_score} - Agent {agent_score}")
        if payoff > 0:
            st.success(f"You win! Payoff: +{payoff:.0f}")
        elif payoff < 0:
            st.error(f"You lose. Payoff: {payoff:.0f}")
        else:
            st.warning("It's a draw!")
            
        if st.button("Play Again"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        return
    
    left_col, board_col, right_col = st.columns([2.5, 5, 3])
    
    with left_col:
        render_left_panel(human_player_id)
    
    with board_col:
        st.subheader("Agent's Layout")
        render_player_layout(env.game.players[1 - human_player_id].layout)
        st.markdown("---")
        st.subheader("Your Layout")
        render_player_layout(env.game.players[human_player_id].layout)
    
    with right_col:
        render_right_panel(st.session_state.state, human_player_id)
    
    if st.session_state.get('pause_after_render', False):
        st.session_state.pause_after_render = False
        time.sleep(1.5)
        st.rerun()
    
    current_pid = env.game.get_player_id()
    if current_pid != human_player_id and not env.game.is_over():
        time.sleep(0.5)
        
        agent = st.session_state.agent
        agent_type = st.session_state.agent_type
        state = st.session_state.state
        
        if agent_type == 'rule':
            if hasattr(agent, 'eval_step'):
                res = agent.eval_step(state)
                agent_action_id = res[0] if isinstance(res, tuple) else res
            else:
                agent_action_id = agent.step(state)
        else:
            action, _ = agent.eval_step(state, env)
            agent_action_id = action
            
        handle_action(ActionEvent.from_action_id(agent_action_id))

if __name__ == "__main__":
    st.set_page_config(page_title="Bauernskat UI", page_icon=ICON_PATH, layout="wide")
    apply_custom_css()
    
    if 'game_setup' not in st.session_state:
        setup_screen()
    else:
        game_screen()
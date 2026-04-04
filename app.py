import streamlit as st
import chess
import chess.svg
import os
import base64
from stockfish import Stockfish
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. Load Secrets ---
try:
    API_KEY = st.secrets["OPENAI_API_KEY"]
    COACH_PROMPT = st.secrets["COACH_PROMPT"]
    MODEL_ID = st.secrets.get("MODEL_NAME", "gpt-4o") 
except Exception as e:
    st.error("Missing Secrets! Please add OPENAI_API_KEY and COACH_PROMPT to Streamlit Cloud Settings.")
    st.stop()

# --- 2. Initialize Models ---
@st.cache_resource
def load_assets():
    chat_model = init_chat_model(
        MODEL_ID, 
        model_provider="openai", 
        api_key=API_KEY,
        temperature=0.2
    )

    possible_paths = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "stockfish"
    ]
    
    sf = None
    for path in possible_paths:
        try:
            test_sf = Stockfish(path=path)
            sf = test_sf
            break
        except Exception:
            continue

    if sf is None:
        st.error("Engine Error: Stockfish not found.")
    
    return chat_model, sf

chat_model, stockfish = load_assets()

# --- 3. Helper Functions ---
def get_coach_analysis(move_history):
    if not move_history:
        return ""
    
    moves_str = ", ".join([m['san'] for m in move_history])
    
    messages = [
        SystemMessage(content=COACH_PROMPT),
        HumanMessage(content=f"Moves played: {moves_str}. Analyze the quality of the very last move.")
    ]
    
    try:
        response = chat_model.invoke(messages)
        return response.content
    except Exception:
        return "Coach is currently offline."

def render_board(board):
    last_move = board.peek() if board.move_stack else None
    board_svg = chess.svg.board(board=board, lastmove=last_move, size=450)
    b64 = base64.b64encode(board_svg.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'

# --- 4. Session State ---
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.coach_text = "Welcome! Make a move to start the session."

# --- 5. Main UI Layout ---
st.set_page_config(page_title="AI Chess Coach", layout="wide")

st.sidebar.header("Settings & History")
current_skill_level = st.sidebar.slider(
    "AI Difficulty (Skill Level)", 
    min_value=1, 
    max_value=20, 
    value=5,
    help="1 is easiest, 20 is Grandmaster level."
)

col1, col2 = st.columns([3, 2])

with col1:
    error_placeholder = st.empty()
    st.write(render_board(st.session_state.board), unsafe_allow_html=True)
    
    with st.form(key="chess_move_form", clear_on_submit=True):
        move_input = st.text_input("Enter Move (SAN notation, e.g., e4, Nf3, O-O):")
        submit_button = st.form_submit_button("Play Move")

    if submit_button and move_input:
        try:
            move = st.session_state.board.parse_san(move_input)
            if move in st.session_state.board.legal_moves:
                # 1. Update Board with Player Move
                san_move = st.session_state.board.san(move)
                st.session_state.board.push(move)
                st.session_state.move_history.append({"san": san_move, "verdict": ""})
                
                # 2. Get Analysis from Coach
                with st.spinner("Grandmaster is analyzing your move..."):
                    analysis = get_coach_analysis(st.session_state.move_history)
                    st.session_state.coach_text = analysis
                    
                    # Extract verdict keyword
                    verdict_found = ""
                    for category in ["[GOOD]", "[INACCURACY]", "[MISTAKE]", "[BLUNDER]"]:
                        if category in analysis.upper():
                            verdict_found = category
                            break
                    st.session_state.move_history[-1]["verdict"] = verdict_found
                
                # 3. AI Counter-move
                if not st.session_state.board.is_game_over() and stockfish:
                    stockfish.set_skill_level(current_skill_level)
                    stockfish.set_fen_position(st.session_state.board.fen())
                    
                    think_time = 100 + (current_skill_level * 50) 
                    best_move_uci = stockfish.get_best_move_time(think_time)
                    
                    if best_move_uci:
                        ai_move = chess.Move.from_uci(best_move_uci)
                        ai_san = st.session_state.board.san(ai_move)
                        st.session_state.board.push(ai_move)
                        st.session_state.move_history.append({"san": ai_san, "verdict": "AI"})
                
                st.rerun()
            else:
                error_placeholder.error("Illegal move!")
        except ValueError:
            error_placeholder.error("Invalid notation! Use e4, Nf3, etc.")

with col2:
    st.subheader("Coach's Corner")
    
    bg_color = "#f0f2f6" 
    
    # --- Logic Fix for Dynamic Coloring ---
    if st.session_state.move_history:
        # We want the color of the last HUMAN move, not the AI move.
        # Human moves are at indices 0, 2, 4... 
        # So we look for the most recent even-indexed move.
        history_len = len(st.session_state.move_history)
        # If last entry is AI, look at index -2. If last is Human, look at index -1.
        target_idx = history_len - 1 if history_len % 2 != 0 else history_len - 2
        
        if target_idx >= 0:
            latest_verdict = st.session_state.move_history[target_idx].get("verdict", "").upper()
            if "[GOOD]" in latest_verdict: bg_color = "#d4edda"
            elif "[INACCURACY]" in latest_verdict: bg_color = "#fff3cd"
            elif "[MISTAKE]" in latest_verdict: bg_color = "#ffe5d0"
            elif "[BLUNDER]" in latest_verdict: bg_color = "#f8d7da"

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; 
        border-left: 5px solid rgba(0,0,0,0.1); color: #000000; font-size: 16px; line-height: 1.6;">
            {st.session_state.coach_text}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    if st.button("Restart Game"):
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.coach_text = "Welcome! Make a move to start the session."
        st.rerun()

    if st.session_state.board.is_game_over():
        st.success(f"Game Over! Result: {st.session_state.board.result()}")

# --- Sidebar History (Reverse Chronological) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Move History")

history = st.session_state.move_history
pairs = []
for i in range(0, len(history), 2):
    move_num = (i // 2) + 1
    w_data = history[i]
    b_data = history[i+1] if i+1 < len(history) else {"san": "...", "verdict": "AI"}
    pairs.append((move_num, w_data, b_data))

for m_num, w, b in reversed(pairs):
    w_verdict = f" `{w['verdict']}`" if w['verdict'] else ""
    st.sidebar.markdown(f"**{m_num}.** {w['san']}{w_verdict} &nbsp;&nbsp; {b['san']}")

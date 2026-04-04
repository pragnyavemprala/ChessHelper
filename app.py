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
    
    # Filter only player and AI moves for the prompt
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

# Sidebar Configuration
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
    # --- Error Placeholder at Top ---
    error_placeholder = st.empty()

    # Display the Board
    st.write(render_board(st.session_state.board), unsafe_allow_html=True)
    
    # Move Input Form
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
                # Append with temporary verdict
                st.session_state.move_history.append({"san": san_move, "verdict": ""})
                
                # 2. Get Analysis from Coach
                with st.spinner("Grandmaster is analyzing your move..."):
                    analysis = get_coach_analysis(st.session_state.move_history)
                    st.session_state.coach_text = analysis
                    
                    # Extract verdict keyword for Sidebar
                    verdict_found = ""
                    for category in ["[GOOD]", "[INACCURACY]", "[MISTAKE]", "[BLUNDER]"]:
                        if category in analysis.upper():
                            verdict_found = category
                            break
                    st.session_state.move_history[-1]["verdict"] = verdict_found
                
                # 3. Trigger AI Counter-move
                if not st.session_state.board.is_game_over() and stockfish:
                    stockfish.set_skill_level(current_skill_level)
                    stockfish.set_fen_position(st.session_state.board.fen())
                    
                    # Scale think time by difficulty
                    think_time = 100 + (current_skill_level * 50) 
                    best_move_uci = stockfish.get_best_move_time(think_time)
                    
                    if best_move_uci:
                        ai_move = chess.Move.from_uci(best_move_uci)
                        ai_san = st.session_state.board.san(ai_move)
                        st.session_state.board.push(ai_move)
                        st.session_state.move_history.append({"san": ai_san, "verdict": "AI"})
                
                st.rerun()
            else:
                error_placeholder.error("Illegal move! Check the board and try again.")
        except ValueError:
            error_placeholder.error("Invalid move format. Use SAN notation (e.g., e4, Nf3).")

with col2:
    st.subheader("Coach's Corner")
    # --- Dynamic Coloring Logic ---
    # Default color (light grey)
    bg_color = "#f0f2f6" 
    text_color = "#000000"
    
    # Determine color based on the latest verdict
    if st.session_state.move_history:
        # Get the verdict of the last player move
        # (AI moves are at odd indices, Player moves at even indices)
        last_index = len(st.session_state.move_history) - 1
        # If the last move in history was the AI's, look at the one before it
        if st.session_state.move_history[last_index]["san"] == "...":
             last_index -= 1
        
        # Ensure we are looking at a move that has a verdict
        latest_verdict = st.session_state.move_history[last_index].get("verdict", "")

        if "[GOOD]" in latest_verdict:
            bg_color = "#d4edda"  # Light Green
        elif "[INACCURACY]" in latest_verdict:
            bg_color = "#fff3cd"  # Yellow
        elif "[MISTAKE]" in latest_verdict:
            bg_color = "#ffe5d0"  # Orange
        elif "[BLUNDER]" in latest_verdict:
            bg_color = "#f8d7da"  # Red

    # Render the Coach's Corner with custom HTML
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color}; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid rgba(0,0,0,0.1);
            color: {text_color};
            font-size: 16px;
            line-height: 1.6;
        ">
            {st.session_state.coach_text}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # --- End of Dynamic Coloring ---
    
    if st.button("Restart Game"):
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.coach_text = "New game started!"
        st.rerun()

    if st.session_state.board.is_game_over():
        st.success(f"Game Over! Result: {st.session_state.board.result()}")

# --- Sidebar Move History (Reverse Chronological) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Move History")

history = st.session_state.move_history
pairs = []
for i in range(0, len(history), 2):
    move_num = (i // 2) + 1
    w_data = history[i]
    b_data = history[i+1] if i+1 < len(history) else {"san": "...", "verdict": ""}
    pairs.append((move_num, w_data, b_data))

for m_num, w, b in reversed(pairs):
    # Display format: 1. e4 [GOOD]   e5
    w_display = f"{w['san']} `{w['verdict']}`" if w['verdict'] else w['san']
    b_display = b['san']
    st.sidebar.markdown(f"**{m_num}.** {w_display} &nbsp;&nbsp; {b_display}")

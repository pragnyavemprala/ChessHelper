import streamlit as st
import chess
import chess.svg
import os
import base64
from stockfish import Stockfish
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. Load Secrets (Configure these in Streamlit Cloud Dashboard) ---
try:
    API_KEY = st.secrets["OPENAI_API_KEY"]
    COACH_PROMPT = st.secrets["COACH_PROMPT"]
    MODEL_ID = st.secrets.get("MODEL_NAME", "gpt-4o") 
except Exception as e:
    st.error("Missing Secrets! Please add OPENAI_API_KEY and COACH_PROMPT to Streamlit Cloud Settings.")
    st.stop()

# --- 2. Initialize Models (Cached for Performance) ---
@st.cache_resource
def load_assets():
    # Initialize LangChain 1.0 Chat Model
    chat_model = init_chat_model(
        MODEL_ID, 
        model_provider="openai", 
        api_key=API_KEY,
        temperature=0.2
    )

    # Possible Linux locations for Stockfish installed via apt
    possible_paths = [
        "/usr/games/stockfish",   # Standard location
        "/usr/bin/stockfish",     # Alternative
        "stockfish"               # Try calling globally via PATH
    ]
    
    sf = None
    for path in possible_paths:
        try:
            # Test if this path works
            test_sf = Stockfish(path=path)
            sf = test_sf
            # st.success(f"Stockfish engine loaded from: {path}")
            break
        except Exception:
            continue

    if sf is None:
        st.error("Engine Error: Stockfish not found. Please ensure 'packages.txt' contains 'stockfish' and the app was fully rebooted.")
    
    return chat_model, sf

chat_model, stockfish = load_assets()

# --- 3. Helper Functions ---
def get_coach_analysis(move_history):
    if not move_history:
        return ""
    
    moves_str = ", ".join([m['san'] for m in move_history])
    
    # The prompt is kept hidden in st.secrets
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
    # Highlight the last move made
    last_move = board.peek() if board.move_stack else None
    board_svg = chess.svg.board(board=board, lastmove=last_move, size=450)
    b64 = base64.b64encode(board_svg.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'

# --- 4. Session State (Game Persistence) ---
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.coach_text = "Welcome! Make a move to start the session."

# --- 5. Main UI Layout ---
st.set_page_config(page_title="AI Chess Coach", layout="wide")
# st.title("♟️ Grandmaster AI Coach")

col1, col2 = st.columns([3, 2])

with col1:
    # Display the Board
    st.write(render_board(st.session_state.board), unsafe_allow_html=True)

    # Place this before your columns or main logic
    st.sidebar.header("Settings & All moves!")

    current_skill_level = st.sidebar.slider(
        "AI Difficulty (Skill Level)", 
        min_value=1, 
        max_value=20, 
        value=5,  # Default starting level
        help="1 is easiest, 20 is Grandmaster level."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Move History")
    
    # Move Input Form
    with st.form(key="chess_move_form", clear_on_submit=True):
        move_input = st.text_input("Enter Move (SAN notation, e.g., e4, Nf3, O-O):")
        submit_button = st.form_submit_button("Play Move")

    if submit_button and move_input:
        try:
            # Parse user move
            move = st.session_state.board.parse_san(move_input)
            if move in st.session_state.board.legal_moves:
                # 1. Update Board with Player Move
                san_move = st.session_state.board.san(move)
                st.session_state.board.push(move)
                st.session_state.move_history.append({"san": san_move})
                
                # 2. Get Analysis from Coach
                with st.spinner("Grandmaster is analyzing your move..."):
                    st.session_state.coach_text = get_coach_analysis(st.session_state.move_history)
                
                # 3. Trigger AI Counter-move (Stockfish)
                if not st.session_state.board.is_game_over() and stockfish:
                    stockfish.set_skill_level(current_skill_level)

                    stockfish.set_fen_position(st.session_state.board.fen())
                    # Instead of get_best_move() which waits for full calculation,
                    # use a time limit. Lower difficulty + lower time = more mistakes.
                    # We can scale the time slightly based on difficulty.
                    think_time = 100 + (current_skill_level * 50) # e.g., Level 1 = 150ms, Level 20 = 1100ms
                    best_move_uci = stockfish.get_best_move(think_time)
                    if best_move_uci:
                        ai_move = chess.Move.from_uci(best_move_uci)
                        ai_san = st.session_state.board.san(ai_move)
                        st.session_state.board.push(ai_move)
                        st.session_state.move_history.append({"san": ai_san})
                
                st.rerun()
            else:
                st.error("Illegal move! Check the board and try again.")
        except ValueError:
            st.error("Invalid move format. Use Standard Algebraic Notation (e.g., e4, d5, Nf3).")

with col2:
    st.subheader("Coach's Corner")
    st.info(st.session_state.coach_text)
    
    st.subheader("Move History")
    # Display the move list in a scrollable-like format
    # Calculate total number of moves (pairs)
    total_moves = len(st.session_state.move_history)
    # Step -2 means we jump backwards by 2 each time
    for i in range(total_moves - (total_moves % 2), -1, -2):
        # Ensure we don't try to access an index out of bounds
        if i >= total_moves:
            i = total_moves - 1
            if i < 0: break

        # Logic to find the start of the 'pair' (White's move)
        # If i is odd, it's a Black move, so White is at i-1
        # If i is even, it's a White move, so White is at i
        start_of_pair = i if i % 2 == 0 else i - 1
    
        move_num = (start_of_pair // 2) + 1
        white = st.session_state.move_history[start_of_pair]['san']
    
        # Black move only exists if the history is long enough
        black = ""
        if start_of_pair + 1 < total_moves:
            black = st.session_state.move_history[start_of_pair + 1]['san']
        else:
            black = "..." # AI is still thinking or game ended on White

        st.sidebar.markdown(f"**{move_num}.** {white} &nbsp;&nbsp; {black}")

    if st.button("Restart Game"):
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.coach_text = "New game started!"
        st.rerun()

    if st.session_state.board.is_game_over():
        st.success(f"Game Over! Result: {st.session_state.board.result()}")

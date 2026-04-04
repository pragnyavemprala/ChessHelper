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
    chat = init_chat_model(
        MODEL_ID, 
        model_provider="openai", 
        api_key=API_KEY,
        temperature=0.2
    )
    
    # Identify Stockfish Path
    # On Streamlit Cloud (Linux), 'packages.txt' installs it to /usr/games/stockfish
    linux_path = "/usr/games/stockfish"
    # windows_path = os.path.join('stockfish', 'stockfish.exe')
    
    engine_path = linux_path if os.path.exists(linux_path) else "/stockfish"
    
    try:
        sf = Stockfish(path=engine_path, parameters={"Threads": 2, "Minimum Thinking Time": 30})
        return chat, sf
    except Exception as e:
        st.warning(f"Stockfish engine not found. AI opponent disabled. Error: {e}")
        return chat, None

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
st.title("♟️ Grandmaster AI Coach")

col1, col2 = st.columns([2, 1])

with col1:
    # Display the Board
    st.write(render_board(st.session_state.board), unsafe_allow_html=True)
    
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
                    stockfish.set_fen_position(st.session_state.board.fen())
                    best_move_uci = stockfish.get_best_move()
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
    for i in range(0, len(st.session_state.move_history), 2):
        white = st.session_state.move_history[i]['san']
        black = st.session_state.move_history[i+1]['san'] if i+1 < len(st.session_state.move_history) else ""
        st.write(f"**{(i//2)+1}.** {white}   {black}")

    if st.button("Restart Game"):
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.coach_text = "New game started!"
        st.rerun()

    if st.session_state.board.is_game_over():
        st.success(f"Game Over! Result: {st.session_state.board.result()}")

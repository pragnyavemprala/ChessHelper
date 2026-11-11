import os
import re
import uuid
import base64
import time
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import boto3
from boto3.dynamodb.conditions import Key
import chess
import chess.svg
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from stockfish import Stockfish
import streamlit as st


# ==========================================
# 1. Secrets & Configuration
# ==========================================
def load_config():
    """Loads required secrets and configuration keys."""
    try:
        config = {
            "openai_api_key": st.secrets["OPENAI_API_KEY"],
            "coach_prompt": st.secrets["COACH_PROMPT"],
            "model_id": st.secrets.get("MODEL_NAME", "gpt-4o"),
            "aws_access_key": st.secrets.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_key": st.secrets.get("AWS_SECRET_ACCESS_KEY"),
            "aws_region": st.secrets.get("AWS_DEFAULT_REGION", "us-east-1"),
            "table_name": st.secrets.get("DYNAMODB_TABLE", "chess_games"),
        }
        return config
    except Exception as e:
        st.error(f"Missing configuration secrets: {e}")
        st.stop()


CONFIG = load_config()

# Thread pool for asynchronous coach analysis
ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=2)


# ==========================================
# 2. Database Layer (DynamoDB)
# ==========================================
@st.cache_resource
def get_dynamo_table():
    """Initializes and caches the DynamoDB table resource."""
    try:
        session = boto3.Session(
            aws_access_key_id=CONFIG["aws_access_key"],
            aws_secret_access_key=CONFIG["aws_secret_key"],
            region_name=CONFIG["aws_region"],
        )
        dynamodb = session.resource("dynamodb")
        return dynamodb.Table(CONFIG["table_name"])
    except Exception as e:
        st.error(f"Failed to connect to DynamoDB: {e}")
        return None


def get_player_stats(player_name: str):
    """Queries DynamoDB for player records and calculates win/total games."""
    table = get_dynamo_table()
    if not table or not player_name:
        return {"total_games": 0, "wins": 0, "is_new": True}

    try:
        response = table.query(
            KeyConditionExpression=Key("player_name").eq(player_name.strip().lower())
        )
        items = response.get("Items", [])
        if not items:
            return {"total_games": 0, "wins": 0, "is_new": True}

        wins = sum(1 for item in items if item.get("game_status") == "Player Won")
        return {"total_games": len(items), "wins": wins, "is_new": False}
    except Exception as e:
        st.warning(f"Could not retrieve player statistics: {e}")
        return {"total_games": 0, "wins": 0, "is_new": True}


def save_game_record(player_name: str, status: str, history: list, summary: str):
    """Saves the full game details to DynamoDB upon game completion."""
    table = get_dynamo_table()
    if not table or not player_name:
        return False

    player_moves = [m["san"] for i, m in enumerate(history) if i % 2 == 0]
    ai_moves = [m["san"] for i, m in enumerate(history) if i % 2 != 0]

    record = {
        "player_name": player_name.strip().lower(),
        "game_id": f"{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex[:6]}",
        "display_name": player_name.strip(),
        "game_status": status,
        "player_moves": player_moves,
        "ai_moves": ai_moves,
        "summary": summary,
        "total_moves": len(history),
    }

    try:
        table.put_item(Item=record)
        return True
    except Exception as e:
        st.error(f"Error saving match to DynamoDB: {e}")
        return False


# ==========================================
# 3. Model & Engine Assets
# ==========================================
@st.cache_resource
def load_assets():
    """Initializes LLM and Stockfish engine."""
    chat_model = init_chat_model(
        CONFIG["model_id"],
        model_provider="openai",
        api_key=CONFIG["openai_api_key"]
    )

    possible_paths = ["/usr/games/stockfish", "/usr/bin/stockfish", "stockfish"]
    sf = None
    for path in possible_paths:
        try:
            sf = Stockfish(path=path)
            break
        except Exception:
            continue

    if sf is None:
        st.error("Engine Error: Stockfish not found.")

    return chat_model, sf


CHAT_MODEL, STOCKFISH = load_assets()


# ==========================================
# 4. Chess, Clock & Coach Logic
# ==========================================
TIME_CONTROLS = {
    "Bullet (1+0)": (60, 0),
    "Blitz (3+2)": (180, 2),
    "Blitz (5+3)": (300, 3),
    "Rapid (10+0)": (600, 0),
    "Rapid (15+10)": (900, 10),
    "Classical (25+5)": (1500, 5),
    "Unlimited": (None, 0),
}


def render_board(board):
    """Renders the chess board SVG as HTML."""
    last_move = board.peek() if board.move_stack else None
    board_svg = chess.svg.board(board=board, lastmove=last_move, size=450)
    b64 = base64.b64encode(board_svg.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'


def background_analyze_move(move_history_snapshot):
    """Runs asynchronously in background thread to analyze the latest move."""
    try:
        if not move_history_snapshot:
            return
        moves_str = ", ".join([m["san"] for m in move_history_snapshot])
        system_text = CONFIG["coach_prompt"].replace("{moves_str}", moves_str)

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=f"Current move history: {moves_str}. Analyze the last move played."),
        ]
        response = CHAT_MODEL.invoke(messages)
        analysis = response.content
        
        # Determine verdict
        verdict = ""
        for category in ["[GOOD]", "[INACCURACY]", "[MISTAKE]", "[BLUNDER]"]:
            if category in analysis.upper():
                verdict = category
                break

        # Safely update session state state
        st.session_state.coach_text = analysis
        if st.session_state.move_history:
            st.session_state.move_history[-1]["verdict"] = verdict
        st.session_state.is_analyzing = False
    except Exception as e:
        st.session_state.coach_text = f"Coach analysis unavailable: {e}"
        st.session_state.is_analyzing = False


def get_conversational_coach_reply(board, move_history, chat_messages):
    """Carries out a continuous conversational dialog with the coach."""
    moves_str = ", ".join([m["san"] for m in move_history]) if move_history else "No moves made yet"
    turn_str = "White" if board.turn == chess.WHITE else "Black"
    
    system_instructions = (
        "You are an encouraging, insightful International Chess Grandmaster acting as a personal coach. "
        "The player is playing as White against Stockfish (Black). Maintain an interactive, conversational tone. "
        f"Game FEN: {board.fen()}\n"
        f"Game Moves so far: {moves_str}\n"
        f"Current Turn to Move: {turn_str}\n"
        "Remember earlier discussion in this conversation. Answer the player's questions with specific board ideas, "
        "tactical patterns, openings, or candidate plans. Keep answers engaging and concise (2-4 paragraphs max)."
    )

    formatted_messages = [SystemMessage(content=system_instructions)]
    for msg in chat_messages:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_messages.append(AIMessage(content=msg["content"]))

    try:
        response = CHAT_MODEL.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Sorry, I had trouble evaluating that: {e}"


def generate_game_summary(move_history, result_text):
    """Generates an end-of-game summary review."""
    moves_str = ", ".join([m["san"] for m in move_history])
    messages = [
        SystemMessage(content=CONFIG["coach_prompt"]),
        HumanMessage(
            content=f"Game finished with result '{result_text}'. Moves: {moves_str}. Provide a concise 2-sentence match summary."
        ),
    ]
    try:
        response = CHAT_MODEL.invoke(messages)
        return response.content
    except Exception:
        return "Game concluded."


def execute_ai_move(board, skill_level):
    """Executes a Stockfish counter-move immediately."""
    if not STOCKFISH or board.is_game_over():
        return None
    STOCKFISH.set_skill_level(skill_level)
    STOCKFISH.set_fen_position(board.fen())
    best_move_uci = STOCKFISH.get_best_move_time(50) # Instantaneous AI response

    if best_move_uci:
        ai_move = chess.Move.from_uci(best_move_uci)
        ai_san = board.san(ai_move)
        board.push(ai_move)
        return ai_san
    return None


def parse_player_move(board: chess.Board, text: str) -> chess.Move:
    """Parses user input into a valid chess.Move."""
    cleaned = text.strip()
    castling_map = {
        "0-0-0": "O-O-O", "o-o-o": "O-O-O", "ooo": "O-O-O",
        "0-0": "O-O", "o-o": "O-O", "oo": "O-O",
    }
    cleaned = castling_map.get(cleaned.lower(), cleaned)

    try:
        return board.parse_san(cleaned)
    except ValueError:
        pass

    try:
        uci_move = chess.Move.from_uci(cleaned.lower())
        if uci_move in board.legal_moves:
            return uci_move
    except ValueError:
        pass

    raise ValueError(f"'{text}' is not recognized as a legal SAN or UCI move.")


# ==========================================
# 5. Session State Management
# ==========================================
def init_session_state():
    """Sets initial variables in st.session_state."""
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.coach_text = "Welcome! Make a move to start the session."
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your coach. Play a move, or ask me anything about chess strategy."}
        ]
        st.session_state.player_name = ""
        st.session_state.game_saved = False
        st.session_state.is_analyzing = False
        st.session_state.time_control_key = "Classical (25+5)"
        base_time, _ = TIME_CONTROLS["Classical (25+5)"]
        st.session_state.white_time = base_time
        st.session_state.black_time = base_time
        st.session_state.last_move_time = time.time()


def reset_game(time_control_name):
    """Resets board and time controls."""
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.coach_text = "New game started! Make your opening move."
    st.session_state.messages = [
        {"role": "assistant", "content": "New game started! What would you like to open with?"}
    ]
    st.session_state.game_saved = False
    st.session_state.is_analyzing = False
    
    base_time, _ = TIME_CONTROLS.get(time_control_name, (None, 0))
    st.session_state.white_time = base_time
    st.session_state.black_time = base_time
    st.session_state.last_move_time = time.time()
    st.rerun()


# ==========================================
# 6. UI Components & Main App
# ==========================================
def main():
    st.set_page_config(page_title="AI Chess Coach with Clocks", layout="wide")
    init_session_state()

    # --- Sidebar Configuration ---
    st.sidebar.header("Player Profile")
    player_input = st.sidebar.text_input(
        "Enter Player Name:",
        value=st.session_state.player_name,
        placeholder="e.g. Kasparov",
    )
    if player_input:
        st.session_state.player_name = player_input
        stats = get_player_stats(player_input)
        if stats["is_new"]:
            st.sidebar.info(f"Welcome, **{player_input}**! New player registered.")
        else:
            win_rate = (stats["wins"] / stats["total_games"] * 100) if stats["total_games"] else 0
            st.sidebar.success(
                f"**{player_input}**'s Record:\n\n"
                f"- Wins: **{stats['wins']}** / {stats['total_games']}\n"
                f"- Win Rate: **{win_rate:.1f}%**"
            )
    else:
        st.sidebar.warning("Please enter your name to track game stats.")

    st.sidebar.markdown("---")
    st.sidebar.header("Game Settings")
    
    selected_tc = st.sidebar.selectbox(
        "Time Control",
        options=list(TIME_CONTROLS.keys()),
        index=list(TIME_CONTROLS.keys()).index(st.session_state.time_control_key)
    )
    if selected_tc != st.session_state.time_control_key:
        st.session_state.time_control_key = selected_tc
        base_time, _ = TIME_CONTROLS[selected_tc]
        st.session_state.white_time = base_time
        st.session_state.black_time = base_time
        st.session_state.last_move_time = time.time()

    current_skill_level = st.sidebar.slider(
        "AI Difficulty (Skill Level)",
        min_value=1,
        max_value=20,
        value=5,
        help="1 is easiest, 20 is Grandmaster level.",
    )

    if st.sidebar.button("Restart / New Game"):
        reset_game(selected_tc)

    # --- Clock Management & Timeouts ---
    base_time, increment = TIME_CONTROLS[st.session_state.time_control_key]
    if base_time is not None and not st.session_state.board.is_game_over():
        elapsed = time.time() - st.session_state.last_move_time
        if st.session_state.board.turn == chess.WHITE:
            st.session_state.white_time = max(0, st.session_state.white_time - elapsed)
            if st.session_state.white_time == 0:
                st.session_state.board.clear() # Flagged out
        st.session_state.last_move_time = time.time()

    # --- Main Layout ---
    col1, col2 = st.columns([3, 2])

    with col1:
        # Display Clocks
        if base_time is not None:
            w_min, w_sec = divmod(int(st.session_state.white_time), 60)
            b_min, b_sec = divmod(int(st.session_state.black_time), 60)
            col_clock1, col_clock2 = st.columns(2)
            col_clock1.metric("⏱️ White (You)", f"{w_min:02d}:{w_sec:02d}")
            col_clock2.metric("⏱️ Black (AI)", f"{b_min:02d}:{b_sec:02d}")

        error_placeholder = st.empty()
        st.write(render_board(st.session_state.board), unsafe_allow_html=True)

        if not st.session_state.board.is_game_over() and st.session_state.white_time != 0:
            with st.form(key="chess_move_form", clear_on_submit=True):
                move_input = st.text_input("Enter Move (SAN notation e.g., e4, Nf3, O-O):")
                submit_button = st.form_submit_button("Play Move")

            if submit_button and move_input:
                if not st.session_state.player_name:
                    error_placeholder.warning("Please enter your player name in the sidebar before moving.")
                else:
                    try:
                        move = parse_player_move(st.session_state.board, move_input)
                        if move in st.session_state.board.legal_moves:
                            # Apply time increment if applicable
                            if base_time is not None:
                                st.session_state.white_time += increment

                            san_move = st.session_state.board.san(move)
                            st.session_state.board.push(move)
                            st.session_state.move_history.append({"san": san_move, "verdict": "Analyzing..."})

                            # Trigger asynchronous background coach analysis
                            st.session_state.is_analyzing = True
                            ANALYSIS_EXECUTOR.submit(background_analyze_move, list(st.session_state.move_history))

                            # Execute immediate AI counter-move
                            if not st.session_state.board.is_game_over():
                                ai_san = execute_ai_move(st.session_state.board, current_skill_level)
                                if ai_san:
                                    st.session_state.move_history.append({"san": ai_san, "verdict": "AI"})

                            st.rerun()
                        else:
                            error_placeholder.error("Illegal move under current board rules!")
                    except ValueError:
                        error_placeholder.error("Invalid move! Use SAN (e.g., e4, Nf3, O-O) or squares (e1g1).")
        else:
            if st.session_state.white_time == 0:
                st.error("Time out! You lost on time.")
            else:
                st.info("Game Over.")

    with col2:
        st.subheader("Coach's Corner")

        bg_color = "#f0f2f6"
        if st.session_state.move_history:
            history_len = len(st.session_state.move_history)
            target_idx = history_len - 1 if history_len

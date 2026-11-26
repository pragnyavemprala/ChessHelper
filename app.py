import os
import re
import uuid
import base64
import time
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
            "model_id": st.secrets.get("MODEL_NAME") or "gpt-6-luna",
            "astra_model_id": st.secrets.get("ASTRA_MODEL_NAME") or "gpt-6.2-astra",
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
ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=2)


# ==========================================
# 2. Opening Repertoire & Time Controls
# ==========================================
OPENINGS = {
    "Free Play (Standard Start)": {
        "color": None,
        "moves": [],
        "description": "Standard starting position. Choose either White or Black.",
    },
    "Italian Game (White)": {
        "color": chess.WHITE,
        "moves": ["e4", "e5", "Nf3", "Nc6", "Bc4"],
        "description": "Classical open game targeting f7 and rapid kingside development.",
    },
    "Ruy Lopez (White)": {
        "color": chess.WHITE,
        "moves": ["e4", "e5", "Nf3", "Nc6", "Bb5"],
        "description": "Applies immediate pressure to Black's knight defending the e5 pawn.",
    },
    "Queen's Gambit (White)": {
        "color": chess.WHITE,
        "moves": ["d4", "d5", "c4"],
        "description": "Offers a flank pawn to gain central control.",
    },
    "London System (White)": {
        "color": chess.WHITE,
        "moves": ["d4", "d5", "Bf4"],
        "description": "Solid, principled setup with early dark-squared bishop development.",
    },
    "Sicilian Defense (Black)": {
        "color": chess.BLACK,
        "moves": ["e4", "c5"],
        "description": "Asymmetrical counter-attack fighting for d4 from the flank.",
    },
    "French Defense (Black)": {
        "color": chess.BLACK,
        "moves": ["e4", "e6", "d4", "d5"],
        "description": "Solid pawn chain challenging White's center immediately.",
    },
    "Caro-Kann Defense (Black)": {
        "color": chess.BLACK,
        "moves": ["e4", "c6", "d4", "d5"],
        "description": "Supports ...d5 without blocking the light-squared bishop.",
    },
    "King's Indian Defense (Black)": {
        "color": chess.BLACK,
        "moves": ["d4", "Nf6", "c4", "g6"],
        "description": "Hypermodern defense allowing White a broad center before striking back.",
    },
}

TIME_CONTROLS = {
    "Bullet (1+0)": (60, 0),
    "Blitz (3+2)": (180, 2),
    "Blitz (5+3)": (300, 3),
    "Rapid (10+0)": (600, 0),
    "Rapid (15+10)": (900, 10),
    "Classical (25+5)": (1500, 5),
    "Unlimited": (None, 0),
}


# ==========================================
# 3. Database Layer (DynamoDB)
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


def save_game_record(player_name: str, status: str, history: list, summary: str, chat_messages: list):
    """Saves the full game details including chat transcript to DynamoDB upon game completion."""
    table = get_dynamo_table()
    if not table or not player_name:
        return False

    player_moves = [m["san"] for i, m in enumerate(history) if i % 2 == 0]
    ai_moves = [m["san"] for i, m in enumerate(history) if i % 2 != 0]
    formatted_chat = [
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in chat_messages
        if msg.get("role") != "system"
    ]

    record = {
        "player_name": player_name.strip().lower(),
        "game_id": f"{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex[:6]}",
        "display_name": player_name.strip(),
        "game_status": status,
        "player_moves": player_moves,
        "ai_moves": ai_moves,
        "summary": summary,
        "chat_transcript": formatted_chat,
        "total_moves": len(history),
    }

    try:
        table.put_item(Item=record)
        return True
    except Exception as e:
        st.error(f"Error saving match to DynamoDB: {e}")
        return False


# ==========================================
# 4. Model & Engine Assets
# ==========================================
@st.cache_resource
def load_assets():
    """Initializes standard LLM, GPT-6.2 Astra coach, and Stockfish engine."""
    chat_model = init_chat_model(
        model=CONFIG["model_id"],
        model_provider="openai",
        api_key=CONFIG["openai_api_key"],
    )

    astra_model = init_chat_model(
        model=CONFIG["astra_model_id"],
        model_provider="openai",
        api_key=CONFIG["openai_api_key"],
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

    return chat_model, astra_model, sf


CHAT_MODEL, ASTRA_MODEL, STOCKFISH = load_assets()


# ==========================================
# 5. Chess & Guided Coaching Logic
# ==========================================
def render_board(board, flipped=False):
    """Renders the chess board SVG as HTML, flipped if playing as Black."""
    last_move = board.peek() if board.move_stack else None
    board_svg = chess.svg.board(board=board, lastmove=last_move, flipped=flipped, size=450)
    b64 = base64.b64encode(board_svg.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'


def analyze_move_task(move_history_snapshot):
    """Runs in background thread for Coach's Corner badge and summary."""
    try:
        if not move_history_snapshot:
            return None, ""
        moves_str = ", ".join([m["san"] for m in move_history_snapshot])
        system_text = CONFIG["coach_prompt"].replace("{moves_str}", moves_str)

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=f"Current move history: {moves_str}. Analyze the last move played."),
        ]
        response = CHAT_MODEL.invoke(messages)
        analysis = response.content

        verdict = ""
        for category in ["[GOOD]", "[INACCURACY]", "[MISTAKE]", "[BLUNDER]"]:
            if category in analysis.upper():
                verdict = category
                break

        return analysis, verdict
    except Exception as e:
        return f"Coach analysis unavailable: {e}", ""


def generate_guided_turn_feedback(
    board: chess.Board,
    move_history: list,
    player_san: str,
    ai_san: str | None,
    player_color: chess.Color,
    opening_name: str,
) -> str:
    """
    Evaluates the player's played move in context of the AI's reply,
    praising good moves or giving Socratic hints without spoiling the next move.
    """
    moves_str = ", ".join([m["san"] for m in move_history])
    user_color_str = "White" if player_color == chess.WHITE else "Black"
    ai_reply_context = (
        f"The AI just responded with **{ai_san}**."
        if ai_san
        else "The game has reached a terminal state."
    )

    system_instructions = (
        "You are an encouraging, Socratic Chess Coach guiding a student move-by-move in chat.\n"
        f"Opening Focus: {opening_name}\n"
        f"Student Color: {user_color_str}\n"
        f"Full Move History: {moves_str}\n"
        f"Current Board FEN: {board.fen()}\n\n"
        "STRICT COACHING RULES:\n"
        f"1. Evaluate the student's last move (**{player_san}**) in relation to the opening plan and the AI's response ({ai_san or 'Game Over'}).\n"
        "2. If the student's move was GOOD or principled: give a warm, encouraging message explaining why it worked well.\n"
        "3. If the student's move had a FLAW (inaccuracy, mistake, or blunder): clearly explain the flaw or missed threat, "
        "and provide 1-2 strategic hints to help them think through the current board.\n"
        "4. CRITICAL: NEVER tell the student the exact move to play next, and NEVER name specific destination squares or piece moves for their upcoming turn.\n"
        "5. If the game is not over, end your message by explicitly asking the student for their next move."
    )

    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(
            content=f"I just played {player_san}. {ai_reply_context} How did my move compare to the AI's response, and what should I think about next?"
        ),
    ]

    try:
        response = ASTRA_MODEL.invoke(messages)
        return response.content
    except Exception as e:
        prompt_next = "What is your next move?" if not board.is_game_over() else "Game over!"
        return f"You played **{player_san}**{f' and AI replied **{ai_san}**' if ai_san else ''}. (Coach note: {e}) {prompt_next}"


def evaluate_candidate_move_with_astra(board, move_history, proposed_move_text, player_color):
    """Uses GPT-6.2 Astra to evaluate a candidate move before the player commits."""
    moves_str = ", ".join([m["san"] for m in move_history]) if move_history else "No moves made yet"
    user_color_str = "White" if player_color == chess.WHITE else "Black"
    legal_sans = ", ".join([board.san(m) for m in board.legal_moves])

    system_instructions = (
        "You are GPT-6.2 Astra, a strict, Socratic Chess Grandmaster coach helping a student learn before they commit to a move.\n"
        f"The student is playing as {user_color_str}.\n"
        f"Current FEN: {board.fen()}\n"
        f"Move History: {moves_str}\n"
        f"Legal Moves in this position: {legal_sans}\n\n"
        "STRICT RULES:\n"
        "1. Evaluate ONLY the candidate move they proposed.\n"
        "2. If GOOD: encourage them to play it.\n"
        "3. If FLAWED: explain the flaw and give a directional hint without revealing the actual best move.\n"
        "4. NEVER recommend, suggest, name, or hint at any specific alternative move.\n"
        "5. End by asking what move they would like to play."
    )

    formatted_messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"I am thinking of playing: {proposed_move_text}"),
    ]

    try:
        response = ASTRA_MODEL.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Astra could not evaluate that move right now: {e}. What move would you like to play?"


def init_chat_history():
    """Initializes persistent chat history and archive containers."""
    if "archived_chats" not in st.session_state:
        st.session_state.archived_chats = []
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Welcome! Select an opening or side in the sidebar, make your move, and I will analyze your play against the AI step by step. What is your first move?",
            }
        ]


def generate_game_summary(move_history, result_text, chat_messages):
    """Generates an end-of-game summary review incorporating move history and chat exchanges."""
    moves_str = ", ".join([m["san"] for m in move_history])
    chat_snippet = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_messages[-6:]])
    messages = [
        SystemMessage(content=CONFIG["coach_prompt"]),
        HumanMessage(
            content=f"Game finished with result '{result_text}'. Moves: {moves_str}. Recent Coach/Player Chat: {chat_snippet}. Provide a concise 2-3 sentence match summary."
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
    best_move_uci = STOCKFISH.get_best_move_time(50)

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


def execute_takeback(player_color):
    """Reverts the player's last turn taking side color into account."""
    if st.session_state.takeback_used or not st.session_state.move_history:
        return

    history_len = len(st.session_state.move_history)
    if player_color == chess.WHITE:
        moves_to_pop = 2 if history_len % 2 == 0 else 1
    else:
        moves_to_pop = 2 if history_len % 2 != 0 and history_len > 0 else 0

    if moves_to_pop == 0:
        return

    for _ in range(moves_to_pop):
        if st.session_state.board.move_stack:
            st.session_state.board.pop()
        if st.session_state.move_history:
            st.session_state.move_history.pop()

    st.session_state.is_analyzing = False
    if "analysis_future" in st.session_state:
        st.session_state.analysis_future.cancel()
        del st.session_state["analysis_future"]

    st.session_state.takeback_used = True
    st.session_state.last_move_time = time.time()
    st.session_state.coach_text = "Takeback used! You have 0 takebacks remaining. Make your new move."
    st.session_state.messages.append(
        {"role": "assistant", "content": "⚠️ Takeback used and position reverted. What is your new move?"}
    )


# ==========================================
# 6. Session State & Opening Setup
# ==========================================
def init_session_state():
    """Initializes each session state key independently to survive hot-reloads."""
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "move_history" not in st.session_state:
        st.session_state.move_history = []
    if "coach_text" not in st.session_state:
        st.session_state.coach_text = "Welcome! Select an opening or make a move to start."
    if "player_name" not in st.session_state:
        st.session_state.player_name = ""
    if "game_saved" not in st.session_state:
        st.session_state.game_saved = False
    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing = False
    if "takeback_used" not in st.session_state:
        st.session_state.takeback_used = False
    if "player_color" not in st.session_state:
        st.session_state.player_color = chess.WHITE
    if "selected_opening" not in st.session_state:
        st.session_state.selected_opening = "Free Play (Standard Start)"
    if "time_control_key" not in st.session_state:
        st.session_state.time_control_key = "Classical (25+5)"
    if "white_time" not in st.session_state or "black_time" not in st.session_state:
        base_time, _ = TIME_CONTROLS[st.session_state.time_control_key]
        st.session_state.white_time = base_time
        st.session_state.black_time = base_time
    if "last_move_time" not in st.session_state:
        st.session_state.last_move_time = time.time()

    init_chat_history()


def reset_game(time_control_name, player_color, skill_level, opening_name="Free Play (Standard Start)"):
    """Archives current chat, sets up chosen opening & color, and prompts first move."""
    # Archive previous game's chat if meaningful exchanges occurred
    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        st.session_state.archived_chats.append(
            {
                "timestamp": datetime.now().strftime("%H:%M"),
                "opening": st.session_state.get("selected_opening", "Game"),
                "messages": list(st.session_state.messages),
            }
        )

    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.game_saved = False
    st.session_state.is_analyzing = False
    st.session_state.takeback_used = False
    st.session_state.selected_opening = opening_name
    st.session_state.time_control_key = time_control_name

    if "analysis_future" in st.session_state:
        del st.session_state["analysis_future"]

    opening_cfg = OPENINGS.get(opening_name, OPENINGS["Free Play (Standard Start)"])
    effective_color = opening_cfg["color"] if opening_cfg["color"] is not None else player_color
    st.session_state.player_color = effective_color

    base_time, _ = TIME_CONTROLS.get(time_control_name, (None, 0))
    st.session_state.white_time = base_time
    st.session_state.black_time = base_time
    st.session_state.last_move_time = time.time()

    # Apply pre-configured opening moves
    for san in opening_cfg["moves"]:
        move = st.session_state.board.parse_san(san)
        st.session_state.board.push(move)
        st.session_state.move_history.append({"san": san, "verdict": "Book"})

    # If it is the AI's turn after the opening sequence (or standard Black start), AI moves now
    ai_opening_reply = None
    if st.session_state.board.turn != effective_color and not st.session_state.board.is_game_over():
        ai_opening_reply = execute_ai_move(st.session_state.board, skill_level)
        if ai_opening_reply:
            st.session_state.move_history.append({"san": ai_opening_reply, "verdict": "AI"})

    side_label = "White" if effective_color == chess.WHITE else "Black"
    book_seq = " ".join(opening_cfg["moves"])

    if opening_cfg["moves"]:
        intro_msg = (
            f"🎯 **Opening Loaded: {opening_name}** (Playing as **{side_label}**)\n\n"
            f"*{opening_cfg['description']}*\n\n"
            f"Book moves played: `{book_seq}`"
            + (f" — AI followed up with **{ai_opening_reply}**." if ai_opening_reply else ".")
            + "\n\nIt's your turn! What is your next move?"
        )
    else:
        intro_msg = (
            f"♟️ **New Game Started as {side_label}**"
            + (f" — AI opened with **{ai_opening_reply}**." if ai_opening_reply else ".")
            + "\n\nWhat is your next move?"
        )

    st.session_state.messages = [{"role": "assistant", "content": intro_msg}]
    st.session_state.coach_text = f"Practicing **{opening_name}** as **{side_label}**. Enter your move on the board!"
    st.rerun()


# ==========================================
# 7. UI Components & Main App
# ==========================================
def main():
    st.set_page_config(page_title="AI Chess Coach with Opening Trainer", layout="wide")
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
    st.sidebar.header("Opening & Game Settings")

    selected_opening = st.sidebar.selectbox(
        "Select Opening to Practice:",
        options=list(OPENINGS.keys()),
        index=list(OPENINGS.keys()).index(st.session_state.selected_opening),
    )
    opening_info = OPENINGS[selected_opening]
    st.sidebar.caption(opening_info["description"])

    # Lock color selector if opening dictates White or Black; allow choice on Free Play
    if opening_info["color"] is not None:
        forced_side = "White" if opening_info["color"] == chess.WHITE else "Black"
        st.sidebar.radio(
            "Play as (Set by Opening):",
            options=["White", "Black"],
            index=0 if opening_info["color"] == chess.WHITE else 1,
            horizontal=True,
            disabled=True,
        )
        new_player_color = opening_info["color"]
    else:
        selected_side = st.sidebar.radio(
            "Play as:",
            options=["White", "Black"],
            index=0 if st.session_state.player_color == chess.WHITE else 1,
            horizontal=True,
        )
        new_player_color = chess.WHITE if selected_side == "White" else chess.BLACK

    selected_tc = st.sidebar.selectbox(
        "Time Control",
        options=list(TIME_CONTROLS.keys()),
        index=list(TIME_CONTROLS.keys()).index(st.session_state.time_control_key),
    )

    current_skill_level = st.sidebar.slider(
        "AI Difficulty (Skill Level)",
        min_value=1,
        max_value=20,
        value=5,
        help="1 is easiest, 20 is Grandmaster level.",
    )

    settings_changed = (
        (selected_opening != st.session_state.selected_opening)
        or (new_player_color != st.session_state.player_color)
        or (selected_tc != st.session_state.time_control_key)
    )

    has_player_moved = any(
        m.get("verdict") not in ("Book", "AI") for m in st.session_state.move_history
    )

    if settings_changed and has_player_moved:
        st.sidebar.warning("Click 'New Game' to apply your new opening/settings.")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    can_takeback = (
        not st.session_state.takeback_used
        and has_player_moved
        and not st.session_state.board.is_game_over()
    )
    takeback_label = "↩️ Takeback (1)" if not st.session_state.takeback_used else "↩️ Takeback (0)"

    if col_btn1.button(takeback_label, disabled=not can_takeback, use_container_width=True):
        execute_takeback(st.session_state.player_color)
        st.rerun()

    if col_btn2.button("🔄 New Game", use_container_width=True):
        reset_game(selected_tc, new_player_color, current_skill_level, selected_opening)

    # Auto-apply immediately if the user hasn't made a personal move yet
    if settings_changed and not has_player_moved:
        reset_game(selected_tc, new_player_color, current_skill_level, selected_opening)

    # --- Clock Management ---
    base_time, increment = TIME_CONTROLS[st.session_state.time_control_key]
    if base_time is not None and not st.session_state.board.is_game_over():
        elapsed = time.time() - st.session_state.last_move_time
        if st.session_state.board.turn == chess.WHITE:
            st.session_state.white_time = max(0, st.session_state.white_time - elapsed)
            if st.session_state.white_time == 0:
                st.session_state.board.clear()
        else:
            st.session_state.black_time = max(0, st.session_state.black_time - elapsed)
            if st.session_state.black_time == 0:
                st.session_state.board.clear()
        st.session_state.last_move_time = time.time()

    # --- Main Layout ---
    col1, col2 = st.columns([3, 2])

    with col1:
        player_time = (
            st.session_state.white_time
            if st.session_state.player_color == chess.WHITE
            else st.session_state.black_time
        )
        game_active = not st.session_state.board.is_game_over() and (
            base_time is None or player_time > 0
        )

        if base_time is not None:
            w_min, w_sec = divmod(int(st.session_state.white_time), 60)
            b_min, b_sec = divmod(int(st.session_state.black_time), 60)
            col_clock1, col_clock2 = st.columns(2)

            if st.session_state.player_color == chess.WHITE:
                col_clock1.metric("⏱️ White (You)", f"{w_min:02d}:{w_sec:02d}")
                col_clock2.metric("⏱️ Black (AI)", f"{b_min:02d}:{b_sec:02d}")
            else:
                col_clock1.metric("⏱️ Black (You)", f"{b_min:02d}:{b_sec:02d}")
                col_clock2.metric("⏱️ White (AI)", f"{w_min:02d}:{w_sec:02d}")

            if game_active and player_time < 300:
                if st.button("➕ Add 2 Minutes (+120s)", use_container_width=True):
                    if st.session_state.player_color == chess.WHITE:
                        st.session_state.white_time += 120
                    else:
                        st.session_state.black_time += 120
                    st.success("Added 2 minutes to your clock!")
                    st.rerun()

        error_placeholder = st.empty()
        is_board_flipped = st.session_state.player_color == chess.BLACK
        st.write(render_board(st.session_state.board, flipped=is_board_flipped), unsafe_allow_html=True)

        is_player_turn = st.session_state.board.turn == st.session_state.player_color
        player_timed_out = base_time is not None and player_time == 0

        if game_active and not player_timed_out:
            with st.form(key="chess_move_form", clear_on_submit=True):
                move_input = st.text_input("Enter Move (SAN notation e.g., e4, Nf3, O-O):")
                submit_button = st.form_submit_button("Play Move")

            if submit_button and move_input:
                if not st.session_state.player_name:
                    error_placeholder.warning("Please enter your player name in the sidebar before moving.")
                elif not is_player_turn:
                    error_placeholder.warning("Please wait for your turn.")
                else:
                    try:
                        move = parse_player_move(st.session_state.board, move_input)
                        if move in st.session_state.board.legal_moves:
                            if base_time is not None:
                                if st.session_state.player_color == chess.WHITE:
                                    st.session_state.white_time += increment
                                else:
                                    st.session_state.black_time += increment

                            san_move = st.session_state.board.san(move)
                            st.session_state.board.push(move)
                            st.session_state.move_history.append({"san": san_move, "verdict": "Analyzing..."})

                            # Background analysis for Coach's Corner badge
                            st.session_state.is_analyzing = True
                            st.session_state.analysis_future = ANALYSIS_EXECUTOR.submit(
                                analyze_move_task, list(st.session_state.move_history)
                            )

                            # AI counter-move
                            ai_san = None
                            if not st.session_state.board.is_game_over():
                                ai_san = execute_ai_move(st.session_state.board, current_skill_level)
                                if ai_san:
                                    st.session_state.move_history.append({"san": ai_san, "verdict": "AI"})

                            # Log move in chat history and generate Socratic guided feedback
                            move_log = f"Played **{san_move}**" + (f" (AI replied **{ai_san}**)" if ai_san else "")
                            st.session_state.messages.append({"role": "user", "content": move_log})

                            guided_reply = generate_guided_turn_feedback(
                                board=st.session_state.board,
                                move_history=st.session_state.move_history,
                                player_san=san_move,
                                ai_san=ai_san,
                                player_color=st.session_state.player_color,
                                opening_name=st.session_state.selected_opening,
                            )
                            st.session_state.messages.append({"role": "assistant", "content": guided_reply})

                            st.rerun()
                        else:
                            error_placeholder.error("Illegal move under current board rules!")
                    except ValueError:
                        error_placeholder.error("Invalid move! Use SAN (e.g., e4, Nf3, O-O) or squares (e1g1).")
        else:
            if player_timed_out:
                st.error("Time out! You lost on time.")
            else:
                st.info("Game Over.")

    with col2:
        st.subheader("Coach's Corner")

        @st.fragment(run_every=2.0 if st.session_state.is_analyzing else None)
        def render_coach_corner():
            if "analysis_future" in st.session_state and st.session_state.analysis_future.done():
                try:
                    analysis, verdict = st.session_state.analysis_future.result()
                    if analysis is not None:
                        st.session_state.coach_text = analysis
                    if st.session_state.move_history:
                        for i in range(len(st.session_state.move_history) - 1, -1, -1):
                            if st.session_state.move_history[i].get("verdict") not in ("AI", "Book"):
                                st.session_state.move_history[i]["verdict"] = verdict
                                break
                except Exception:
                    pass
                st.session_state.is_analyzing = False
                del st.session_state["analysis_future"]

            bg_color = "#f0f2f6"
            if st.session_state.move_history:
                for item in reversed(st.session_state.move_history):
                    if item.get("verdict") not in ("AI", "Book"):
                        latest_verdict = item.get("verdict", "").upper()
                        if "[GOOD]" in latest_verdict:
                            bg_color = "#d4edda"
                        elif "[INACCURACY]" in latest_verdict:
                            bg_color = "#fff3cd"
                        elif "[MISTAKE]" in latest_verdict:
                            bg_color = "#ffe5d0"
                        elif "[BLUNDER]" in latest_verdict:
                            bg_color = "#f8d7da"
                        break

            analysis_display = st.session_state.coach_text
            if st.session_state.is_analyzing:
                analysis_display += "\n\n*⏳ Grandmaster is reviewing your move in the background...*"

            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 16px; border-radius: 8px; 
                border-left: 5px solid rgba(0,0,0,0.1); color: #000000; font-size: 15px; line-height: 1.5; margin-bottom: 12px;">
                    {analysis_display.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_coach_corner()

        st.markdown("#### Guided Coaching Chat")
        # Scrollable container preserving full game chat history
        chat_container = st.container(height=360, border=True)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if user_query := st.chat_input("Test a candidate move (e.g., Nf6) or ask for a hint..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.spinner("Coach Astra is analyzing..."):
                if len(user_query.split()) <= 2:
                    reply = evaluate_candidate_move_with_astra(
                        st.session_state.board,
                        st.session_state.move_history,
                        user_query,
                        st.session_state.player_color,
                    )
                else:
                    guided_system = (
                        f"{CONFIG['coach_prompt']}\n\n"
                        f"Current Opening: {st.session_state.selected_opening}\n"
                        f"Current FEN: {st.session_state.board.fen()}\n"
                        "IMPORTANT: Never reveal the exact best move or destination square. "
                        "Give strategic hints only and end by asking the student for their next move."
                    )
                    chat_history_objs = [
                        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                        for m in st.session_state.messages
                    ]
                    response = CHAT_MODEL.invoke([SystemMessage(content=guided_system)] + chat_history_objs)
                    reply = response.content

                st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

        # Optional viewer for previous games' chat transcripts during the session
        if st.session_state.archived_chats:
            with st.expander(f"📜 Previous Game Chats ({len(st.session_state.archived_chats)})"):
                for idx, archive in enumerate(reversed(st.session_state.archived_chats), 1):
                    st.markdown(f"**Game {idx}: {archive['opening']}** *(ended {archive['timestamp']})*")
                    for msg in archive["messages"]:
                        st.caption(f"**{msg['role'].upper()}:** {msg['content']}")
                    st.divider()

        handle_game_completion()

    # --- Sidebar Move History ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Move History")
    history = st.session_state.move_history
    pairs = []
    for i in range(0, len(history), 2):
        move_num = (i // 2) + 1
        w_data = history[i]
        b_data = history[i + 1] if i + 1 < len(history) else {"san": "...", "verdict": ""}
        pairs.append((move_num, w_data, b_data))

    for m_num, w, b in reversed(pairs):
        w_verdict = f" `{w['verdict']}`" if w.get("verdict") and w["verdict"] != "Analyzing..." else ""
        b_verdict = f" `{b['verdict']}`" if b.get("verdict") and b["verdict"] != "Analyzing..." else ""
        st.sidebar.markdown(f"**{m_num}.** {w['san']}{w_verdict} &nbsp;&nbsp; {b['san']}{b_verdict}")


def handle_game_completion():
    """Handles end-of-game checks and triggers DynamoDB record write once."""
    base_time, _ = TIME_CONTROLS[st.session_state.time_control_key]
    player_timed_out = base_time is not None and (
        st.session_state.white_time == 0
        if st.session_state.player_color == chess.WHITE
        else st.session_state.black_time == 0
    )
    if not st.session_state.board.is_game_over() and not player_timed_out:
        return

    if player_timed_out:
        status = "AI Won"
        result = "0-1" if st.session_state.player_color == chess.WHITE else "1-0"
    else:
        result = st.session_state.board.result()
        if result == "1-0":
            status = "Player Won" if st.session_state.player_color == chess.WHITE else "AI Won"
        elif result == "0-1":
            status = "AI Won" if st.session_state.player_color == chess.WHITE else "Player Won"
        else:
            status = "Draw"

    st.success(f"Game Over! Result: {result} ({status})")

    if not st.session_state.game_saved:
        with st.spinner("Saving match summary and chat log to DynamoDB..."):
            summary = generate_game_summary(st.session_state.move_history, status, st.session_state.messages)
            if st.session_state.player_name:
                saved = save_game_record(
                    st.session_state.player_name,
                    status,
                    st.session_state.move_history,
                    summary,
                    st.session_state.messages,
                )
                if saved:
                    st.toast("Game outcome and chat transcript recorded in DynamoDB!")
            st.session_state.game_saved = True


if __name__ == "__main__":
    main()

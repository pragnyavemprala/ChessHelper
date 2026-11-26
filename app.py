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


def save_game_record(player_name: str, status: str, history: list, summary: str, chat_messages: list):
    """Saves the full game details including chat transcript to DynamoDB upon game completion."""
    table = get_dynamo_table()
    if not table or not player_name:
        return False

    player_moves = [m["san"] for i, m in enumerate(history) if i % 2 == 0]
    ai_moves = [m["san"] for i, m in enumerate(history) if i % 2 != 0]
    
    formatted_chat = [f"{msg['role'].upper()}: {msg['content']}" for msg in chat_messages if msg.get("role") != "system"]

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
# 3. Model & Engine Assets
# ==========================================
@st.cache_resource
def load_assets():
    """Initializes standard LLM, GPT-6.2 Astra pre-move coach, and Stockfish engine."""
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


def render_board(board, flipped=False):
    """Renders the chess board SVG as HTML, flipped if playing as Black."""
    last_move = board.peek() if board.move_stack else None
    board_svg = chess.svg.board(board=board, lastmove=last_move, flipped=flipped, size=450)
    b64 = base64.b64encode(board_svg.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'


def analyze_move_task(move_history_snapshot):
    """Runs in background thread and returns the result safely."""
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


def evaluate_candidate_move_with_astra(board, move_history, proposed_move_text, player_color):
    """Uses GPT-6.2 Astra to evaluate the player's proposed move."""
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
        "1. Evaluate ONLY the move they proposed.\n"
        "2. State clearly whether their proposed move is a Good Move, Dubious/Inaccurate, or a Mistake/Blunder in this position, and explain the concrete tactical or positional consequences.\n"
        "3. NEVER recommend, suggest, name, or hint at any alternative move or different piece to play.\n"
        "4. If their input is not a legal chess move in the current position, briefly tell them it is illegal.\n"
        "5. Keep your response concise (1-2 short paragraphs)."
    )

    formatted_messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"I am thinking of playing: {proposed_move_text}"),
    ]

    try:
        response = ASTRA_MODEL.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Astra could not evaluate that move right now: {e}. Please go ahead and make your move on the board."


def init_chat_history():
    """Initializes persistent chat history if not already present."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI Chess Coach powered by GPT-6.2 Astra. Ask me questions or check your candidate moves here throughout the game.",
            }
        ]


def generate_game_summary(move_history, result_text, chat_messages):
    """Generates an end-of-game summary review incorporating move history and chat exchanges."""
    moves_str = ", ".join([m["san"] for m in move_history])
    chat_snippet = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_messages[-6:]]) 
    messages = [
        SystemMessage(content=CONFIG["coach_prompt"]),
        HumanMessage(
            content=f"Game finished with result '{result_text}'. Moves: {moves_str}. Recent Coach/Player Chat: {chat_snippet}. Provide a concise 2-3 sentence match summary incorporating key discussions."
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

    st.session_

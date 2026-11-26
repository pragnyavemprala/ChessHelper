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
    
    # Serialize chat messages into a readable format for DynamoDB
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
    """
    Uses GPT-6.2 Astra to evaluate the player's proposed move.
    """
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
        "2. State clearly whether their proposed move is a **Good Move**, **Dubious/Inaccurate**, or a **Mistake/Blunder** in this position, and explain the concrete tactical or positional consequences.\n"
        "

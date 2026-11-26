import os
import re
import uuid
import base64
import random
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import List

import boto3
from boto3.dynamodb.conditions import Key
import chess
import chess.svg
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
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
PUZZLE_RECORD_ID = "PUZZLE_PROGRESS_PROFILE"


# ==========================================
# 2. Pydantic Schema for AI Puzzle Generation
# ==========================================
class GeneratedPuzzle(BaseModel):
    title: str = Field(description="Short descriptive name of the tactical or positional puzzle.")
    fen: str = Field(description="Valid, legal chess FEN string for the puzzle starting position.")
    solution: List[str] = Field(
        description=(
            "Exact sequence of 1 to 5 moves in Standard Algebraic Notation (SAN) starting with the side to move in the FEN. "
            "If multi-move (3 or 5 plies), include the opponent's forced replies at odd indices."
        )
    )
    theme: str = Field(description="Core chess tactical or strategic motif (e.g., Pin, Fork, Lucena Bridge, Greek Gift).")
    hint: str = Field(description="One helpful Socratic hint without naming the exact move or destination square.")


# ==========================================
# 3. Opening Repertoire, Time Controls & Fallback Puzzles
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

OPENING_BOOK_TOP5 = {
    chess.STARTING_FEN: [
        {"rank": 1, "san": "e4", "uci": "e2e4", "eval": "+0.30"},
        {"rank": 2, "san": "d4", "uci": "d2d4", "eval": "+0.28"},
        {"rank": 3, "san": "Nf3", "uci": "g1f3", "eval": "+0.25"},
        {"rank": 4, "san": "c4", "uci": "c2c4", "eval": "+0.22"},
        {"rank": 5, "san": "g3", "uci": "g2g3", "eval": "+0.15"},
    ],
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": [
        {"rank": 1, "san": "c5", "uci": "c7c5", "eval": "+0.25"},
        {"rank": 2, "san": "e5", "uci": "e7e5", "eval": "+0.28"},
        {"rank": 3, "san": "e6", "uci": "e7e6", "eval": "+0.32"},
        {"rank": 4, "san": "c6", "uci": "c7c6", "eval": "+0.34"},
        {"rank": 5, "san": "d6", "uci": "d7d6", "eval": "+0.45"},
    ],
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": [
        {"rank": 1, "san": "Nf6", "uci": "g8f6", "eval": "+0.25"},
        {"rank": 2, "san": "d5", "uci": "d7d5", "eval": "+0.28"},
        {"rank": 3, "san": "e6", "uci": "e7e6", "eval": "+0.32"},
        {"rank": 4, "san": "f5", "uci": "f7f5", "eval": "+0.45"},
        {"rank": 5, "san": "c6", "uci": "c7c6", "eval": "+0.48"},
    ],
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

FALLBACK_PUZZLES = {
    "Opening": [
        {
            "id": "fb_op_1",
            "difficulty": 1,
            "title": "Scholar's Mate Execution",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "solution": ["Qxf7#"],
            "theme": "Weak f7 Square",
            "hint": "Look at Black's weakest square near the king, attacked by both queen and bishop.",
        },
        {
            "id": "fb_op_2",
            "difficulty": 2,
            "title": "Fried Liver Attack Fork",
            "fen": "r1bqkb1r/ppp2ppp/2n5/3np1N1/2B5/8/PPPP1PPP/RNBQK2R w KQkq - 0 6",
            "solution": ["Nxf7"],
            "theme": "Royal Fork on f7",
            "hint": "Sacrifice the knight on f7 to fork Black's queen and rook.",
        },
    ],
    "Middlegame": [
        {
            "id": "fb_mg_1",
            "difficulty": 1,
            "title": "Back-Rank Checkmate",
            "fen": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
            "solution": ["Rd8#"],
            "theme": "Back-Rank Weakness",
            "hint": "Black's king has no luft behind its own pawns.",
        },
        {
            "id": "fb_mg_2",
            "difficulty": 2,
            "title": "Discovered Attack on the Queen",
            "fen": "3q2k1/5ppp/8/3B4/8/3R4/5PPP/6K1 w - - 0 1",
            "solution": ["Bxf7+", "Kxf7", "Rxd8"],
            "theme": "Discovered Attack",
            "hint": "Move your bishop with check so your rook uncovers an attack on Black's queen.",
        },
    ],
    "Endgame": [
        {
            "id": "fb_eg_1",
            "difficulty": 1,
            "title": "Ladder Checkmate",
            "fen": "7k/R7/1R6/8/8/8/8/6K1 w - - 0 1",
            "solution": ["Rb8#"],
            "theme": "Two-Rook Mate",
            "hint": "Your rook on a7 cuts off the 7th rank. Deliver mate on the 8th rank.",
        },
        {
            "id": "fb_eg_2",
            "difficulty": 2,
            "title": "Rook Skewer on the 8th Rank",
            "fen": "3k3r/8/8/8/8/8/8/R3K3 w - - 0 1",
            "solution": ["Ra8+", "Kc7", "Rxh8"],
            "theme": "Linear Skewer",
            "hint": "Check the king along the 8th rank to win the rook behind it.",
        },
    ],
}


def default_puzzle_stats():
    """Returns clean default puzzle tracking structure for all three phases."""
    return {
        "Opening": {
            "level": 1,
            "attempted": 0,
            "correct": 0,
            "streak": 0,
            "seen_titles": [],
            "seen_fens": [],
            "missed_themes": [],
        },
        "Middlegame": {
            "level": 1,
            "attempted": 0,
            "correct": 0,
            "streak": 0,
            "seen_titles": [],
            "seen_fens": [],
            "missed_themes": [],
        },
        "Endgame": {
            "level": 1,
            "attempted": 0,
            "correct": 0,
            "streak": 0,
            "seen_titles": [],
            "seen_fens": [],
            "missed_themes": [],
        },
    }


# ==========================================
# 4. Database Layer (DynamoDB)
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
    """Queries DynamoDB for player match records (excluding puzzle profile item)."""
    table = get_dynamo_table()
    if not table or not player_name:
        return {"total_games": 0, "wins": 0, "is_new": True}

    try:
        response = table.query(
            KeyConditionExpression=Key("player_name").eq(player_name.strip().lower())
        )
        items = [
            item
            for item in response.get("Items", [])
            if item.get("game_id") != PUZZLE_RECORD_ID
        ]
        if not items:
            return {"total_games": 0, "wins": 0, "is_new": True}

        wins = sum(1 for item in items if item.get("game_status") == "Player Won")
        return {"total_games": len(items), "wins": wins, "is_new": False}
    except Exception as e:
        st.warning(f"Could not retrieve player statistics: {e}")
        return {"total_games": 0, "wins": 0, "is_new": True}


def load_puzzle_progress(player_name: str) -> dict:
    """Loads saved puzzle stats, levels, and missed themes from DynamoDB for `player_name`."""
    base = default_puzzle_stats()
    table = get_dynamo_table()
    if not table or not player_name:
        return base

    try:
        resp = table.get_item(
            Key={
                "player_name": player_name.strip().lower(),
                "game_id": PUZZLE_RECORD_ID,
            }
        )
        item = resp.get("Item")
        if not item or "sections" not in item:
            return base

        saved_sections = item["sections"]
        for sec in ["Opening", "Middlegame", "Endgame"]:
            if sec in saved_sections:
                sec_data = saved_sections[sec]
                base[sec]["level"] = int(sec_data.get("level", 1))
                base[sec]["attempted"] = int(sec_data.get("attempted", 0))
                base[sec]["correct"] = int(sec_data.get("correct", 0))
                base[sec]["streak"] = int(sec_data.get("streak", 0))
                base[sec]["seen_titles"] = list(sec_data.get("seen_titles", []))[-30:]
                base[sec]["seen_fens"] = list(sec_data.get("seen_fens", []))[-30:]
                base[sec]["missed_themes"] = list(sec_data.get("missed_themes", []))[-15:]
        return base
    except Exception:
        return base


def save_puzzle_progress(player_name: str, puzzle_stats: dict) -> bool:
    """Persists puzzle record, accuracy, difficulty levels, and missed themes to DynamoDB."""
    table = get_dynamo_table()
    if not table or not player_name:
        return False

    clean_sections = {}
    total_attempted = 0
    total_correct = 0

    for sec in ["Opening", "Middlegame", "Endgame"]:
        data = puzzle_stats.get(sec, {})
        att = int(data.get("attempted", 0))
        cor = int(data.get("correct", 0))
        total_attempted += att
        total_correct += cor

        clean_sections[sec] = {
            "level": int(data.get("level", 1)),
            "attempted": att,
            "correct": cor,
            "streak": int(data.get("streak", 0)),
            "seen_titles": list(data.get("seen_titles", []))[-30:],
            "seen_fens": list(data.get("seen_fens", []))[-30:],
            "missed_themes": list(data.get("missed_themes", []))[-15:],
        }

    record = {
        "player_name": player_name.strip().lower(),
        "game_id": PUZZLE_RECORD_ID,
        "display_name": player_name.strip(),
        "record_type": "puzzle_progress",
        "total_attempted": total_attempted,
        "total_correct": total_correct,
        "sections": clean_sections,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        table.put_item(Item=record)
        return True
    except Exception as e:
        st.warning(f"Could not save puzzle progress to DynamoDB: {e}")
        return False


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
# 5. Model & Engine Assets
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
            sf = Stockfish(path=path, depth=10, parameters={"Threads": 1, "Hash": 32})
            break
        except Exception:
            continue

    if sf is None:
        st.error("Engine Error: Stockfish not found.")

    return chat_model, astra_model, sf


CHAT_MODEL, ASTRA_MODEL, STOCKFISH = load_assets()


# ==========================================
# 6. Chess, Engine Ranking & Coaching Logic
# ==========================================
def render_board(board, flipped=False):
    """Renders the chess board SVG as HTML, flipped if playing as Black."""
    last_move = board.peek() if board.move_stack else None
    board_svg = chess.svg.board(board=board, lastmove=last_move, flipped=flipped, size=450)
    b64 = base64.b64encode(board_svg.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'


def get_ranked_move_options(board: chess.Board, num_options: int = 5):
    """
    Returns up to `num_options` legal moves ranked from best to next-best,
    along with a randomized display list for multiple-choice buttons.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return [], []

    limit = min(num_options, len(legal_moves))
    current_fen = board.fen()

    if current_fen in OPENING_BOOK_TOP5:
        ranked = [dict(item) for item in OPENING_BOOK_TOP5[current_fen][:limit]]
        shuffled = list(ranked)
        random.shuffle(shuffled)
        return ranked, shuffled

    ranked = []
    if STOCKFISH:
        try:
            STOCKFISH.set_depth(8)
            STOCKFISH.set_fen_position(current_fen)
            top_moves = STOCKFISH.get_top_moves(limit)

            for idx, item in enumerate(top_moves, 1):
                uci_str = item.get("Move")
                move_obj = chess.Move.from_uci(uci_str)
                if move_obj in board.legal_moves:
                    san_str = board.san(move_obj)
                    cp = item.get("Centipawn")
                    mate = item.get("Mate")
                    if mate is not None:
                        eval_str = f"Mate in {abs(mate)}"
                    elif cp is not None:
                        eval_str = f"{cp / 100:+.2f}"
                    else:
                        eval_str = "N/A"
                    ranked.append(
                        {"rank": idx, "san": san_str, "uci": uci_str, "eval": eval_str}
                    )
        except Exception:
            ranked = []

    if len(ranked) < limit:
        existing_sans = {r["san"] for r in ranked}
        for m in legal_moves:
            if len(ranked) >= limit:
                break
            san_str = board.san(m)
            if san_str not in existing_sans:
                ranked.append(
                    {
                        "rank": len(ranked) + 1,
                        "san": san_str,
                        "uci": m.uci(),
                        "eval": "N/A",
                    }
                )
                existing_sans.add(san_str)

    shuffled = list(ranked)
    random.shuffle(shuffled)
    return ranked, shuffled


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


def generate_turn_coaching_bundle(
    fen_before_move: str,
    board_after: chess.Board,
    move_history: list,
    ranked_options: list,
    chosen_san: str,
    ai_san: str | None,
    player_color: chess.Color,
    opening_name: str,
) -> tuple[str, str]:
    """
    Single fast LLM call returning BOTH:
    1. Ranked feedback on all 5 candidate options (starting with #1 Best Move).
    2. Socratic guided chat feedback evaluating the played move vs. AI reply.
    """
    user_color_str = "White" if player_color == chess.WHITE else "Black"
    moves_str = ", ".join([m["san"] for m in move_history])
    options_summary = "\n".join(
        [
            f"Rank #{opt['rank']}: {opt['san']} (Eval: {opt['eval']})"
            + (" <-- STUDENT PLAYED" if opt["san"] == chosen_san else "")
            for opt in ranked_options
        ]
    )

    system_instructions = (
        "You are a concise, Socratic Chess Grandmaster Coach.\n"
        f"Opening: {opening_name} | Student Color: {user_color_str}\n"
        f"FEN before student move: {fen_before_move}\n"
        f"Student Played: {chosen_san} | AI Replied: {ai_san or 'Game Over'}\n"
        f"Full Move History: {moves_str}\n\n"
        f"Candidate Options Ranked Best to 5th Best:\n{options_summary}\n\n"
        "Output TWO clearly separated sections using the exact delimiter `===CHAT_COACH===`:\n\n"
        "SECTION 1 (Before delimiter): Ranked breakdown of EVERY option starting with **#1 (Best Move)** down to the last option. "
        "Give 1 crisp bullet per option explaining why it is best, playable, or flawed, and tag the student's move with `(Your Move)`.\n\n"
        "===CHAT_COACH===\n\n"
        "SECTION 2 (After delimiter): Socratic chat reply to the student:\n"
        f"- Evaluate **{chosen_san}** relative to the opening and the AI's reply (**{ai_san or 'Game Over'}**).\n"
        "- If GOOD: give an encouraging message.\n"
        "- If FLAWED: explain the flaw and give 1-2 hints.\n"
        "- CRITICAL: NEVER reveal or name the exact move or destination square to play next.\n"
        "- End by asking the student for their next move."
    )

    try:
        response = ASTRA_MODEL.invoke(
            [
                SystemMessage(content=system_instructions),
                HumanMessage(content=f"I played {chosen_san}. Review the 5 options and guide my next turn."),
            ]
        )
        raw = response.content
        if "===CHAT_COACH===" in raw:
            options_part, chat_part = raw.split("===CHAT_COACH===", 1)
            return options_part.strip(), chat_part.strip()
        return raw.strip(), raw.strip()
    except Exception:
        lines = []
        for opt in ranked_options:
            tag = " **(Your Move)**" if opt["san"] == chosen_san else ""
            best_label = " — Best Move" if opt["rank"] == 1 else ""
            lines.append(f"- **#{opt['rank']}. {opt['san']}** (`{opt['eval']}`){best_label}{tag}")
        if ranked_options and chosen_san not in {o["san"] for o in ranked_options}:
            lines.append(f"- **Your Move ({chosen_san}):** Played outside the top {len(ranked_options)} options.")

        next_prompt = "What is your next move?" if not board_after.is_game_over() else "Game over!"
        chat_fallback = (
            f"You played **{chosen_san}**"
            + (f" and the AI answered with **{ai_san}**." if ai_san else ".")
            + f" Look at how the center and piece activity changed. {next_prompt}"
        )
        return "\n".join(lines), chat_fallback


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
                "content": "Welcome! Pick one of the 5 candidate opening moves below the board or type your move to begin. What is your first move?",
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
    try:
        STOCKFISH.set_skill_level(skill_level)
        STOCKFISH.set_fen_position(board.fen())
        best_move_uci = STOCKFISH.get_best_move_time(50)

        if best_move_uci:
            ai_move = chess.Move.from_uci(best_move_uci)
            ai_san = board.san(ai_move)
            board.push(ai_move)
            return ai_san
    except Exception:
        legal = list(board.legal_moves)
        if legal:
            ai_move = random.choice(legal)
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


def process_player_turn(
    move: chess.Move,
    base_time: int | None,
    increment: int,
    current_skill_level: int,
):
    """Executes player move, plays AI reply, and updates ranked options & coaching chat."""
    fen_before = st.session_state.board.fen()
    ranked_snapshot = list(st.session_state.get("current_ranked_options", []))

    if base_time is not None:
        if st.session_state.player_color == chess.WHITE:
            st.session_state.white_time += increment
        else:
            st.session_state.black_time += increment

    san_move = st.session_state.board.san(move)
    st.session_state.board.push(move)
    st.session_state.move_history.append({"san": san_move, "verdict": "Analyzing..."})

    st.session_state.options_fen = None
    st.session_state.current_ranked_options = []
    st.session_state.current_shuffled_options = []

    ai_san = None
    if not st.session_state.board.is_game_over():
        ai_san = execute_ai_move(st.session_state.board, current_skill_level)
        if ai_san:
            st.session_state.move_history.append({"san": ai_san, "verdict": "AI"})

    options_feedback, guided_reply = generate_turn_coaching_bundle(
        fen_before_move=fen_before,
        board_after=st.session_state.board,
        move_history=st.session_state.move_history,
        ranked_options=ranked_snapshot,
        chosen_san=san_move,
        ai_san=ai_san,
        player_color=st.session_state.player_color,
        opening_name=st.session_state.selected_opening,
    )
    st.session_state.last_options_feedback = options_feedback

    st.session_state.is_analyzing = True
    st.session_state.analysis_future = ANALYSIS_EXECUTOR.submit(
        analyze_move_task, list(st.session_state.move_history)
    )

    move_log = f"Played **{san_move}**" + (f" (AI replied **{ai_san}**)" if ai_san else "")
    st.session_state.messages.append({"role": "user", "content": move_log})
    st.session_state.messages.append({"role": "assistant", "content": guided_reply})


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

    st.session_state.options_fen = None
    st.session_state.current_ranked_options = []
    st.session_state.current_shuffled_options = []
    st.session_state.last_options_feedback = ""
    st.session_state.takeback_used = True
    st.session_state.last_move_time = time.time()
    st.session_state.coach_text = "Takeback used! You have 0 takebacks remaining. Make your new move."
    st.session_state.messages.append(
        {"role": "assistant", "content": "⚠️ Takeback used and position reverted. What is your new move?"}
    )


# ==========================================
# 7. OpenAI Dynamic Puzzle Generator & Spaced Repetition
# ==========================================
def validate_and_verify_puzzle(candidate: GeneratedPuzzle, section: str, difficulty: int) -> dict | None:
    """
    Validates that the LLM-generated FEN is legal, the board is not already over,
    all solution moves are strictly legal SANs, and aligns the first move with Stockfish.
    """
    try:
        board = chess.Board(candidate.fen)
        if not board.is_valid() or board.is_game_over() or not candidate.solution:
            return None

        verified_solution = []
        sim_board = board.copy()

        for idx, san_str in enumerate(candidate.solution):
            cleaned_san = san_str.strip()
            move = sim_board.parse_san(cleaned_san)
            if move not in sim_board.legal_moves:
                return None

            if idx == 0 and STOCKFISH:
                try:
                    STOCKFISH.set_skill_level(20)
                    STOCKFISH.set_depth(10)
                    STOCKFISH.set_fen_position(sim_board.fen())
                    sf_best_uci = STOCKFISH.get_best_move_time(80)
                    if sf_best_uci:
                        sf_move = chess.Move.from_uci(sf_best_uci)
                        if sf_move in sim_board.legal_moves and sf_move != move:
                            test_b = sim_board.copy()
                            test_b.push(move)
                            if not test_b.is_checkmate():
                                verified_solution = [sim_board.san(sf_move)]
                                break
                except Exception:
                    pass

            verified_solution.append(sim_board.san(move))
            sim_board.push(move)
            if sim_board.is_game_over():
                break

        if not verified_solution:
            return None

        return {
            "id": f"ai_{section.lower()}_{uuid.uuid4().hex[:8]}",
            "difficulty": difficulty,
            "title": candidate.title,
            "fen": board.fen(),
            "solution": verified_solution,
            "theme": candidate.theme,
            "hint": candidate.hint,
            "source": "OpenAI Generated",
        }
    except Exception:
        return None


def generate_ai_puzzle(
    section: str,
    difficulty: int,
    seen_titles: list,
    seen_fens: list,
    target_weak_theme: str | None = None,
) -> dict:
    """
    Calls OpenAI structured output (`ASTRA_MODEL`) to generate a brand-new, non-repeating
    chess puzzle tailored to `section` and `difficulty`. If `target_weak_theme` is provided
    from the player's missed-puzzle queue, composes a fresh reinforcement position on that motif.
    """
    difficulty_guide = {
        1: "Level 1 (Beginner): 1-move mate, hanging piece capture, or simple 1-move fork.",
        2: "Level 2 (Developing): 1-to-3 ply tactical motif (pin, skewer, discovered attack, or opening trap).",
        3: "Level 3 (Intermediate): 3-ply forcing combination, sacrifice on f7/h7, or key endgame opposition.",
        4: "Level 4 (Advanced): Multi-move tactical sequence, Lucena/Philidor rook technique, or deep counter-attack.",
        5: "Level 5+ (Master): Complex quiet move, Zugzwang, or multi-step sacrificial combination.",
    }
    diff_desc = difficulty_guide.get(min(difficulty, 5), difficulty_guide[5])

    avoid_titles = ", ".join(seen_titles[-15:]) if seen_titles else "None yet"
    avoid_fens = "\n".join(seen_fens[-10:]) if seen_fens else "None yet"

    reinforcement_directive = (
        f"PRIORITY REINFORCEMENT FOCUS: The student recently struggled with the tactical/positional motif **'{target_weak_theme}'**. "
        f"Compose a BRAND-NEW board position (different FEN and piece placement) that tests this exact '{target_weak_theme}' concept so they can practice and master it!"
        if target_weak_theme
        else "Choose a fresh tactical or positional motif appropriate for this game phase."
    )

    system_prompt = (
        "You are a Chess Grandmaster Puzzle Composer generating unique, verified tactical and positional puzzles.\n"
        f"Phase Section: {section}\n"
        f"Target Difficulty: {diff_desc}\n"
        f"{reinforcement_directive}\n\n"
        "STRICT COMPOSITION RULES:\n"
        "1. Compose a completely NEW, realistic chess position appropriate for the requested phase.\n"
        "2. Ensure the FEN is 100% legal (exactly one White king, one Black king, no pawns on 1st/8th ranks, side not to move is NOT in check).\n"
        "3. Provide exact legal SAN moves in `solution` (1 move for 1-ply puzzles, or 3 moves [player, opponent forced reply, player finisher] for combinations).\n"
        f"4. NEVER repeat any of these previously generated puzzle titles: [{avoid_titles}]\n"
        f"5. NEVER reuse any of these previous FENs:\n{avoid_fens}"
    )

    structured_llm = ASTRA_MODEL.with_structured_output(GeneratedPuzzle)

    for _ in range(2):
        try:
            candidate = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=f"Generate a fresh, non-repeating {section} chess puzzle at Difficulty Level {difficulty}."
                    ),
                ]
            )
            if candidate and candidate.fen not in seen_fens:
                verified = validate_and_verify_puzzle(candidate, section, difficulty)
                if verified:
                    verified["is_reinforcement"] = bool(target_weak_theme)
                    return verified
        except Exception:
            continue

    backup_pool = FALLBACK_PUZZLES[section]
    unseen_backups = [p for p in backup_pool if p["fen"] not in seen_fens]
    chosen = dict(random.choice(unseen_backups or backup_pool))
    chosen["difficulty"] = difficulty
    chosen["source"] = "Curated Backup"
    chosen["is_reinforcement"] = bool(target_weak_theme)
    return chosen


def load_next_puzzle(section: str):
    """Generates and loads a new OpenAI puzzle, prioritizing weak themes from missed attempts."""
    stats = st.session_state.puzzle_stats[section]
    target_diff = stats["level"]
    missed_queue = stats.get("missed_themes", [])
    target_weak_theme = missed_queue[0] if missed_queue else None

    spinner_msg = (
        f"🔁 Generating reinforcement {section} puzzle on **{target_weak_theme}**..."
        if target_weak_theme
        else f"✨ Generating unique Level {target_diff} {section} puzzle via OpenAI..."
    )

    with st.spinner(spinner_msg):
        puzzle = generate_ai_puzzle(
            section=section,
            difficulty=target_diff,
            seen_titles=stats["seen_titles"],
            seen_fens=stats["seen_fens"],
            target_weak_theme=target_weak_theme,
        )

    stats["seen_titles"].append(puzzle["title"])
    stats["seen_fens"].append(puzzle["fen"])

    st.session_state.current_puzzle = puzzle
    st.session_state.puzzle_board = chess.Board(puzzle["fen"])
    st.session_state.puzzle_step = 0
    st.session_state.puzzle_solved = False
    st.session_state.puzzle_counted_attempt = False
    st.session_state.puzzle_failed_attempts = 0
    st.session_state.puzzle_options_fen = None
    st.session_state.puzzle_shuffled_options = []

    if st.session_state.player_name:
        save_puzzle_progress(st.session_state.player_name, st.session_state.puzzle_stats)

    turn_label = "White" if st.session_state.puzzle_board.turn == chess.WHITE else "Black"
    reinforce_tag = (
        f"\n\n🎯 *Reinforcement Drill: Practicing **{target_weak_theme}** from a previous miss.*"
        if target_weak_theme
        else ""
    )

    st.session_state.puzzle_feedback = (
        f"🧩 **Level {puzzle['difficulty']} — {puzzle['title']}** *(Theme: {puzzle['theme']})*\n\n"
        f"You are playing **{turn_label}**. Find the strongest move!{reinforce_tag}"
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                f"🧩 **New {section} Puzzle (Level {puzzle['difficulty']}): {puzzle['title']}**\n\n"
                f"Theme: *{puzzle['theme']}* — Find the best move for **{turn_label}**!{reinforce_tag}"
            ),
        }
    )


def evaluate_puzzle_attempt(move: chess.Move):
    """
    Validates a player's move against the puzzle solution, records Attempted vs. Correct
    in DynamoDB, queues missed themes for repetition, and scales difficulty.
    """
    puzzle = st.session_state.current_puzzle
    board = st.session_state.puzzle_board
    step = st.session_state.puzzle_step
    section = st.session_state.puzzle_section
    stats = st.session_state.puzzle_stats[section]

    expected_san = puzzle["solution"][step]
    played_san = board.san(move)

    board_copy = board.copy()
    board_copy.push(move)
    is_correct = (played_san == expected_san) or board_copy.is_checkmate()

    if is_correct:
        board.push(move)
        st.session_state.puzzle_step += 1
        st.session_state.puzzle_options_fen = None
        st.session_state.puzzle_shuffled_options = []

        if st.session_state.puzzle_step < len(puzzle["solution"]):
            opp_san = puzzle["solution"][st.session_state.puzzle_step]
            opp_move = board.parse_san(opp_san)
            board.push(opp_move)
            st.session_state.puzzle_step += 1

            msg = (
                f"✅ **{played_san}** is spot-on! The opponent replied **{opp_san}**. "
                "Keep calculating—what is your winning follow-up move?"
            )
            st.session_state.puzzle_feedback = msg
            st.session_state.messages.append({"role": "user", "content": f"Puzzle move: **{played_san}**"})
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            st.session_state.puzzle_solved = True

            # Count each puzzle once in Attempted & Correct (first-try clean solve counts as correct)
            if not st.session_state.puzzle_counted_attempt:
                stats["attempted"] += 1
                if st.session_state.puzzle_failed_attempts == 0:
                    stats["correct"] += 1
                    stats["streak"] += 1
                st.session_state.puzzle_counted_attempt = True

            # If this puzzle was reinforcing a missed theme and solved cleanly, clear that theme from retry queue
            cleared_theme_note = ""
            if puzzle.get("theme") in stats["missed_themes"] and st.session_state.puzzle_failed_attempts == 0:
                stats["missed_themes"].remove(puzzle["theme"])
                cleared_theme_note = f"\n\n✨ **Pattern Mastered!** Cleared **{puzzle['theme']}** from your practice queue."

            leveled_up = False
            if stats["streak"] >= 2:
                stats["level"] += 1
                stats["streak"] = 0
                leveled_up = True

            if st.session_state.player_name:
                save_puzzle_progress(st.session_state.player_name, st.session_state.puzzle_stats)

            acc_pct = (stats["correct"] / stats["attempted"] * 100) if stats["attempted"] else 0.0
            score_line = f"📊 **{section} Record:** **{stats['correct']} / {stats['attempted']}** correct ({acc_pct:.0f}%)"

            level_note = (
                f"\n\n🚀 **Level Up!** You've advanced to **Level {stats['level']}** in {section} puzzles!"
                if leveled_up
                else f"\n\n🔥 Streak: **{stats['streak']}/2** (Solve {2 - stats['streak']} more cleanly to reach Level {stats['level'] + 1}!)"
            )

            msg = (
                f"🎉 **Puzzle Solved!** **{played_san}** nails the **{puzzle['theme']}** pattern.\n\n"
                f"{score_line}{cleared_theme_note}{level_note}\n\nClick **Next Puzzle ➡️** to continue!"
            )
            st.session_state.puzzle_feedback = msg
            st.session_state.messages.append({"role": "user", "content": f"Puzzle move: **{played_san}**"})
            st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        st.session_state.puzzle_failed_attempts += 1
        stats["streak"] = 0

        # Record failed attempt once and queue this theme for spaced repetition
        if not st.session_state.puzzle_counted_attempt:
            stats["attempted"] += 1
            st.session_state.puzzle_counted_attempt = True

        theme_name = puzzle.get("theme", "Tactical Motif")
        if theme_name not in stats["missed_themes"]:
            stats["missed_themes"].append(theme_name)

        leveled_down = False
        if st.session_state.puzzle_failed_attempts >= 2 and stats["level"] > 1:
            stats["level"] -= 1
            leveled_down = True

        if st.session_state.player_name:
            save_puzzle_progress(st.session_state.player_name, st.session_state.puzzle_stats)

        flaw_prompt = (
            "You are a Socratic Chess Puzzle Coach.\n"
            f"Puzzle Title: {puzzle['title']} | Theme: {puzzle['theme']}\n"
            f"Current FEN: {board.fen()}\n"
            f"Student tried: {played_san} (Incorrect)\n"
            f"Coach Hint: {puzzle['hint']}\n\n"
            "STRICT RULES:\n"
            f"1. Explain briefly in 1-2 sentences why **{played_san}** fails or misses the tactic.\n"
            "2. Give a helpful hint based on the theme without naming the solution move or destination square.\n"
            "3. Ask the student to try again."
        )
        try:
            resp = ASTRA_MODEL.invoke(
                [
                    SystemMessage(content=flaw_prompt),
                    HumanMessage(content=f"I tried {played_san} in this puzzle. Why didn't it work?"),
                ]
            )
            coach_hint_msg = resp.content
        except Exception:
            coach_hint_msg = (
                f"❌ **{played_san}** isn't the strongest move here. "
                f"**Hint:** {puzzle['hint']} Try again!"
            )

        coach_hint_msg += f"\n\n🔁 *Queued **{theme_name}** so we'll practice similar positions next.*"
        if leveled_down:
            coach_hint_msg += f"\n📉 *Difficulty adjusted to **Level {stats['level']}** to reinforce fundamentals.*"

        st.session_state.puzzle_feedback = coach_hint_msg
        st.session_state.messages.append({"role": "user", "content": f"Tried **{played_san}**"})
        st.session_state.messages.append({"role": "assistant", "content": coach_hint_msg})


# ==========================================
# 8. Session State & Opening Setup
# ==========================================
def init_session_state():
    """Initializes each session state key independently to survive hot-reloads."""
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Play Game"
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "move_history" not in st.session_state:
        st.session_state.move_history = []
    if "coach_text" not in st.session_state:
        st.session_state.coach_text = "Welcome! Select an opening or pick one of the 5 candidate moves below the board."
    if "player_name" not in st.session_state:
        st.session_state.player_name = ""
    if "loaded_puzzle_player" not in st.session_state:
        st.session_state.loaded_puzzle_player = ""
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
    if "options_fen" not in st.session_state:
        st.session_state.options_fen = None
    if "current_ranked_options" not in st.session_state:
        st.session_state.current_ranked_options = []
    if "current_shuffled_options" not in st.session_state:
        st.session_state.current_shuffled_options = []
    if "last_options_feedback" not in st.session_state:
        st.session_state.last_options_feedback = ""

    # Puzzle mode session state
    if "puzzle_section" not in st.session_state:
        st.session_state.puzzle_section = "Opening"
    if "puzzle_stats" not in st.session_state:
        st.session_state.puzzle_stats = default_puzzle_stats()
    if "current_puzzle" not in st.session_state:
        st.session_state.current_puzzle = None
    if "puzzle_board" not in st.session_state:
        st.session_state.puzzle_board = chess.Board()
    if "puzzle_step" not in st.session_state:
        st.session_state.puzzle_step = 0
    if "puzzle_solved" not in st.session_state:
        st.session_state.puzzle_solved = False
    if "puzzle_counted_attempt" not in st.session_state:
        st.session_state.puzzle_counted_attempt = False
    if "puzzle_failed_attempts" not in st.session_state:
        st.session_state.puzzle_failed_attempts = 0
    if "puzzle_feedback" not in st.session_state:
        st.session_state.puzzle_feedback = ""
    if "puzzle_options_fen" not in st.session_state:
        st.session_state.puzzle_options_fen = None
    if "puzzle_shuffled_options" not in st.session_state:
        st.session_state.puzzle_shuffled_options = []

    init_chat_history()


def reset_game(time_control_name, player_color, skill_level, opening_name="Free Play (Standard Start)"):
    """Archives current chat, sets up chosen opening & color, and prepares top-5 options."""
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
    st.session_state.options_fen = None
    st.session_state.current_ranked_options = []
    st.session_state.current_shuffled_options = []
    st.session_state.last_options_feedback = ""

    if "analysis_future" in st.session_state:
        del st.session_state["analysis_future"]

    opening_cfg = OPENINGS.get(opening_name, OPENINGS["Free Play (Standard Start)"])
    effective_color = opening_cfg["color"] if opening_cfg["color"] is not None else player_color
    st.session_state.player_color = effective_color

    base_time, _ = TIME_CONTROLS.get(time_control_name, (None, 0))
    st.session_state.white_time = base_time
    st.session_state.black_time = base_time
    st.session_state.last_move_time = time.time()

    for san in opening_cfg["moves"]:
        move = st.session_state.board.parse_san(san)
        st.session_state.board.push(move)
        st.session_state.move_history.append({"san": san, "verdict": "Book"})

    ai_opening_reply = None
    if st.session_state.board.turn != effective_color and not st.session_state.board.is_game_over():
        ai_opening_reply = execute_ai_move(st.session_state.board, skill_level)
        if ai_opening_reply:
            st.session_state.move_history.append({"san": ai_opening_reply, "verdict": "AI"})

    ranked_opts, shuffled_opts = get_ranked_move_options(st.session_state.board, num_options=5)
    st.session_state.options_fen = st.session_state.board.fen()
    st.session_state.current_ranked_options = ranked_opts
    st.session_state.current_shuffled_options = shuffled_opts

    side_label = "White" if effective_color == chess.WHITE else "Black"
    book_seq = " ".join(opening_cfg["moves"])

    if opening_cfg["moves"]:
        intro_msg = (
            f"🎯 **Opening Loaded: {opening_name}** (Playing as **{side_label}**)\n\n"
            f"*{opening_cfg['description']}*\n\n"
            f"Book moves played: `{book_seq}`"
            + (f" — AI followed up with **{ai_opening_reply}**." if ai_opening_reply else ".")
            + "\n\nPick one of the 5 candidate moves below the board or type your own! What is your next move?"
        )
    else:
        intro_msg = (
            f"♟️ **New Game Started as {side_label}**"
            + (f" — AI (White) opened with **{ai_opening_reply}**." if ai_opening_reply else ".")
            + "\n\nChoose from the 5 top candidate moves below the board or enter your move. What is your next move?"
        )

    st.session_state.messages = [{"role": "assistant", "content": intro_msg}]
    st.session_state.coach_text = f"Playing as **{side_label}** ({opening_name}). Pick a move from the 5 options below the board!"
    st.rerun()


# ==========================================
# 9. UI Components & Main App
# ==========================================
def main():
    st.set_page_config(page_title="AI Chess Coach & Adaptive Puzzles", layout="wide")
    init_session_state()

    # --- Sidebar: Mode Selection & Profile ---
    st.sidebar.header("Training Mode")
    selected_mode = st.sidebar.radio(
        "Choose Activity:",
        options=["Play Game", "Chess Puzzles"],
        index=0 if st.session_state.app_mode == "Play Game" else 1,
        horizontal=True,
    )
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        if selected_mode == "Chess Puzzles" and st.session_state.current_puzzle is None:
            load_next_puzzle(st.session_state.puzzle_section)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("Player Profile")
    player_input = st.sidebar.text_input(
        "Enter Player Name:",
        value=st.session_state.player_name,
        placeholder="e.g. Kasparov",
    )
    if player_input:
        clean_name = player_input.strip()
        st.session_state.player_name = clean_name

        # Automatically load saved DynamoDB puzzle record when player name is entered/changed
        if st.session_state.loaded_puzzle_player.lower() != clean_name.lower():
            st.session_state.puzzle_stats = load_puzzle_progress(clean_name)
            st.session_state.loaded_puzzle_player = clean_name

        stats = get_player_stats(clean_name)
        total_p_att = sum(st.session_state.puzzle_stats[s]["attempted"] for s in ["Opening", "Middlegame", "Endgame"])
        total_p_cor = sum(st.session_state.puzzle_stats[s]["correct"] for s in ["Opening", "Middlegame", "Endgame"])
        p_acc = (total_p_cor / total_p_att * 100) if total_p_att else 0.0

        if stats["is_new"] and total_p_att == 0:
            st.sidebar.info(f"Welcome, **{clean_name}**! New profile ready.")
        else:
            win_rate = (stats["wins"] / stats["total_games"] * 100) if stats["total_games"] else 0
            st.sidebar.success(
                f"**{clean_name}**'s Record:\n\n"
                f"- Match Wins: **{stats['wins']}** / {stats['total_games']} ({win_rate:.1f}%)\n"
                f"- Puzzles Solved: **{total_p_cor}** / {total_p_att} ({p_acc:.0f}%)"
            )
    else:
        st.sidebar.warning("Enter your name to save & restore your match and puzzle records in DynamoDB.")

    st.sidebar.markdown("---")

    # ==========================================
    # MODE A: AI-GENERATED CHESS PUZZLES (3 SECTIONS)
    # ==========================================
    if st.session_state.app_mode == "Chess Puzzles":
        st.sidebar.header("Puzzle Section & Progress")
        section = st.sidebar.radio(
            "Select Phase:",
            options=["Opening", "Middlegame", "Endgame"],
            index=["Opening", "Middlegame", "Endgame"].index(st.session_state.puzzle_section),
        )
        if section != st.session_state.puzzle_section:
            st.session_state.puzzle_section = section
            load_next_puzzle(section)
            st.rerun()

        p_stats = st.session_state.puzzle_stats[section]
        sec_acc = (p_stats["correct"] / p_stats["attempted"] * 100) if p_stats["attempted"] else 0.0

        c_m1, c_m2 = st.sidebar.columns(2)
        c_m1.metric("Difficulty", f"Lvl {p_stats['level']}")
        c_m2.metric("Score", f"{p_stats['correct']}/{p_stats['attempted']}", f"{sec_acc:.0f}%")

        st.sidebar.progress(min(1.0, p_stats["streak"] / 2))
        st.sidebar.caption(f"🔥 Streak to Level {p_stats['level'] + 1}: **{p_stats['streak']}/2**")

        if p_stats.get("missed_themes"):
            st.sidebar.warning(
                "🔁 **Focus Themes (Missed):**\n- " + "\n- ".join(p_stats["missed_themes"])
            )

        if st.sidebar.button("🔄 Reset Section Progress", use_container_width=True):
            st.session_state.puzzle_stats[section] = default_puzzle_stats()[section]
            if st.session_state.player_name:
                save_puzzle_progress(st.session_state.player_name, st.session_state.puzzle_stats)
            load_next_puzzle(section)
            st.rerun()

        if st.session_state.current_puzzle is None:
            load_next_puzzle(section)

        puzzle = st.session_state.current_puzzle
        p_board = st.session_state.puzzle_board
        is_black_to_move = chess.Board(puzzle["fen"]).turn == chess.BLACK

        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader(f"🧩 {section} Puzzle — Level {puzzle['difficulty']}: {puzzle['title']}")
            st.caption(
                f"Theme: **{puzzle['theme']}** | "
                f"Section Score: **{p_stats['correct']} / {p_stats['attempted']}** ({sec_acc:.0f}%) | "
                f"Source: `{puzzle.get('source', 'OpenAI Generated')}`"
            )

            p_error = st.empty()
            st.write(render_board(p_board, flipped=is_black_to_move), unsafe_allow_html=True)

            if not st.session_state.puzzle_solved:
                with st.form(key="puzzle_move_form", clear_on_submit=True):
                    p_move_input = st.text_input("Enter Best Move (SAN e.g., Qxf7#, Rd8+, Nc7+):")
                    p_submit = st.form_submit_button("Submit Move")

                if p_submit and p_move_input:
                    try:
                        move = parse_player_move(p_board, p_move_input)
                        if move in p_board.legal_moves:
                            evaluate_puzzle_attempt(move)
                            st.rerun()
                        else:
                            p_error.error("Illegal move in this puzzle position!")
                    except ValueError:
                        p_error.error("Invalid notation! Use SAN (e.g., Nf3, Qxf7+) or UCI.")

                cur_p_fen = p_board.fen()
                if st.session_state.puzzle_options_fen != cur_p_fen or not st.session_state.puzzle_shuffled_options:
                    _, shuffled_p_opts = get_ranked_move_options(p_board, num_options=5)
                    expected_san = puzzle["solution"][st.session_state.puzzle_step]
                    if expected_san not in {o["san"] for o in shuffled_p_opts}:
                        sol_move = p_board.parse_san(expected_san)
                        if shuffled_p_opts:
                            shuffled_p_opts[-1] = {
                                "rank": 1,
                                "san": expected_san,
                                "uci": sol_move.uci(),
                                "eval": "Best",
                            }
                        else:
                            shuffled_p_opts = [
                                {"rank": 1, "san": expected_san, "uci": sol_move.uci(), "eval": "Best"}
                            ]
                        random.shuffle(shuffled_p_opts)

                    st.session_state.puzzle_options_fen = cur_p_fen
                    st.session_state.puzzle_shuffled_options = shuffled_p_opts

                p_opts = st.session_state.puzzle_shuffled_options
                if p_opts:
                    st.markdown(f"**🎯 Candidate Moves ({len(p_opts)} choices — shuffled):**")
                    p_cols = st.columns(len(p_opts))
                    for idx, opt in enumerate(p_opts):
                        if p_cols[idx].button(
                            opt["san"],
                            key=f"puz_btn_{cur_p_fen}_{opt['uci']}",
                            use_container_width=True,
                        ):
                            chosen_m = chess.Move.from_uci(opt["uci"])
                            if chosen_m in p_board.legal_moves:
                                evaluate_puzzle_attempt(chosen_m)
                                st.rerun()

                c_hint, c_skip = st.columns(2)
                if c_hint.button("💡 Show Hint", use_container_width=True):
                    st.info(f"**Hint:** {puzzle['hint']}")
                if c_skip.button("✨ Generate New Puzzle", use_container_width=True):
                    load_next_puzzle(section)
                    st.rerun()
            else:
                if st.button("Next Puzzle ➡️", type="primary", use_container_width=True):
                    load_next_puzzle(section)
                    st.rerun()

            if st.session_state.puzzle_feedback:
                st.markdown("---")
                if st.session_state.puzzle_solved:
                    st.success(st.session_state.puzzle_feedback)
                elif st.session_state.puzzle_failed_attempts > 0:
                    st.warning(st.session_state.puzzle_feedback)
                else:
                    st.info(st.session_state.puzzle_feedback)

        with col2:
            st.subheader("Puzzle Coach Chat")
            chat_container = st.container(height=420, border=True)
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if p_query := st.chat_input("Ask Coach Astra for a hint on this puzzle..."):
                st.session_state.messages.append({"role": "user", "content": p_query})
                with st.spinner("Coach Astra is thinking..."):
                    reply = evaluate_candidate_move_with_astra(
                        p_board,
                        [],
                        p_query,
                        p_board.turn,
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

        return

    # ==========================================
    # MODE B: PLAY FULL GAME & OPENING TRAINER
    # ==========================================
    st.sidebar.header("Opening & Game Settings")

    opening_keys = list(OPENINGS.keys())
    current_opening = st.session_state.get("selected_opening", opening_keys[0])
    default_idx = opening_keys.index(current_opening) if current_opening in opening_keys else 0

    selected_opening = st.sidebar.selectbox(
        "Select Opening to Practice:",
        options=opening_keys,
        index=default_idx,
    )
    opening_info = OPENINGS[selected_opening]
    st.sidebar.caption(opening_info["description"])

    if opening_info["color"] is not None:
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

    if settings_changed and not has_player_moved:
        reset_game(selected_tc, new_player_color, current_skill_level, selected_opening)

    if (
        len(st.session_state.move_history) == 0
        and st.session_state.player_color == chess.BLACK
        and st.session_state.board.turn == chess.WHITE
    ):
        reset_game(selected_tc, chess.BLACK, current_skill_level, selected_opening)

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
                if not is_player_turn:
                    error_placeholder.warning("Please wait for your turn.")
                else:
                    try:
                        move = parse_player_move(st.session_state.board, move_input)
                        if move in st.session_state.board.legal_moves:
                            with st.spinner("Analyzing move options & AI response..."):
                                process_player_turn(move, base_time, increment, current_skill_level)
                            st.rerun()
                        else:
                            error_placeholder.error("Illegal move under current board rules!")
                    except ValueError:
                        error_placeholder.error("Invalid move! Use SAN (e.g., e4, Nf3, O-O) or squares (e1g1).")

            if is_player_turn:
                current_fen = st.session_state.board.fen()
                if st.session_state.options_fen != current_fen or not st.session_state.current_shuffled_options:
                    ranked_opts, shuffled_opts = get_ranked_move_options(st.session_state.board, num_options=5)
                    st.session_state.options_fen = current_fen
                    st.session_state.current_ranked_options = ranked_opts
                    st.session_state.current_shuffled_options = shuffled_opts

                shuffled_opts = st.session_state.current_shuffled_options
                side_text = "White" if st.session_state.player_color == chess.WHITE else "Black"
                if shuffled_opts:
                    st.markdown(
                        f"**🎯 Best {side_text} Move Challenge:** Pick one of the top {len(shuffled_opts)} candidate moves *(displayed in random order)*:"
                    )
                    btn_cols = st.columns(len(shuffled_opts))
                    for idx, opt in enumerate(shuffled_opts):
                        if btn_cols[idx].button(
                            opt["san"],
                            key=f"opt_btn_{current_fen}_{opt['uci']}",
                            use_container_width=True,
                        ):
                            chosen_move = chess.Move.from_uci(opt["uci"])
                            if chosen_move in st.session_state.board.legal_moves:
                                with st.spinner(f"Playing {opt['san']} and ranking all options..."):
                                    process_player_turn(chosen_move, base_time, increment, current_skill_level)
                                st.rerun()

            if st.session_state.last_options_feedback:
                st.markdown("---")
                st.markdown("#### 📊 Last Turn Options Breakdown *(Ranked Best → Next Best)*")
                st.info(st.session_state.last_options_feedback)

        else:
            if player_timed_out:
                st.error("Time out! You lost on time.")
            else:
                st.info("Game Over.")

            if st.session_state.last_options_feedback:
                st.markdown("#### 📊 Final Turn Options Breakdown *(Ranked Best → Next Best)*")
                st.info(st.session_state.last_options_feedback)

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

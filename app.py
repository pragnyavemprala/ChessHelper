import os
import re
import uuid
import base64
from datetime import datetime, timezone

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
    """
    Queries DynamoDB for player records and calculates win/total games.
    """
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
    """
    Saves the full game details to DynamoDB upon game completion.
    """
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
# 4. Chess & Coach Logic
# ==========================================
def render_board(board):
    """Renders the chess board SVG as HTML."""
    last_move = board.peek() if board.move_stack else None
    board_svg = chess.svg.board(board=board, lastmove=last_move, size=450)
    b64 = base64.b64encode(board_svg.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'


def get_coach_analysis(move_history):
    """Provides short tactical move feedback for the banner."""
    if not move_history:
        return ""
    moves_str = ", ".join([m["san"] for m in move_history])
    system_text = CONFIG["coach_prompt"].replace("{moves_str}", moves_str)

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=f"Current move history: {moves_str}. Analyze the last move played."),
    ]
    try:
        response = CHAT_MODEL.invoke(messages)
        return response.content
    except Exception as e:
        return f"Coach is currently offline: {e}"


def get_conversational_coach_reply(board, move_history, chat_messages):
    """
    Carries out a continuous conversational dialog with the coach.
    Preserves prior chat history and provides current board state/history context.
    """
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


def evaluate_verdict(analysis_text: str) -> str:
    """Parses coach analysis for classification tags."""
    for category in ["[GOOD]", "[INACCURACY]", "[MISTAKE]", "[BLUNDER]"]:
        if category in analysis_text.upper():
            return category
    return ""


def execute_ai_move(board, skill_level):
    """Executes a Stockfish counter-move."""
    if not STOCKFISH or board.is_game_over():
        return None
    STOCKFISH.set_skill_level(skill_level)
    STOCKFISH.set_fen_position(board.fen())
    think_time = 100 + (skill_level * 50)
    best_move_uci = STOCKFISH.get_best_move_time(think_time)

    if best_move_uci:
        ai_move = chess.Move.from_uci(best_move_uci)
        ai_san = board.san(ai_move)
        board.push(ai_move)
        return ai_san
    return None


def parse_player_move(board: chess.Board, text: str) -> chess.Move:
    """
    Parses user input into a valid chess.Move.
    Accepts standard SAN, common castling variants (0-0, o-o), and UCI moves (e1g1).
    """
    cleaned = text.strip()

    castling_map = {
        "0-0-0": "O-O-O",
        "o-o-o": "O-O-O",
        "ooo": "O-O-O",
        "0-0": "O-O",
        "o-o": "O-O",
        "oo": "O-O",
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


def load_game_from_notation(notation_text: str, current_skill_level: int):
    """Rebuilds game state and board from a raw SAN move list or PGN sequence."""
    raw = notation_text.replace(",", " ")
    tokens = re.split(r"\s+", raw.strip())
    valid_tokens = [
        t for t in tokens
        if t and not re.match(r"^\d+\.+$", t) and t not in ["1-0", "0-1", "1/2-1/2", "*"]
    ]

    temp_board = chess.Board()
    new_history = []

    for idx, token in enumerate(valid_tokens):
        token = re.sub(r"^\d+\.+", "", token).strip()
        if not token:
            continue
        try:
            move = parse_player_move(temp_board, token)
            san_str = temp_board.san(move)
            temp_board.push(move)
            verdict = "AI" if idx % 2 != 0 else ""
            new_history.append({"san": san_str, "verdict": verdict})
        except Exception:
            return False, f"Illegal or unrecognized move '{token}' at position {idx + 1}."

    st.session_state.board = temp_board
    st.session_state.move_history = new_history
    st.session_state.game_saved = False
    st.session_state.messages = []

    if temp_board.turn == chess.BLACK and not temp_board.is_game_over():
        ai_san = execute_ai_move(st.session_state.board, current_skill_level)
        if ai_san:
            st.session_state.move_history.append({"san": ai_san, "verdict": "AI"})

    st.session_state.coach_text = (
        f"Game restored to move {len(new_history)}. "
        f"{'White' if st.session_state.board.turn == chess.WHITE else 'Black'} to move."
    )
    return True, "Game position successfully loaded!"


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
            {"role": "assistant", "content": "Hello! I am your coach. Play a move, or ask me anything about chess strategy or the current position."}
        ]
        st.session_state.player_name = ""
        st.session_state.game_saved = False


def reset_game():
    """Resets board and in-game states while keeping player profile."""
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.coach_text = "Welcome! Make a move to start the session."
    st.session_state.messages = [
        {"role": "assistant", "content": "New game started! What would you like to open with?"}
    ]
    st.session_state.game_saved = False
    st.rerun()


# ==========================================
# 6. UI Components
# ==========================================
def render_player_header():
    """Renders player registration input and displays win statistics."""
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
            st.sidebar.info(f"Welcome, **{player_input}**! You are registered as a new player.")
        else:
            win_rate = (stats["wins"] / stats["total_games"] * 100) if stats["total_games"] else 0
            st.sidebar.success(
                f"**{player_input}**'s Record:\n\n"
                f"- Wins: **{stats['wins']}** / {stats['total_games']}\n"
                f"- Win Rate: **{win_rate:.1f}%**"
            )
    else:
        st.sidebar.warning("Please enter your name to track game progress.")


def render_sidebar_history():
    """Renders move list in reverse chronological order."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Move History")

    history = st.session_state.move_history
    pairs = []
    for i in range(0, len(history), 2):
        move_num = (i // 2) + 1
        w_data = history[i]
        b_data = history[i + 1] if i + 1 < len(history) else {"san": "...", "verdict": "AI"}
        pairs.append((move_num, w_data, b_data))

    for m_num, w, b in reversed(pairs):
        w_verdict = f" `{w['verdict']}`" if w["verdict"] else ""
        st.sidebar.markdown(f"**{m_num}.** {w['san']}{w_verdict} &nbsp;&nbsp; {b['san']}")


def handle_game_completion():
    """Handles end-of-game checks and triggers DynamoDB record write once."""
    if not st.session_state.board.is_game_over():
        return

    result = st.session_state.board.result()
    if result == "1-0":
        status = "Player Won"
    elif result == "0-1":
        status = "AI Won"
    else:
        status = "Draw"

    st.success(f"Game Over! Result: {result} ({status})")

    if not st.session_state.game_saved:
        with st.spinner("Generating match summary and saving to DynamoDB..."):
            summary = generate_game_summary(st.session_state.move_history, status)
            if st.session_state.player_name:
                saved = save_game_record(
                    st.session_state.player_name,
                    status,
                    st.session_state.move_history,
                    summary,
                )
                if saved:
                    st.toast("Game outcome recorded in DynamoDB!")
            st.session_state.game_saved = True


# ==========================================
# 7. Main Application Flow
# ==========================================
def main():
    st.set_page_config(page_title="AI Chess Coach", layout="wide")
    init_session_state()

    render_player_header()

    current_skill_level = st.sidebar.slider(
        "AI Difficulty (Skill Level)",
        min_value=1,
        max_value=20,
        value=5,
        help="1 is easiest, 20 is Grandmaster level.",
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        error_placeholder = st.empty()
        st.write(render_board(st.session_state.board), unsafe_allow_html=True)

        with st.form(key="chess_move_form", clear_on_submit=True):
            move_input = st.text_input("Enter Move (SAN notation, e.g., e4, Nf3, O-O):")
            submit_button = st.form_submit_button("Play Move")

        if submit_button and move_input:
            if not st.session_state.player_name:
                error_placeholder.warning("Please enter your player name in the sidebar before moving.")
            else:
                try:
                    move = parse_player_move(st.session_state.board, move_input)
                    if move in st.session_state.board.legal_moves:
                        san_move = st.session_state.board.san(move)
                        st.session_state.board.push(move)
                        st.session_state.move_history.append({"san": san_move, "verdict": ""})

                        with st.spinner("Grandmaster is analyzing your move..."):
                            analysis = get_coach_analysis(st.session_state.move_history)
                            st.session_state.coach_text = analysis
                            st.session_state.move_history[-1]["verdict"] = evaluate_verdict(analysis)

                        ai_san = execute_ai_move(st.session_state.board, current_skill_level)
                        if ai_san:
                            st.session_state.move_history.append({"san": ai_san, "verdict": "AI"})

                        st.rerun()
                    else:
                        error_placeholder.error("Illegal move under current board rules!")
                except ValueError:
                    error_placeholder.error("Invalid move! Use SAN (e.g., e4, Nf3, O-O) or squares (e1g1).")

    with col2:
        st.subheader("Coach's Corner")

        bg_color = "#f0f2f6"
        if st.session_state.move_history:
            history_len = len(st.session_state.move_history)
            target_idx = history_len - 1 if history_len % 2 != 0 else history_len - 2
            if target_idx >= 0:
                latest_verdict = st.session_state.move_history[target_idx].get("verdict", "").upper()
                if "[GOOD]" in latest_verdict:
                    bg_color = "#d4edda"
                elif "[INACCURACY]" in latest_verdict:
                    bg_color = "#fff3cd"
                elif "[MISTAKE]" in latest_verdict:
                    bg_color = "#ffe5d0"
                elif "[BLUNDER]" in latest_verdict:
                    bg_color = "#f8d7da"

        st.markdown(
            f"""
            <div style="background-color: {bg_color}; padding: 16px; border-radius: 8px; 
            border-left: 5px solid rgba(0,0,0,0.1); color: #000000; font-size: 15px; line-height: 1.5; margin-bottom: 12px;">
                {st.session_state.coach_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Snapshot / Game Restore Placeholder
        with st.expander("Resume Game from Snapshot Notation", expanded=False):
            st.caption("Paste full PGN or moves (e.g., `1. e4 e5 2. Nf3 Nc6` or `e4 e5 Nf3 Nc6`).")
            snapshot_input = st.text_area(
                "Game Notation:",
                placeholder="1. e4 e5 2. Nf3 Nc6 3. Bc4",
                key="snapshot_notation_input",
                height=80
            )
            if st.button("Load Snapshot & Continue"):
                if not st.session_state.player_name:
                    st.warning("Please enter your player name in the sidebar before restoring.")
                elif not snapshot_input.strip():
                    st.warning("Please paste valid game notation.")
                else:
                    success, msg = load_game_from_notation(snapshot_input, current_skill_level)
                    if success:
                        st.toast(msg, icon="♟️")
                        st.rerun()
                    else:
                        st.error(msg)

        # -------------------------------------------------------------
        # Interactive Multi-Turn Chat with Coach
        # -------------------------------------------------------------
        st.markdown("#### Conversation with Coach")
        chat_container = st.container(height=320)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if user_query := st.chat_input("Ask your coach about strategy, mistakes, or plans..."):
            # 1. Append player question
            st.session_state.messages.append({"role": "user", "content": user_query})

            # 2. Query LLM with full context
            with st.spinner("Coach is thinking..."):
                reply = get_conversational_coach_reply(
                    st.session_state.board,
                    st.session_state.move_history,
                    st.session_state.messages
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})

            # 3. Refresh display smoothly
            st.rerun()

        if st.button("Restart Game"):
            reset_game()

        handle_game_completion()

    render_sidebar_history()


if __name__ == "__main__":
    main()

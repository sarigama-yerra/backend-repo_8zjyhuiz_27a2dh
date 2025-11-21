import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import chess.pgn
import chess.engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LegalMovesRequest(BaseModel):
    fen: str
    square: str  # e.g., "e2"


class MakeMoveRequest(BaseModel):
    fen: str
    from_square: str  # e.g., "e2"
    to_square: str    # e.g., "e4"
    promotion: str | None = None  # e.g., "q"
    depth: int = 2


@app.get("/")
def read_root():
    return {"message": "Chess AI Backend Ready"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Simple health check; database optional for this app."""
    return {
        "backend": "✅ Running",
        "database": "ℹ️ Not required for chess session",
    }


def uci_to_san(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except Exception:
        return move.uci()


def evaluate_board(board: chess.Board) -> int:
    # Simple material evaluation
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }
    value = 0
    for piece_type in piece_values:
        value += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        value -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    # small bonus for mobility
    value += (len(list(board.legal_moves)) if board.turn == chess.WHITE else -len(list(board.legal_moves)))
    return value


def minimax(board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool) -> tuple[int, chess.Move | None]:
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move: chess.Move | None = None

    if maximizing:
        max_eval = -10_000_000
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = 10_000_000
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move


@app.post("/api/legal-moves")
def legal_moves(req: LegalMovesRequest):
    try:
        board = chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    try:
        sq = chess.SQUARE_NAMES.index(req.square)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid square")

    moves = []
    for move in board.legal_moves:
        if move.from_square == sq:
            moves.append({
                "to": chess.SQUARE_NAMES[move.to_square],
                "promotion": chess.piece_symbol(move.promotion) if move.promotion else None,
                "uci": move.uci(),
            })
    return {"moves": moves}


@app.post("/api/make-move")
def make_move(req: MakeMoveRequest):
    try:
        board = chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    try:
        move = chess.Move.from_uci(
            req.from_square + req.to_square + (req.promotion if req.promotion else "")
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid move format")

    if move not in board.legal_moves:
        raise HTTPException(status_code=400, detail="Illegal move")

    # Apply player's move
    board.push(move)
    player_san = uci_to_san(board.copy(stack=False), move)  # after push, SAN is known via previous board, but keep simple

    game_over = board.is_game_over()
    ai_move_san = None
    ai_move_uci = None

    if not game_over:
        # AI plays as the current side to move (after player's move)
        maximizing = board.turn == chess.WHITE
        _, best = minimax(board, max(1, min(3, req.depth)), -10_000_000, 10_000_000, maximizing)
        if best is None:
            # no legal moves -> game over
            game_over = True
        else:
            board.push(best)
            ai_move_uci = best.uci()
            ai_move_san = uci_to_san(board.copy(stack=False), best)

    result = None
    if board.is_checkmate():
        result = "checkmate"
    elif board.is_stalemate():
        result = "stalemate"
    elif board.is_insufficient_material():
        result = "insufficient_material"
    elif board.can_claim_fifty_moves():
        result = "fifty_move_rule"
    elif board.can_claim_threefold_repetition():
        result = "threefold_repetition"

    return {
        "fen": board.fen(),
        "player_move": {
            "uci": move.uci(),
            "san": player_san,
        },
        "ai_move": {
            "uci": ai_move_uci,
            "san": ai_move_san,
        } if ai_move_uci else None,
        "game_over": game_over or board.is_game_over(),
        "result": result,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "legal_in_check": board.is_check(),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

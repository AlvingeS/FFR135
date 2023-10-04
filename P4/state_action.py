
class State:
    def __init__(self, board_size: int = 3):
        self.board_size = board_size
        self.board = self.generate_board()

    def generate_board(self):
        board = []
        for _ in range(self.board_size):
            row = []
            for _ in range(self.board_size):
                row.append(0)
            board.append(tuple(row))
        return tuple(board)
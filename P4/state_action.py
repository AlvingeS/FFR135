from icecream import ic
class State:
    def __init__(self, board_size: int = 3):
        self.board_size = board_size
        self.board = self.generate_board()
        self.winner = None

    def generate_board(self):
        board = []
        for _ in range(self.board_size):
            row = []
            for _ in range(self.board_size):
                row.append(0)
            board.append(tuple(row))
        return tuple(board)
    
    def apply_action(self, action: (int, int), player_symbol):
        row, col = action
        if self.board[row][col] != 0:
            raise Exception("Invalid action")
        new_board = [list(row) for row in self.board]
        new_board[row][col] = player_symbol
        future_state = State(board_size=self.board_size)
        future_state.board=tuple([tuple(row) for row in new_board])
        return future_state

    def is_game_over(self):
        if self.is_won():
            return True
        elif self.is_full():
            self.winner = 0
            return True
        else:
            return False
    
    def is_won(self):
        return self.is_won_row() or self.is_won_col() or self.is_won_diagonal()

    def is_won_row(self):
        for row in self.board:
            if len(set(row)) == 1 and row[0] != 0:
                self.winner = row[0]
                return True
        return False
    
    def is_won_col(self):
        for col in range(self.board_size):
            if len(set([row[col] for row in self.board])) == 1 and self.board[0][col] != 0:
                self.winner = self.board[0][col]
                return True
        return False
    
    def is_won_diagonal(self):
        if len(set([self.board[i][i] for i in range(self.board_size)])) == 1 and self.board[0][0] != 0:
            self.winner = self.board[0][0]
            return True
        if len(set([self.board[i][self.board_size - i - 1] for i in range(self.board_size)])) == 1 and self.board[0][self.board_size - 1] != 0:
            self.winner = self.board[0][self.board_size - 1]
            return True
        return False
    
    def is_full(self):
        for row in self.board:
            if 0 in row:
                return False
        return True
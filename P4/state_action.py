
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
    
    def is_game_over(self):
        if self.is_won()[0]:
            return True, self.is_won()[1]
        elif self.is_full():
            return True, None
        else:
            return False, None
    
    def is_won(self):
        return self.is_won_row() or self.is_won_col() or self.is_won_diagonal()

    def is_won_row(self):
        for row in self.board:
            if len(set(row)) == 1 and row[0] != 0:
                return True, row[0]
        return False, None
    
    def is_won_col(self):
        for col in range(self.board_size):
            if len(set([row[col] for row in self.board])) == 1 and self.board[0][col] != 0:
                return True, self.board[0][col]
        return False, None
    
    def is_win_diagonal(self):
        if len(set([self.board[i][i] for i in range(self.board_size)])) == 1 and self.board[0][0] != 0:
            return True, self.board[0][0]
        if len(set([self.board[i][self.board_size - i - 1] for i in range(self.board_size)])) == 1 and self.board[0][self.board_size - 1] == 1:
            return True, self.board[0][self.board_size - 1]
        return False, None
    
    def is_full(self):
        for row in self.board:
            if 0 in row:
                return False
        return True
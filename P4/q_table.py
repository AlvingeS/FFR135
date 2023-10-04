from state_action import State, Action

class QTable:
    def __init__(self, board_size: int = 3):
        self.q_table = {}
        self.board_size = board_size
        self.all_possible_actions = self.generate_all_possible_actions()

    def generate_all_possible_actions(self):
        all_possible_actions = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                all_possible_actions.append((row, col))
        return all_possible_actions

    def initialize_state(self, state: State):
        legal_actions = [(i, j) for i, row in enumerate(state.board) for j, cell in enumerate(row) if cell == 0]
        self.q_table[state.board] = {action: 0 for action in legal_actions}

    
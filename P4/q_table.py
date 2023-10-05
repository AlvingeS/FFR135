from state_action import State, Action

class QTable:
    def __init__(self, board_size: int = 3, learning_rate: float = 0.1, ):
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

    def select_action(self, state: State):
        if state.board not in self.q_table:
            self.initialize_state(state)
        return self.select_best_action(state)
    
    def select_best_action(self, state: State):
        best_action = None
        best_value = float('-inf')
        for action, value in self.q_table[state.board].items():
            if value > best_value:
                best_value = value
                best_action = action
        return best_action    

    def update_q_value(state: State, action: (int, int)):
        self.q_table[state.board][action] +=         
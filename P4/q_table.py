from state_action import State
import random
import numpy as np
import csv

class QTable:
    def __init__(self, player_symbol: int, board_size: int = 3, learning_rate: float = 0.1, win_reward: float = 1, draw_punishment: float = 0, loss_punishment: float = -1, starting_epsilon: float = 0.25, minimum_epsilon: float = 0.01, epsilon_decay: float = 0.999):
        self.q_table = {}
        self.board_size = board_size
        self.all_possible_actions = self.generate_all_possible_actions()
        self.learning_rate = learning_rate
        self.player_symbol = player_symbol
        self.win_reward = win_reward
        self.draw_punishment = draw_punishment
        self.loss_punishment = loss_punishment
        self.last_state = None
        self.last_action = None
        self.epsilon = starting_epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay = epsilon_decay

    def generate_all_possible_actions(self):
        all_possible_actions = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                all_possible_actions.append((row, col))
        return all_possible_actions

    def initialize_state(self, state: State):
        legal_actions = [(i, j) for i, row in enumerate(state.board) for j, cell in enumerate(row) if cell == 0]
        self.q_table[state.board] = {action: 0 for action in legal_actions}

    def set_q_value(self, state: State, action: (int, int), value: float):
        self.q_table[state.board][action] = value

    def move(self, current_state: State):
        self.check_if_initialized(current_state)
        selected_action = self.select_action(current_state)
        opponents_next_state = current_state.apply_action(selected_action, self.player_symbol)
        game_over = opponents_next_state.is_game_over()
        
        if game_over:
            reward = self.calculate_reward(opponents_next_state, self.player_symbol, game_over)
            self.set_q_value(current_state, selected_action, reward)

        if self.last_state is not None:
            self.update_q_value(current_state)
        
        self.update_old_state_action(current_state, selected_action)
        return opponents_next_state, game_over
    
    def reset_last_state_action(self):
        self.last_state = None
        self.last_action = None


    def evaluation_move(self, current_state: State):
        selected_action = self.select_action(current_state)
        future_state = current_state.apply_action(selected_action, self.player_symbol)
        game_over = future_state.is_game_over()
        return future_state, game_over
    
    def check_if_initialized(self, state: State):
        if state.board not in self.q_table:
            self.initialize_state(state)
    
    def select_action(self, current_state: State):
        if random.random() < self.epsilon:
            # Selects a random move from all legal moves given the current state
            legal_actions = [(i, j) for i, row in enumerate(current_state.board) for j, cell in enumerate(row) if cell == 0]
            selected_action = random.choice(legal_actions)
            return selected_action
        else:
            best_value = float('-inf')
            best_actions = []
            
            for action, value in self.q_table[current_state.board].items():
                if value > best_value:
                    best_value = value
                    best_actions = [action]
                elif value == best_value:
                    best_actions.append(action)
            selected_action = random.choice(best_actions)
    
        return selected_action

    def update_old_state_action(self, current_state: State, selected_action: (int, int)):
        self.last_state = current_state
        self.last_action = selected_action


    def calculate_reward(self, future_state: State, player_symbol: int, game_over: bool):
        if game_over:
            if future_state.winner == player_symbol:
                return self.win_reward
            else:
                return self.draw_punishment
        else:
            return 0.0

    def update_q_value(self, current_state: State):
            update_rule = self.q_table[self.last_state.board][self.last_action] + self.learning_rate*(max(self.q_table[current_state.board].values()) - self.q_table[self.last_state.board][self.last_action])
            self.set_q_value(self.last_state, self.last_action, update_rule)

    def punish_last_move(self, punishment: float):
        self.set_q_value(self.last_state, self.last_action, punishment)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.minimum_epsilon)

    def save_q_table_to_csv(self, filename):
        num_states = len(self.q_table)
        
        board_matrix = np.zeros((3, num_states * 3), dtype=int)
        q_values_matrix = np.full((3, num_states * 3), 'NaN', dtype=object)
        
        for idx, (state, actions) in enumerate(self.q_table.items()):
            col_start = idx * 3
            col_end = (idx + 1) * 3
            
            board_matrix[:, col_start:col_end] = np.array(state).reshape(3, 3)
            
            for (row, col), q_value in actions.items():
                q_values_matrix[row, col_start + col] = q_value

        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            for row in board_matrix:
                csv_writer.writerow(row)
                
            for row in q_values_matrix:
                csv_writer.writerow(row)
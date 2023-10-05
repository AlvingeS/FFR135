

from state_action import State

state = State()
state.board = ((1, 1, 0), (-1, 0, 1), (1, 1, -1))

print(state.is_won())
print(state.is_full())
print(state.is_game_over())
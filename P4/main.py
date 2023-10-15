from state_action import State
from q_table import QTable
from icecream import ic
import matplotlib.pyplot as plt
import math

# Learning parameters
learning_rate = 0.1
starting_epsilon = 0.5
minimum_epsilon = 0.001
num_episodes = 1000000
full_decay_limit = 0.5*num_episodes
epsilon_decay = math.exp(math.log(minimum_epsilon / starting_epsilon) / full_decay_limit)

# Q-tables
players = [QTable(player_symbol=1, learning_rate=learning_rate, starting_epsilon=starting_epsilon, minimum_epsilon=minimum_epsilon, epsilon_decay=epsilon_decay),
           QTable(player_symbol=-1, learning_rate=learning_rate, starting_epsilon=starting_epsilon, minimum_epsilon=minimum_epsilon, epsilon_decay=epsilon_decay)]

# Win statistics
player_one_rates, player_two_rates, draw_rates, intervals = [], [], [], []
interval_length = 10000
results = {1: 0, -1: 0, 0: 0}

# Training loop
for i in range(num_episodes):
    current_state = State()
    turn = 0

    # Game loop
    while (True):
        new_state, game_over = players[turn].move(current_state)

        # On your own move, you can only win or draw the game
        if game_over:      
            if new_state.winner == players[turn].player_symbol:
                players[1 - turn].punish_last_move(players[1 - turn].loss_punishment)
            else:
                players[1 - turn].punish_last_move(players[1 - turn].draw_punishment)

            results[new_state.winner] += 1

            break

        turn = 1 - turn
        current_state = new_state
    
    # Reset last played moves/states and decay epsilon
    players[0].reset_last_state_action()
    players[1].reset_last_state_action()
    players[0].decay_epsilon()
    players[1].decay_epsilon()

    # Save win statistics
    if i % interval_length == 0:
        total_games = results[1] + results[-1] + results[0]
        player_one_rates.append(results[1] / total_games)
        player_two_rates.append(results[-1] / total_games)
        draw_rates.append(results[0] / total_games)
        intervals.append(i)

        results = {1: 0, -1: 0, 0: 0}

# Plotting
plt.plot(intervals, player_one_rates, label="Player 1")
plt.plot(intervals, player_two_rates, label="Player 2")
plt.plot(intervals, draw_rates, label="Draw")
plt.xlabel("Number of episodes")
plt.ylabel("Win rate")
plt.legend()
plt.savefig("rates.png")

# Evaluation
test_runs = 1000
results = {1: 0, -1: 0, 0: 0}
players[0].epsilon = 0
players[1].epsilon = 0

for i in range(test_runs):
    current_state = State()
    turn = 0

    while(True):
        new_state, game_over = players[turn].evaluation_move(current_state=current_state)

        if game_over:
            results[new_state.winner] += 1
            break

        turn = 1 - turn
        current_state = new_state

# Print results
print("Player 1 wins: " + str(results[1]))
print("Player 2 wins: " + str(results[-1]))
print("Draws: " + str(results[0]))

ic(players[0].q_table[State().board])

# Save Q-tables if all games were draws
if results[0] == test_runs:
    players[0].save_q_table_to_csv('player1.csv')
    players[1].save_q_table_to_csv('player2.csv')
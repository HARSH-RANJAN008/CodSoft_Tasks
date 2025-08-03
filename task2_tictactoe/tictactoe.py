import numpy as np
import random
import pickle
import os

# Represent board as a string of 9 chars: 'X', 'O', or '-'
def check_winner(state):
    b = list(state)
    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
    for a,b_idx,c in wins:
        if b[a] == b[b_idx] == b[c] and b[a] != '-':
            return b[a]
    if '-' not in b:
        return 'Draw'
    return None

def available_actions(state):
    return [i for i, ch in enumerate(state) if ch == '-']

def make_move(state, action, player):
    lst = list(state)
    lst[action] = player
    return ''.join(lst)

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, eps=0.2):
        self.q = {}  # state -> action-values dict
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def get_qs(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(9)  # 9 positions
        return self.q[state]

    def choose_action(self, state):
        if random.random() < self.eps:
            return random.choice(available_actions(state))
        qs = self.get_qs(state)
        # mask invalid
        acts = available_actions(state)
        vals = [(qs[a], a) for a in acts]
        return max(vals)[1]

    def update(self, state, action, reward, next_state, done):
        qs = self.get_qs(state)
        target = reward
        if not done:
            next_qs = self.get_qs(next_state)
            target += self.gamma * np.max(next_qs)
        qs[action] += self.alpha * (target - qs[action])
        self.q[state] = qs

def train(agent, episodes=50000):
    for ep in range(episodes):
        state = '---------'  # empty
        player = 'X'  # agent plays X
        while True:
            action = agent.choose_action(state)
            next_state = make_move(state, action, player)
            winner = check_winner(next_state)
            done = winner is not None
            reward = 0
            if done:
                if winner == 'X':
                    reward = 1
                elif winner == 'O':
                    reward = -1
                else:
                    reward = 0.5
            else:
                # opponent random move
                opp_actions = available_actions(next_state)
                if not opp_actions:
                    done = True
                    reward = 0.5
                    winner = 'Draw'
                else:
                    opp_move = random.choice(opp_actions)
                    next_state = make_move(next_state, opp_move, 'O')
                    winner = check_winner(next_state)
                    done = winner is not None
                    if done:
                        if winner == 'X':
                            reward = 1
                        elif winner == 'O':
                            reward = -1
                        else:
                            reward = 0.5

            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

def play_vs_agent(agent):
    state = '---------'
    while True:
        # human as O
        print_board(state)
        human_move = int(input("Your move (0-8): "))
        if state[human_move] != '-':
            print("Invalid")
            continue
        state = make_move(state, human_move, 'O')
        if check_winner(state):
            print_board(state)
            print("Result:", check_winner(state))
            break
        ai_action = agent.choose_action(state)
        state = make_move(state, ai_action, 'X')
        if check_winner(state):
            print_board(state)
            print("Result:", check_winner(state))
            break

def print_board(state):
    for i in range(0, 9, 3):
        row = list(state[i:i+3])
        print("|".join(row))
    print()

if __name__ == "__main__":
    agent = QLearningAgent()
    # Train or load
    if os.path.exists("q_table.pkl"):
        with open("q_table.pkl", "rb") as f:
            agent.q = pickle.load(f)
    else:
        print("Training agent... (may take ~30 seconds)")
        train(agent, episodes=20000)
        with open("q_table.pkl", "wb") as f:
            pickle.dump(agent.q, f)
    print("Play against trained agent (you are O).")
    play_vs_agent(agent)
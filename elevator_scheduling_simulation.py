import numpy as np
import random
import matplotlib.pyplot as plt

# === ENVIRONMENT SETUP ===
floors = 5
directions = ["up", "down", "idle"]
direction_map = {"up": 0, "down": 1, "idle": 2}
actions = ["up", "down", "idle"]

# === Q-LEARNING PARAMETERS ===
q_table = {}
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05
episodes = 500
max_steps = 200

# === STATE MANAGEMENT ===
def get_state(current_floor, direction, requests):
    request_pattern = tuple(requests)
    return (current_floor, direction_map[direction], request_pattern)

def initialize_q_table():
    for floor in range(floors):
        for direction in range(3):
            for r1 in [0, 1]:
                for r2 in [0, 1]:
                    for r3 in [0, 1]:
                        for r4 in [0, 1]:
                            for r5 in [0, 1]:
                                state = (floor, direction, (r1, r2, r3, r4, r5))
                                q_table[state] = [0, 0, 0]  # 3 actions

# === SIMULATION ===
def generate_requests(prob=0.1):
    return [1 if random.random() < prob else 0 for _ in range(floors)]

def step(state, action_idx):
    current_floor, direction_idx, requests = state
    direction = directions[direction_idx]
    requests = list(requests)
    reward = -1  # base penalty per step

    action = actions[action_idx]
    old_floor = current_floor  # store for progress check
    valid_move = True

    # === Move elevator with bounds check ===
    if action == "up":
        if current_floor < floors - 1:
            current_floor += 1
            direction = "up"
        else:
            valid_move = False
    elif action == "down":
        if current_floor > 0:
            current_floor -= 1
            direction = "down"
        else:
            valid_move = False
    else:
        direction = "idle"

    # === Serve request only if valid move or idle ===
    if requests[current_floor] == 1 and (action == "idle" or valid_move):
        requests[current_floor] = 0
        reward += 10

    # === Progress-based bonus ===
    requested_floors = [i for i, r in enumerate(requests) if r == 1]
    if requested_floors:
        closest_request = min(requested_floors, key=lambda x: abs(x - old_floor))
        new_floor = current_floor
        if abs(new_floor - closest_request) < abs(old_floor - closest_request):
            reward += 1  # moved closer to request
        elif abs(new_floor - closest_request) > abs(old_floor - closest_request):
            reward -= 1  # moved farther (optional)

    # === No-request idle fix ===
    if sum(requests) == 0 and action == "idle":
        reward = 0  # no penalty if idle when no requests

    # === New requests appear ===
    new_requests = generate_requests(prob=0.05)
    requests = [max(r, new) for r, new in zip(requests, new_requests)]

    # === Re-bin and return next state ===
    next_state = get_state(current_floor, direction, requests)
    return next_state, reward

# === TRAINING ===
initialize_q_table()
episode_rewards = []

for ep in range(episodes):
    current_floor = random.randint(0, floors - 1)
    direction = "idle"
    requests = generate_requests(prob=0.3)
    state = get_state(current_floor, direction, requests)

    total_reward = 0

    for step_i in range(max_steps):
        if random.random() < epsilon:
            action_idx = random.randint(0, 2)
        else:
            action_idx = np.argmax(q_table[state])

        next_state, reward = step(state, action_idx)
        old_value = q_table[state][action_idx]
        next_max = max(q_table[next_state])

        q_table[state][action_idx] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

# === PLOT TRAINING REWARDS ===
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Elevator Controller Training")
plt.grid(True)
plt.tight_layout()
plt.show()


# === CONSOLE SIMULATION USING TRAINED Q-TABLE ===
def simulate_trained_agent(steps=20):
    current_floor = 0
    direction = "idle"
    requests = [0, 1, 0, 1, 0]  # Manual test setup: people waiting at floor 1 and 3
    state = get_state(current_floor, direction, requests)

    print("\n--- Elevator Simulation (Trained Agent) ---\n")
    for t in range(steps):
        action_idx = np.argmax(q_table[state])
        action = actions[action_idx]
        next_state, reward = step(state, action_idx)

        print(f"Step {t+1}")
        print(f"  Current Floor: {state[0]}")
        print(f"  Direction: {directions[state[1]]}")
        print(f"  Requests: {list(state[2])}")
        print(f"  Action Taken: {action}")
        print(f"  Reward: {reward}")
        print("-" * 35)

        state = next_state

simulate_trained_agent()

def evaluate_action_accuracy(steps=200):
    current_floor = 0
    direction = "idle"
    requests = generate_requests(prob=0.3)  # Initial requests
    state = get_state(current_floor, direction, requests)

    correct_actions = 0
    total_actions = 0
    total_reward = 0
    requests_served = 0

    for _ in range(steps):
        action_idx = np.argmax(q_table[state])
        action = actions[action_idx]
        old_floor = state[0]
        old_requests = list(state[2])

        # Save current state to determine optimal action
        static_requests = old_requests.copy()
        has_requests = sum(static_requests) > 0

        # Step to next state
        new_state, reward = step(state, action_idx)
        new_floor = new_state[0]
        new_requests = list(new_state[2])
        total_reward += reward
        total_actions += 1

        # Determine optimal action manually
        if has_requests:
            closest = min([i for i, r in enumerate(static_requests) if r == 1],
                          key=lambda x: abs(x - old_floor))
            if closest > old_floor:
                expected_action = "up"
            elif closest < old_floor:
                expected_action = "down"
            else:
                expected_action = "idle"
        else:
            expected_action = "idle"

        if action == expected_action:
            correct_actions += 1

        # Requests served
        served = [
            i for i in range(floors)
            if static_requests[i] == 1 and new_requests[i] == 0
        ]
        requests_served += len(served)

        state = new_state

    action_accuracy = correct_actions / total_actions * 100
    avg_reward = total_reward / total_actions
    serve_rate = requests_served / total_actions * 100

    print("\n--- Action Accuracy Metrics ---")
    print(f"Total Steps: {total_actions}")
    print(f"Total Reward: {total_reward}")
    print(f"Average Reward/Step: {avg_reward:.2f}")
    print(f"Correct Actions Taken: {correct_actions} ({action_accuracy:.1f}%)")
    print(f"Requests Served: {requests_served} ({serve_rate:.1f}% of steps)")

evaluate_action_accuracy()


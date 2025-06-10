import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product

# === CONSTANTS ===
floors = 5
actions = ["up", "down", "idle"]
episodes = 2000
max_steps = 200
learning_rate = 0.3
discount_rate = 0.95
epsilon = 1.0
epsilon_decay = 0.990
min_epsilon = 0.05

# === Q-TABLE INITIALIZATION ===
def initialize_q_table():
    """Initializes a simplified Q-table with state = (floor, request pattern)"""
    q_table = {}
    for floor in range(floors):
        for request_pattern in product([0, 1], repeat=floors):
            q_table[(floor, request_pattern)] = [0, 0, 0]  # actions: up, down, idle
    return q_table

q_table = initialize_q_table()

# === HELPERS ===
def get_state(current_floor, requests):
    return (current_floor, tuple(requests))

def generate_requests(prob=0.05):
    return [1 if random.random() < prob else 0 for _ in range(floors)]

# === STEP FUNCTION ===
def step(state, action_idx):
    current_floor, requests = state
    requests = list(requests)
    reward = -1
    action = actions[action_idx]
    old_floor = current_floor
    valid_move = True

    # Movement
    if action == "up":
        if current_floor < floors - 1:
            current_floor += 1
        else:
            valid_move = False
    elif action == "down":
        if current_floor > 0:
            current_floor -= 1
        else:
            valid_move = False

    # Serve request
    # if requests[current_floor] == 1 and (action == "idle" or valid_move):
    #     requests[current_floor] = 0
    #     reward += 10
    if sum(requests) == 0:
        if action == "idle":
            reward = 15  # small positive reward for saving energy
        else:
            reward = -5  # discourage unnecessary movement

    if sum(requests) == 5:
        if action == "idle":
            reward = 15  # small positive reward for saving energy
        else:
            reward = -5  # discourage unnecessary movement


    if requests[current_floor] == 1:
        if action == "idle":
            requests[current_floor] = 0
            reward += 20  # larger reward for idling to serve
        if valid_move:
            requests[current_floor] = 0
            reward += 5   # lower reward for serving after movin

    if requests[old_floor] == 1 and action != "idle":
        reward -= 5  # penalty for ignoring request at current floor


    # Directional progress reward
    requested_floors = [i for i, r in enumerate(requests) if r == 1]
    if requested_floors:
        min_dist = min(abs(i - old_floor) for i in requested_floors)
        closest_requests = [i for i in requested_floors if abs(i - old_floor) == min_dist]

        if any(abs(current_floor - i) < abs(old_floor - i) for i in closest_requests):
            reward += 1
        elif all(abs(current_floor - i) > abs(old_floor - i) for i in closest_requests):
            reward -= 1


    # New requests appear
    new_requests = generate_requests(prob=0.05)
    requests = [max(r, new) for r, new in zip(requests, new_requests)]

    next_state = get_state(current_floor, requests)
    return next_state, reward

# TRAINING LOOP
rewards_per_episode = []

for ep in range(episodes):
    current_floor = random.randint(0, floors - 1)
    requests = generate_requests(prob=0.3)
    state = get_state(current_floor, requests)

    total_reward = 0

    for step_i in range(max_steps):
        if random.random() < epsilon:
            action_idx = random.randint(0, 2)
        else:
            action_idx = np.argmax(q_table[state])

        next_state, reward = step(state, action_idx)
        old_value = q_table[state][action_idx]
        next_max = max(q_table[next_state])
        q_table[state][action_idx] = old_value + learning_rate * (reward + discount_rate * next_max - old_value)
        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

# === VISUALIZATION ===
window = 30  # or any size you prefer

# Simple moving average (causal: includes current and previous)
moving_avg = []
for i in range(len(rewards_per_episode)):
    start = max(0, i - window + 1)
    avg = np.mean(rewards_per_episode[start:i+1])
    moving_avg.append(avg)

# Plot
plt.plot(rewards_per_episode, label="Episode Reward", color='b', alpha=0.4)
plt.plot(moving_avg, label=f"Moving Avg ({window})", color='red', linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward Over Episodes (Simple Moving Average)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Updated simulate_trained_agent and evaluate_action_accuracy for simplified Q-table

def simulate_trained_agent(steps=20):
    current_floor = 0
    requests = [0, 1, 0, 1, 0]  # Manual test setup
    state = get_state(current_floor, requests)

    print("\n--- Elevator Simulation (Trained Agent) ---\n")
    for t in range(steps):
        action_idx = np.argmax(q_table[state])
        action = actions[action_idx]
        next_state, reward = step(state, action_idx)

        print(f"Step {t+1}")
        print(f"  Current Floor: {state[0]}")
        print(f"  Requests: {list(state[1])}")
        print(f"  Action Taken: {action}")
        print(f"  Reward: {reward}")
        print("-" * 35)

        state = next_state

def evaluate_action_accuracy(return_only=False, steps=200):
    current_floor = 0
    requests = generate_requests(prob=0.3)
    state = get_state(current_floor, requests)

    correct_actions = 0
    total_actions = 0
    total_reward = 0
    requests_served = 0

    for _ in range(steps):
        action_idx = np.argmax(q_table[state])
        action = actions[action_idx]
        old_floor = state[0]
        old_requests = list(state[1])
        static_requests = old_requests.copy()
        has_requests = sum(static_requests) > 0

        new_state, reward = step(state, action_idx)
        new_floor = new_state[0]
        new_requests = list(new_state[1])
        total_reward += reward
        total_actions += 1

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

        served = [
            i for i in range(floors)
            if static_requests[i] == 1 and new_requests[i] == 0
        ]
        requests_served += len(served)

        state = new_state

    action_accuracy = correct_actions / total_actions * 100
    if return_only:
        return action_accuracy

    avg_reward = total_reward / total_actions
    serve_rate = requests_served / total_actions * 100

    print("\n--- Action Accuracy Metrics ---")
    print(f"Total Steps: {total_actions}")
    print(f"Total Reward: {total_reward}")
    print(f"Average Reward/Step: {avg_reward:.2f}")
    print(f"Correct Actions Taken: {correct_actions} ({action_accuracy:.1f}%)")
    print(f"Requests Served: {requests_served} ({serve_rate:.1f}% of steps)")

# Run evaluation 10 times and compute average accuracy
accuracies = [evaluate_action_accuracy(return_only=True) for _ in range(100)]
average_accuracy = sum(accuracies) / len(accuracies)

print(f"\n📊 Average Action Accuracy over 100 runs: {average_accuracy:.2f}%")

stddev = (sum((x - average_accuracy) ** 2 for x in accuracies) / len(accuracies))**0.5
print(f"StdDev: {stddev:.2f}%")


# simulate_trained_agent()
# evaluate_action_accuracy()

def evaluate_fixed_cases():
    fixed_test_cases = [
        (0, [0, 0, 0, 0, 1]),  # Request at top floor
        (4, [1, 0, 0, 0, 0]),  # Request at bottom floor
        (2, [0, 1, 0, 1, 0]),  # Requests on both sides
        (1, [0, 0, 0, 0, 0]),  # No requests
        (3, [0, 0, 1, 0, 0]),  # Request below
        (1, [0, 0, 1, 0, 0]),  # Request above
        (0, [1, 1, 1, 1, 1]),  # All floors requested
    ]

    total_cases = len(fixed_test_cases)
    correct_actions = 0
    requests_served = 0

    for current_floor, request_list in fixed_test_cases:
        state = get_state(current_floor, request_list)
        action_idx = np.argmax(q_table[state])
        action = actions[action_idx]

        # Determine valid expected actions
        expected_actions = set()
        if sum(request_list) == 0:
            expected_actions.add("idle")
        else:
            distances = [abs(i - current_floor) if r == 1 else float('inf')
                         for i, r in enumerate(request_list)]
            min_distance = min(distances)
            closest_floors = [i for i, d in enumerate(distances) if d == min_distance]
            for f in closest_floors:
                if f > current_floor:
                    expected_actions.add("up")
                elif f < current_floor:
                    expected_actions.add("down")
                else:
                    expected_actions.add("idle")


        is_correct = action in expected_actions
        if is_correct:
            correct_actions += 1
        if request_list[current_floor] == 1 and action == "idle":
            requests_served += 1

        status = "✅ Correct" if is_correct else "❌ Incorrect"
        print(f"Test Case - Floor: {current_floor}, Requests: {request_list}")
        print(f"  Action Taken: {action}")
        print(f"  Expected Actions: {list(expected_actions)}")
        print(f"  {status}")
        print("-" * 40)

    accuracy = correct_actions / total_cases * 100
    serve_rate = requests_served / total_cases * 100

    print(f"\n📊 Fixed Case Evaluation (Improved):")
    print(f"Correct Actions: {correct_actions}/{total_cases} ({accuracy:.1f}%)")
    # print(f"Immediate Requests Served by Idling: {requests_served}/{total_cases} ({serve_rate:.1f}%)")

evaluate_fixed_cases()

def print_q_table_sample(limit=20):
    """
    Prints a sample of the Q-table in a readable format.
    Each row shows the state (floor + request pattern) and corresponding Q-values.
    `limit` specifies how many rows to print (for readability).
    """
    print(f"{'State':<10} {'up':>6} {'down':>6} {'idle':>6}")
    print("-" * 30)

    count = 0
    for (floor, reqs), q_values in q_table.items():
        if count >= limit:
            break
        req_str = ''.join(str(r) for r in reqs)
        state_str = f"F{floor}-{req_str}"
        print(f"{state_str:<10} {q_values[0]:>6.2f} {q_values[1]:>6.2f} {q_values[2]:>6.2f}")
        count += 1

# print_q_table_sample(160)

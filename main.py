import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

""" 1 """
desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
rows = len(desc)
cols = len(desc[0])
env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True)
env = env.unwrapped


""" 2.1 """
gamma = 0.5  # коефіцієнт знижки
theta = 1e-8  # поріг для зупинки
V = np.zeros(env.observation_space.n)  # функція вартості стану
policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n  # рівноймовірна стратегія


def policy_evaluation(policy, env, gamma, theta):
    V = np.zeros(env.observation_space.n)  # початкові значення функції вартості стану
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        if delta < theta:
            break
    return V


# Обчислюємо функцію вартості стану
V = policy_evaluation(policy, env, gamma, theta)

# Виведення теплової карти
V_grid = V.reshape((rows, cols))  # перетворюємо у матрицю 4x4
plt.figure(figsize=(6, 6))
sns.heatmap(V_grid, annot=True, cmap="coolwarm", cbar=True)
plt.title("Функція ціни стану V для рівноймовірної стратегії")
plt.show()


""" 2.2 """
def policy_evaluation_LA(policy, env, gamma, theta):
    n_states = env.observation_space.n
    A = np.zeros((n_states, n_states))
    b = np.zeros(n_states)
    for s in range(n_states):
        A[s, s] = 1
        for a, action_prob in enumerate(policy[s]):
            for prob, next_state, reward, done in env.P[s][a]:
                A[s][next_state] -= action_prob * prob * gamma
                b[s] += action_prob * prob * reward
    V = np.linalg.solve(A, b)
    return V


# Розв'язуємо систему лінійних рівнянь
V = policy_evaluation_LA(policy, env, gamma, theta)

# Виведення теплової карти
V_grid = V.reshape((rows, cols))  # перетворюємо у матрицю 4x4
plt.figure(figsize=(6, 6))
sns.heatmap(V_grid, annot=True, cmap="coolwarm", cbar=True)
plt.title("Функція ціни стану V для рівноймовірної стратегії")
plt.show()


""" 3 """
def compute_q_value_function(V, env, gamma):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[s][a]:
                Q[s, a] += prob * (reward + gamma * V[next_state])
    return Q


# Обчислюємо функцію ціни дії-стану q(s, a)
Q = compute_q_value_function(V, env, gamma)
Q_equal = np.copy(Q)

# Виведення теплової карти
Q_grid = Q_equal.reshape((env.observation_space.n, env.action_space.n))  # перетворюємо у матрицю 16x4
plt.figure(figsize=(6, 12))
sns.heatmap(Q_grid, annot=True, cmap="coolwarm", cbar=True)
plt.title("Функція ціни дії-стану Q для рівноймовірної стратегії")
plt.show()

""" 4 """
def equiprobable(env):
    action = np.random.choice(env.action_space.n)
    return action


""" 5 """
def get_episode(env):
    episode = []
    state, info = env.reset()  # Початковий стан

    while True:
        action = equiprobable(env)  # Вибір випадкової дії
        next_state, reward, terminated, truncated, info = env.step(action)  # Виконання дії

        episode.append((state, action, reward, next_state, terminated, truncated))

        state = next_state
        if terminated or truncated:
            break

    return episode


""" 6 """
n_episodes = 75
rewards_equal = []
lengths_equal = []

for _ in range(n_episodes):
    episode = get_episode(env)

    total_reward = sum([step[2] for step in episode])
    rewards_equal.append(total_reward)

    episode_length = len(episode)
    lengths_equal.append(episode_length)

# Графік винагород
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_equal)
plt.title("Винагорода за епізод")
plt.xlabel("Номер епізоду")
plt.ylabel("Загальна винагорода")

# Графік тривалості епізодів
plt.subplot(1, 2, 2)
plt.plot(lengths_equal)
plt.title("Тривалість епізоду")
plt.xlabel("Номер епізоду")
plt.ylabel("Кількість кроків")

plt.show()

""" 7 """


def policy_improvement(env, policy, V, gamma, theta):
    policy_stable = True
    new_policy = np.zeros([env.observation_space.n, env.action_space.n])

    for s in range(env.observation_space.n):
        old_action = np.argmax(policy[s])
        action_values = np.zeros(env.action_space.n)

        # Обчислюємо значення для кожної дії
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])

        # Вибираємо всі дії, які мають максимальне значення
        best_actions = np.flatnonzero(action_values == np.max(action_values))

        # Рівноймовірно розподіляємо ймовірності між найкращими діями
        new_policy[s, best_actions] = 1 / len(best_actions)

        if old_action != np.argmax(new_policy[s]):
            policy_stable = False

    return new_policy, policy_stable


def policy_iteration(env, policy, gamma, theta):
    V = np.zeros(env.observation_space.n)
    while True:
        V = policy_evaluation(policy, env, gamma, theta)
        policy, policy_stable = policy_improvement(env, policy, V, gamma, theta)

        if policy_stable:
            break

    return policy, V


""" 8 """
new_policy, new_V = policy_iteration(env, policy, gamma, theta)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

V_grid_1 = V.reshape((rows, cols))
sns.heatmap(V_grid_1, annot=True, cmap="coolwarm", cbar=True, ax=axs[0])
axs[0].set_title("Функція ціни стану V для рівноймовірної стратегії")

V_grid_2 = new_V.reshape((rows, cols))
sns.heatmap(V_grid_2, annot=True, cmap="coolwarm", cbar=True, ax=axs[1])
axs[1].set_title("Оптимальна функція ціни стану V*")

# Виведення графіків
plt.tight_layout()
plt.show()

print("Оптимальна стратегія π*:")
print(new_policy)

""" 9.1 """


def get_optimal_action(policy, state):
    action_probabilities = policy[state]
    optimal_action = action_probabilities.argmax()
    return optimal_action


def get_optimal_episode(env, policy):
    episode = []
    state, info = env.reset()  # Початковий стан

    while True:
        action = get_optimal_action(policy, state)  # Вибір оптимальної дії
        next_state, reward, terminated, truncated, info = env.step(action)  # Виконання дії

        episode.append((state, action, reward, next_state, terminated, truncated))

        state = next_state
        if terminated or truncated:
            break

    return episode


""" 9.2 """
n_episodes = 75
rewards = []
lengths = []

for _ in range(n_episodes):
    # епізоди з оптимальною стратегією
    episode = get_optimal_episode(env, new_policy)

    total_reward = sum([step[2] for step in episode])
    rewards.append(total_reward)

    episode_length = len(episode)
    lengths.append(episode_length)

# Графік винагород
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards, color='blue', label='стратегія π*')
plt.plot(rewards_equal, color='orange', label='стратегія π1')
plt.title("Винагорода за епізод")
plt.xlabel("Номер епізоду")
plt.ylabel("Загальна винагорода")
plt.legend()

# Графік тривалості епізодів
plt.subplot(1, 2, 2)
plt.plot(lengths, color='blue', label='стратегія π*')
plt.plot(lengths_equal, color='orange', label='стратегія π1')
plt.title("Тривалість епізоду")
plt.xlabel("Номер епізоду")
plt.ylabel("Кількість кроків")
plt.legend()

plt.show()

""" 10 """
# Обчислюємо функцію ціни дії-стану q(s, a)
new_Q = compute_q_value_function(new_V, env, gamma)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

Q_grid_1 = Q_equal.reshape((env.observation_space.n, env.action_space.n))
sns.heatmap(Q_grid_1, annot=True, cmap="coolwarm", cbar=True, ax=axs[0])
axs[0].set_title("Функція ціни дії_стану Q для рівноймовірної стратегії")

Q_grid_2 = new_Q.reshape((env.observation_space.n, env.action_space.n))
sns.heatmap(Q_grid_2, annot=True, cmap="coolwarm", cbar=True, ax=axs[1])
axs[1].set_title("Оптимальна функція ціни дії_стану Q*")

# Виведення графіків
plt.tight_layout()  # Автоматичне налаштування інтервалів
plt.show()

""" 11 """


def eps_greedy_policy(Q_values, epsilon):
    """
    Вибирає дію за допомогою eps-жадібної стратегії.

    Аргументи:
    Q_values -- масив значень q(s, a) для всіх дій у стані s.
    epsilon -- параметр ε, що визначає ймовірність випадкового вибору дії.

    Повертає:
    Вибраний номер дії.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q_values))
    else:
        return np.argmax(Q_values)


""" 12 """


def sarsa_max(env, Q, num_episodes, alpha, gamma, epsilon):
    """
    Реалізація методу SARSAmax для пошуку оптимальної функції ціни дії-стану Q_*(s, a).

    Аргументи:
    env -- середовище Gymnasium.
    num_episodes -- кількість епізодів для навчання.
    alpha -- швидкість навчання.
    gamma -- коефіцієнт дисконтування.
    epsilon -- параметр ε для eps-жадібної стратегії.

    Повертає:
    Масив значень функції Q(s, a).
    """
    Q_values = np.copy(Q)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # Обираємо дію за допомогою eps-жадібної стратегії
            action = eps_greedy_policy(Q_values[state], epsilon)

            # Виконуємо дію та отримуємо новий стан, винагороду та статус завершення
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Оновлюємо значення функції q(s, a) за методом SARSAmax
            next_action = eps_greedy_policy(Q_values[next_state], epsilon)
            td_target = reward + gamma * np.max(Q_values[next_state])
            td_error = td_target - Q_values[state][action]
            Q_values[state][action] += alpha * td_error

            state = next_state

    return Q_values


""" 13 """
alpha = 0.1  # Швидкість навчання
num_episodes = 5000  # Кількість епізодів для навчання

# Оцінка для epsilon = 0.1
epsilon = 0.1
Q_eps_01 = sarsa_max(env, Q, num_episodes, alpha, gamma, epsilon)

# Оцінка для epsilon = 0.5
epsilon = 0.5
Q_eps_05 = sarsa_max(env, Q, num_episodes, alpha, gamma, epsilon)

# Оцінка для epsilon = 0.9
epsilon = 0.9
Q_eps_09 = sarsa_max(env, Q, num_episodes, alpha, gamma, epsilon)

# Теплові карти для порівняння q(s, a) при різних epsilon
plt.figure(figsize=(30, 10))

plt.subplot(1, 5, 1)
sns.heatmap(Q_eps_01, annot=True, cmap="coolwarm")
plt.title('Q для epsilon = 0.1')

plt.subplot(1, 5, 2)
sns.heatmap(Q_eps_05, annot=True, cmap="coolwarm")
plt.title('Q для epsilon = 0.5')

plt.subplot(1, 5, 3)
sns.heatmap(Q_eps_09, annot=True, cmap="coolwarm")
plt.title('Q для epsilon = 0.9')

plt.subplot(1, 5, 4)
sns.heatmap(Q, annot=True, cmap="coolwarm")
plt.title('Q для рівномірної стратегії')

plt.subplot(1, 5, 5)
sns.heatmap(new_Q, annot=True, cmap="coolwarm")
plt.title('Q*')

plt.show()

""" 14 """


def create_eps_greedy_policy(Q, epsilon, env):
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        Q_values = Q[s]
        if np.random.rand() < epsilon:
            best_actions = [np.random.choice(len(Q_values))]
        else:
            best_actions = np.flatnonzero(Q_values == np.max(Q_values))

        policy[s, best_actions] = 1 / len(best_actions)

    return policy


# Створюємо ε-жадібну стратегію π2 для epsilon = 0.1 на основі q*(s, a)
epsilon = 0.1
policy_sarsa_max = create_eps_greedy_policy(Q_eps_01, epsilon, env)

print("\nОцінена оптимальну стратегія (ліво = 0, низ = 1, праворуч = 2, вверх = 3):")
print(policy_sarsa_max)

""" 15 """
n_episodes = 75
rewards_sarsa_max = []
lengths_sarsa_max = []

for _ in range(n_episodes):
    episode = get_optimal_episode(env, policy_sarsa_max)

    total_reward = sum([step[2] for step in episode])
    rewards_sarsa_max.append(total_reward)

    episode_length = len(episode)
    lengths_sarsa_max.append(episode_length)

# Графік винагород
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_sarsa_max, color='red', label='стратегія sarsa_max')
plt.plot(rewards, color='blue', label='стратегія π*')
plt.plot(rewards_equal, color='orange', label='стратегія π1')
plt.title("Винагорода за епізод")
plt.xlabel("Номер епізоду")
plt.ylabel("Загальна винагорода")
plt.legend()

# Графік тривалості епізодів
plt.subplot(1, 2, 2)
plt.plot(lengths_sarsa_max, color='red', label='стратегія sarsa_max')
plt.plot(lengths, color='blue', label='стратегія π*')
plt.plot(lengths_equal, color='orange', label='стратегія π1')
plt.title("Тривалість епізоду")
plt.xlabel("Номер епізоду")
plt.ylabel("Кількість кроків")
plt.legend()

plt.show()

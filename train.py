import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO


class Lattice(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=10, max_steps=200):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Action Space: Up, Right, Down, Left
        self.action_space = spaces.Discrete(4)

        # State Space：[agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(4,), dtype=np.int32
        )

        # for Plotting
        self.fig, self.ax = None, None

        # Record the last action for comparing
        self.last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.steps = 0
        self.goal = np.random.randint(0, self.grid_size, size=2)
        self.last_action = None
        return np.array([*self.agent_pos, *self.goal], dtype=np.int32), {}

    def step(self, action):
        self.steps += 1

        # Actions of Agent
        if action == 0 and self.agent_pos[1] < self.grid_size - 1:  # up
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:  # right
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # down
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1

        reward = -0.011

        terminated = False
        if np.array_equal(self.agent_pos, self.goal):
            reward = 1.0
            terminated = True

        if self.last_action is not None and action == self.last_action:
            reward += 0.01

        self.last_action = action

        truncated = self.steps >= self.max_steps

        obs = np.array([*self.agent_pos, *self.goal], dtype=np.int32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True)

        # Destination in Red
        self.ax.add_patch(
            patches.Rectangle(
                (self.goal[0] - 0.4, self.goal[1] - 0.4),
                0.8, 0.8, linewidth=1, edgecolor="red", facecolor="red"
            )
        )
        # Agent in Blue
        self.ax.add_patch(
            patches.Rectangle(
                (self.agent_pos[0] - 0.4, self.agent_pos[1] - 0.4),
                0.8, 0.8, linewidth=1, edgecolor="blue", facecolor="blue"
            )
        )

        plt.pause(0.2)


# =====================
# Training
# =====================
def train_model():
    env = Lattice(grid_size=10)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save("ppo_lattice_v2")
    env.close()
    print("Over, the model is saved ppo_lattice_v2.zip")


# =====================
# Testing
# =====================
def test_model(episodes=5):
    env = Lattice(grid_size=10)
    model = PPO.load("ppo_lattice_v2.zip")

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
        if terminated == True:
            print(f"Episode {ep + 1}: Destination Founded!")
        else:
            print(f"Episode {ep + 1}: Destination not Founded!")


    plt.show()
    env.close()


if __name__ == "__main__":
    train_model()

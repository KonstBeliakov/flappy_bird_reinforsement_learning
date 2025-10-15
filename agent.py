import flappy_bird_gymnasium
import gymnasium
import torch.cuda
import yaml

from dqn import DQN
from experience_replay import ReplayMemory
import itertools

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']

    def run(self, is_training=True, render=False):
        #env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_episode = []

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:  # Если идет обучение, то сохраняем данные, чтобы потом учиться
            memory = ReplayMemory(self.replay_memory_size)

        for episode in itertools.count():  # iterating indefinetely, because we will stop manualy (играем бесконечное количество игр)
            state, _ = env.reset()
            terminated = False

            episode_reward = 0.0

            while True:  # Одна игра
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample() # выбор действия агента (пока что случайное)

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)

                episode_reward += reward

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                state = new_state

                # Checking if the player is still alive
                if terminated:
                    break

            reward_per_episode.append(episode_reward)

        env.close()

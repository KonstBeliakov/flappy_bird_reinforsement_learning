import argparse
import random
from datetime import datetime

import flappy_bird_gymnasium
import gymnasium
import matplotlib.pyplot as plt
import torch.cuda
from torch import nn
import yaml
import os
import matplotlib
import numpy as np

from dqn import DQN
from experience_replay import ReplayMemory
import itertools

device = 'cpu'  #'cuda' if torch.cuda.is_available() else 'cpu'

DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")


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
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params', {})  # Get optional environment-specific parameters
        self.network_sync_rate = hyperparameters['network_sync_rate']

        self.loss_fn = nn.MSELoss()  # NN Loss function. MSE=Mean Squared Error can be swapped by something else
        self.optimizer = None

        # Path to Run info
        name = 'cartpole1'
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{name}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{name}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{name}.png')

    def run(self, is_training=True, render=False):
        #env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w', encoding='utf-8') as file:
                file.write(log_message + '\n')

        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:  # Если идет обучение, то сохраняем данные, чтобы потом учиться
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Track number of steps taken. Used for syncing policy => target network
            step_count = 0

            best_reward = -10**9

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        for episode in itertools.count():  # iterating indefinetely, because we will stop manualy (играем бесконечное количество игр)
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False

            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:  # Одна игра
                # Next action:
                # (feed the observation to your agent here)

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # tensor([1, 2, 3...]) -> tensor([[1, 2, 3...]]) adding another dimension
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                    step_count += 1

                state = new_state

            reward_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f'{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward: 0.1f}'
                    print(log_message)
                    with open(self.LOG_FILE, 'a', encoding='utf-8') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)  # saving the model with the best reward
                    best_reward = episode_reward
                self.save_graph(reward_per_episode, epsilon_history)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        env.close()

    def save_graph(self, reward_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(reward_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(reward_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)

        plt.ylabel("Mean Rewards")
        plt.plot(mean_rewards)
        plt.subplot(122)

        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (extended returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        #current_q = policy_dqn(states)
        #loss = self.loss_fn(current_q, target_q)

        q_pred_all = policy_dqn(states)  # shape [B, A]
        q_pred = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # shape [B]
        loss = self.loss_fn(q_pred, target_q)  # both [B]

        self.optimizer.zero_grad()  # Clearing gradients
        loss.backward()             # Compute gradients (backpropagation)
        self.optimizer.step()       # Update network parameters i.e. weights and biases


if __name__ == '__main__':
    #agent = Agent("cartpole1")
    #agent.run(is_training=True, render=True)
    #parser = argparse.ArgumentParser(description='Train or test model.')
    #parser.add_argument('hyperparameters', help='')
    #parser.add_argument('--train', help='Training mode', action='store_true')
    #args = parser.parse_args()

    dql = Agent('cartpole1') #hyperparameter_set=args.hyperparameters)
    dql.run(is_training=True)
    #if args.train:
    #    dql.run(is_training=True)
    #else:
    #    dql.run(is_training=False, render=True)

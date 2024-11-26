import environment
import nnAgent as nn
import dataLoader as dl
import numpy as np
import matplotlib.pyplot as plt
from nnAgent import DQNAgent
data = dl.returns
env = environment.Env(data)
state_dim = env.state_space_dim * 10  # Sabit boyut
action_dim = len(env.action_space)
print("koyun")
agent = nn.DQNAgent(state_dim, action_dim)
print("koyun") 
num_episodes = 500
episode_rewards = []
episode_losses = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    episode_loss = 0
    loss_count = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        
        agent.buffer.store(state, action, reward, next_state, done)
        loss = agent.train()

        if loss is not None:
            episode_loss += loss
            loss_count += 1

        state = next_state
        total_reward += reward
        """
        if agent.step % 500 == 0:
            agent.update_target_network()
        """
        if done:
            break

    episode_rewards.append(total_reward)
    if loss_count > 0:
        episode_losses.append(episode_loss / loss_count)
    else:
        episode_losses.append(None)
    if episode % 10 == 0:
        print(f"Average loss after 10 episodes: {np.mean(episode_losses[-10:])}")
        print(f"Episode {episode}, Total Reward: {total_reward}")


# Eğitim tamamlandığında sonuçları yazdır
print("Training completed.")
print(f"Total Episodes: {num_episodes}")
print(f"Average Reward: {sum(episode_rewards) / num_episodes}")
print(f"Max Reward: {max(episode_rewards)}")
print(f"Min Reward: {min(episode_rewards)}")
print(f"Average Loss: {sum(filter(None, episode_losses)) / len(filter(None, episode_losses))}")



# Define rewards_per_episode, losses, and all_actions
rewards_per_episode = episode_rewards
losses = [loss for loss in episode_losses if loss is not None]
all_actions = [agent.select_action(env.reset()) for _ in range(num_episodes)]

plt.plot(rewards_per_episode)
plt.title("Episode Rewards Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
    
plt.plot(losses)
plt.title("Loss Over Time")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()

actions = np.array(all_actions)  # Aksiyonların kaydını tutun
plt.hist(actions, bins=[0, 1, 2, 3], align='left')
plt.title("Action Distribution")
plt.xlabel("Action")
plt.ylabel("Frequency")
plt.show()




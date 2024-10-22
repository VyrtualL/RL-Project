from game_environments.pong_env import PongEnv
from learning_agents.deep_qlearning import DeepQLearning
from learning_agents.testing_eps_greedy import TestingEpsGreedy

import joblib
import matplotlib.pyplot as plt
import time
import shutil

game_env = PongEnv(mode=0, difficulty=0)
actions = list(range(game_env.num_actions))

nb_iters = 10000000
rewards = []
mem = 10000
eps_decay = nb_iters // 10

#agent = DeepQLearning(actions, learning_rate=0.25e-4, epsilon_steps=eps_decay, memory=mem)
#i = 0

agent, i = joblib.load("rl_model.joblib")
agent.set_device()
nb_iters -= agent.step

while nb_iters > 0:
    agent.reset_context()
    game_env.train_agent(agent, max_iter=None)
    rewards.append(game_env.total_reward)
    print(f"Game {i + 1} played: reward is {rewards[-1]}")
    len_game = len(game_env.renders) - 1
    print(f"Game lasted {len_game} frames")
    nb_iters -= len_game
    print(f"{nb_iters} frames left in training")
    print(f"Epsilon is currently {agent.epsilon}")
    i += 1
    if i % 20 == 0:
        game_env.animate(save=True, path=f"animations/training/game_num_{i}_train.gif")
        agent.reset_context()
        #test_agent = TestingEpsGreedy(actions, agent)
        game_env.use_agent(agent)
        game_env.animate(save=True, path=f"animations/using/game_num_{i}_use.gif")
        print("Saving: do not interrupt...")
        start_time = time.time()
        joblib.dump((agent, i), "rl_model.joblib")
        end_time = time.time()
        print(f"Model saved in {end_time - start_time:.2f}s")
        print("Copying...")
        shutil.copyfile("rl_model.joblib", "tmp_rl_model.joblib")
        print("Copy over")


agent.reset_context()
agent.training_stop = True
game_env.use_agent(agent, max_iter=None)

print(game_env.total_reward)
game_env.animate(save=True, path="animations/final_game.gif")
joblib.dump(agent, "rl_model.joblib")

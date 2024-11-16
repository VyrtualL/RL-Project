from game_environments.pong_env import PongEnv
from learning_agents.ddqn import DDQN
from learning_agents.testing_eps_greedy import TestingEpsGreedy

import numpy as np
import joblib
import matplotlib.pyplot as plt

mem_size = 50000
tau = 0.005
lr = 1e-6

total_iter = 1#0000000
nb_iter = total_iter
i = 0

test_rate = 50

game_env = PongEnv(mode=0, difficulty=0)
actions = list(range(game_env.num_actions))

agent = DDQN(actions, learning_rate=lr, memory=mem_size, tau=tau)
joblib.dump(agent.model, "eval_model.joblib")

def evaluate(game_env, agent):
    print("Testing models")
    curr_best_model = joblib.load("eval_model.joblib")
    curr_best_agent = DDQN(actions)
    curr_best_agent.model = curr_best_model

    test_current_scores = []
    test_prev_best_scores = []

    for _ in range(10):
        game_env.reset()
        agent.reset_context()
        game_env.use_agent(agent)
        test_current_scores.append(game_env.total_reward)

        game_env.reset()
        curr_best_agent.reset_context()
        game_env.use_agent(curr_best_agent)
        test_prev_best_scores.append(game_env.total_reward)

    for _ in range(10):
        game_env.reset()
        agent.reset_context()
        test_agent = TestingEpsGreedy(actions, agent)
        game_env.train_agent(test_agent)
        test_current_scores.append(game_env.total_reward)

        game_env.reset()
        curr_best_agent.reset_context()
        test_agent = TestingEpsGreedy(actions, curr_best_agent)
        game_env.train_agent(test_agent)
        test_prev_best_scores.append(game_env.total_reward)
    print("Evaluation games over")

    avg_current_score = np.mean(test_current_scores)
    avg_prev_best = np.mean(test_prev_best_scores)
    return avg_current_score, avg_prev_best

scores, step_nb = [], []
best_score = -21.0

while nb_iter > 0:
    game_env.reset()
    agent.reset_context()

    print(f"Game {i + 1} ({nb_iter} frames left, epsilon is {agent.get_epsilon()}):")
    game_env.train_agent(agent, max_iter=None)
    scores.append(game_env.total_reward)
    step_nb.append(agent.step)
    nb_iter = total_iter - agent.step
    curr_score = game_env.total_reward

    if (i + 1) % 10 == 0:
        last_reward = game_env.total_reward
        
        game_env.reset()
        agent.reset_context()
        test_agent = TestingEpsGreedy(actions, agent)
        game_env.train_agent(test_agent)
        rd_reward = game_env.total_reward
        game_env.animate(save=True, path=f"animations/test_greedy/game_{i+1}.gif")

        game_env.reset()
        agent.reset_context()
        game_env.use_agent(agent)
        game_env.animate(save=True, path=f"animations/using/game_{i+1}.gif")
        curr_score = max(curr_score, game_env.total_reward)

    if (i + 1) % test_rate == 0 or curr_score > best_score:
        avg_current_score, avg_prev_best = evaluate(game_env, agent)
        best_score = max(avg_current_score, avg_prev_best)

        if avg_current_score > avg_prev_best:
            joblib.dump(agent.model, "eval_model.joblib")
    i += 1

avg_current_score, avg_prev_best = evaluate(game_env, agent)

if avg_current_score > avg_prev_best:
    joblib.dump(agent.model, "eval_model.joblib")

last_model = joblib.load("eval_model.joblib")
last_agent = DDQN(actions)
last_agent.model = last_model

game_env.reset()
last_agent.reset_context()
game_env.use_agent(last_agent, max_iter=None)

game_env.animate(save=True, path="animations/final_game.gif")

plt.plot(step_nb, scores)
plt.xlabel("Training steps")
plt.ylabel("Score")
plt.savefig("training_curve.png")

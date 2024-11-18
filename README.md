## RL-Project TP4

Deep Reinforcement Learning project on the Pong game developed by:
- Juliette Jacquot
- Matis Braun


# Deep Q Learning:
If you want to use the Deep Q Learning model, you have to launch the `main.py`, which will launch the training on 1,000,000 frames.
This model has the worst results.

# Double Q Learning:
If you want to use the Double Q Learning model, you have to launch the `main_ddqn.py`, which will launch the training on 2,000,000 frames.
This model has the best results.
The best model has been saved in the file `model_ddqn_best.joblib`

# Dueling Double Q Learning + Noisy Network
If you want to use the Dueling Double Q Learning + Noisy Network model, you have to launch the `upgrade/main_ddqn.py` which will launch the training on 3,000,000 frames.
This model theoretically has the best results.


# Animation
In the `animation` folder, we can find `best_final_game_ddqn.gif` which shows a game using the best model we had. The model wins with a 19-point lead.

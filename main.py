import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

import pandas as pd
import tkinter as tk
from tkinter import *


root = tk.Tk()
root.title = "Stock Trading"
root.geometry("720x576")
root.configure(bg='#97AAA8')

canvas = Canvas(root, height=400, width=700)
canvas.configure(bg='#ebf2f1')
canvas.pack()

# Vertical lines
for i in range(0,700,20):
    canvas.create_line(i,0,i,700,width=1, fill='#e2e1e1')

# Horizontal lines
for i in range(0,700,20):
    canvas.create_line(0,i,700,i,width=1, fill='#ddeff5')

data_file = pd.read_csv('./data/AAPL.csv')
data_file = data_file.sort_values('Date')

# multiprocess environment
env = DummyVecEnv([lambda: StockTradingEnv(data_file)])

model = PPO2(MlpPolicy, env, verbose=1)

file_learn = open("AI_LEARN.txt", "r+")
file_learn.truncate(0)
file_learn.write(str(model.learn(total_timesteps=1000, log_interval=100).__str__))
file_learn.write("\n" + str(env))
file_learn.close()


# model.save("ppo2_stocktrade")
# del model # remove to demonstrate saving and loading
# model = PPO2.load("ppo2_stocktrade")

stepSummary = []

# Enjoy trained agent
obs = env.reset()

for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print("REWARDS:", str(rewards), "\n")
    #stepSummary.append(env.render())
    print(env.render())


T = tk.Text(root, height=30, width=60)
T.pack()

fullSummary = ""
testSummary = ""

print("***SUMMARY***", stepSummary)


for i in range(len(stepSummary)):
    fullSummary += str(stepSummary[i])
    fullSummary += "\n\n"

T.insert(tk.END, fullSummary)
T.insert(tk.END, testSummary)
root.update()
tk.mainloop()
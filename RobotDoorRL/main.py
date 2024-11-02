import time
import os
import gym
import pybullet_envs
#import gnumpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter #for logging data
import robosuite as suite
# from robosuite_environment import RoboSuiteWrapper
from robosuite.wrappers import GymWrapper
from networks import ActorNetwork,CriticNetwork
from  buffer import ReplayBuffer
from td3_torch import Agent

if __name__ == '__main__':


    env_name = "Door" #load this enviroment from robosuite examples

    env = suite.make(
        env_name,  # Environment
        robots=["Panda"],  # Use two Panda robots # type of robot
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller #pass joint vel
        # controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,  # Enable rendering
        use_camera_obs=False,
        horizon=300, #time it takes the robot to figure out a solution DEPENDS ON PROBLEM
        #render_camera= "frontview", #"sideview",           # Camera view
        #has_offscreen_renderer=True,        # No offscreen rendering
        reward_shaping=True, #on false it only gives reward when the task is fulfilled maks learning harder
        control_freq=20,  # Control frequency
    )
    env = GymWrapper(env) #fits robosuite env into gym framework


actor_learning_rate=0.001
critic_learning_rate = 0.001
batch_size = 128
layer1_size=256
layer2_size=256

agent = Agent(actor_learning_rate=actor_learning_rate,critic_learning_rate=critic_learning_rate,tau=0.005,input_dims=env.observation_space.shape,
              env=env,n_actions=env.action_space.shape[0],layer1_size=layer1_size,layer2_size=layer2_size,batch_size=batch_size)

writer = SummaryWriter('logs')
n_games = 10000
best_score=0
episode_identifier=f"0 - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer_1_size={layer1_size} layer_2_size={layer2_size}"
#for figuring aout whatlearning worked

agent.load_models()

#training loop
for i in range(n_games):
    observation = env.reset() #start a new sim
    done = False
    score =0

    while not done:
        action = agent.choose_action(observation)

        next_observation, reward, done, info = env.step(action)

        score += reward

        agent.remember(observation,action,reward,next_observation,done)

        agent.learn()

        observation = next_observation

    writer.add_scalar(f"score - {episode_identifier}",score,global_step=i)

    if(i % 10):
        agent.save_models()

    print(f"episode: {i} score {score}")













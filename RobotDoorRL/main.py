import time
import os
import gym
import pybullet_envs
import gnumpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter #for logging data
import robosuite as suite
# from robosuite_environment import RoboSuiteWrapper
from robosuite.wrappers import GymWrapper
from networks import ActorNetwork,CriticNetwork
from  buffer import ReplayBuffer


if __name__ == '__main__':
    env_name = "DoorRobotRL"

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

    ###
    critic_n = CriticNetwork([8],8);
    actor_n = ActorNetwork([8],8)
    replay_buffer = ReplayBuffer(8,[8],8)
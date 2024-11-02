import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import  ReplayBuffer
from networks import ActorNetwork,CriticNetwork


class Agent:

    def __init__(self,actor_learning_rate,critic_learning_rate,input_dims,tau,env,gamma=0.99,
                 update_actor_interval=2,warmup=1000,n_actions=2,
                 max_size=10000000,layer1_size=256,layer2_size= 128,batch_size = 100,noise = 0.1):

        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low # from gym library
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                  fc2_dims=layer2_size, n_actions=n_actions,
                                  name='actor', learning_rate=actor_learning_rate)

        self.critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name='critic_1', learning_rate=critic_learning_rate)

        self.critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name='critic_2', learning_rate=critic_learning_rate)

        #target networks

        self.target_actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                         fc2_dims=layer2_size, n_actions=n_actions,
                                         name='target_actor', learning_rate=actor_learning_rate)

        self.target_critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                             fc2_dims=layer2_size, n_actions=n_actions,
                                             name='target_critic_1', learning_rate=critic_learning_rate)

        self.target_critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                             fc2_dims=layer2_size, n_actions=n_actions,
                                             name='target_critic_2', learning_rate=critic_learning_rate)
        self.noise = noise

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, validation=False):
        #pass you can fill it in later

        if self.time_step < self.warmup and validation is False: #while in warmup just call random tensor of actions
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device).to(self.actor.device)

        #exploration every few actions you take a random action
        # print("Mu: ", mu)
        noise = T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        # print("Noise: ", noise)
        mu_prime = mu + noise #add noise to our action
        # print("Mu prime", mu_prime)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        # exploration

        self.time_step += 1

        return mu_prime.cpu().detach().numpy() #move to the cpu and move to numpy

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self): #most of the logic
        if self.memory.mem_ctr < self.batch_size * 10: #only start learning if you have enough data
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device) #important to send to the right device
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(next_state)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # might break if elements of min and max are not all equal.
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0]) #forces actor to explor new things

        next_q1_ = self.target_critic_1.forward(next_state, target_actions)
        next_q2_ = self.target_critic_2.forward(next_state, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        next_q1_[done] = 0.0
        next_q2_[done] = 0.0

        q1_ = next_q1_.view(-1)
        q2_ = next_q2_.view(-1)

        next_critic_value = T.min(q1_, q2_)
        target = reward + self.gamma * next_critic_value #understand the value the events colser to the solution are more valuable
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state)) #critic value vs the actors own decision
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):

        if tau == None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict: #bringing the critic value closer to the target
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + (1 - tau) * \
                                        target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_1_state_dict[name] = tau * critic_2_state_dict[name].clone() + (1 - tau) * \
                                        target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[
                name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        try:
            self.actor.load_checkpoint()
            print(f"Successfully loaded actor model on {self.actor.device}")
            self.target_actor.load_checkpoint()
            print(f"Successfully loaded target actor model on {self.target_actor.device}")
            self.critic_1.load_checkpoint()
            print(f"Successfully loaded critic 1 model on {self.critic_1.device}")
            self.critic_2.load_checkpoint()
            print(f"Successfully loaded critic 2 model on {self.critic_2.device}")
            self.target_critic_1.load_checkpoint()
            print(f"Successfully loaded target critic 1 model on {self.target_critic_1.device}")
            self.target_critic_2.load_checkpoint()
            print(f"Successfully loaded target critic 2 model on {self.target_critic_2.device}")
            print("-----------------------------------------")
            print("Successfully loaded models")
        except:
            print("Failed to load models. Starting from scratch")
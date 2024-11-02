import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import torch

class CriticNetwork(nn.Module): # gives all modules from torch

    ####critic network that decides if an action was good or bad

    #actor_path = os.path.join("tmp", "td3", "actor_td3") #create dirrectoris if they dont exist
    #os.makedirs(os.path.dirname(actor_path), exist_ok=True)
    #os.makedirs("tmp/td3", exist_ok=True) #create dirrectoris if they dont exist
    def __init__(self,input_dims, n_actions,fc1_dims=256,fc2_dims=128,name='critic',checkpoint_dir='tmp/td3',learning_rate=10e-3):
        super(CriticNetwork,self).__init__()

        self.input_dims = input_dims #enviroemt
        self.n_actions = n_actions #robtos actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'_td3') #if you stop it mid learning ot will continue where it was cancelled

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.AdamW(self.parameters(),lr=learning_rate,weight_decay=0.005) #weight decay encurages more exploration

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        print(f"Create Critic Network on device: {self.device}")  #validate that we are running on the right device

        self.to(self.device)

    def forward(self,state,action):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(state, np.ndarray): #converting numpy to tensor
            state = torch.from_numpy(state).float().to(device)

        action_value = self.fc1(T.cat([state,action],dim=1)) #running on Tcat
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q1 = self.q1(action_value)

        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file)) #weights_only=False
    ####critic network that decides if an action was good or bad

    ####actor network gets a state and ecides what to do with it

class ActorNetwork(nn.Module):

    def __init__(self, input_dims, fc1_dims=256,fc2_dims=128, learning_rate=10e-3, n_actions=2, name='actor', checkpoint_dir='tmp/td3'):

        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        #self.bn1 = nn.BatchNorm2d(self.fc1_dims)  # BatchNorm1d layer added
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #self.bn2 = nn.BatchNorm2d(self.fc2_dims)  # BatchNorm1d layer added
        self.output = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate) #difference between adam and adamW is that more exploration if one is andm and other is adamw
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self,state):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(state, np.ndarray): #converting numpy to tensor
            state = torch.from_numpy(state).float().to(device)
        #print(state)
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = T.tanh(self.output(x))

        return x


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

####actor network gets a state and ecides what to do with it
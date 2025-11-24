import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_NUM = 32
GRU_H = 64
ACTION_EPS = 1e-4
GAMMA = 0.99
EPS = 0.2  # PPO2 epsilon

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, feature_num=FEATURE_NUM, gru_hidden=GRU_H):
        super(Actor, self).__init__()
        # Actor network
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.feature_num = feature_num
        self.gru_hidden = gru_hidden

        # per-channel linear encoders (applied per time step)
        # we will build a per-time-step feature of size FEATURE_NUM * 6
        self.fc1_actor = nn.Linear(1, self.feature_num)
        self.fc2_actor = nn.Linear(1, self.feature_num)
        self.conv1_actor = nn.Linear(1, self.feature_num)   # change from self.s_dim[1]
        self.conv2_actor = nn.Linear(1, self.feature_num)   # change from self.s_dim[1]
        self.conv3_actor = nn.Linear(1, self.feature_num)   # change from self.a_dim
        self.fc3_actor = nn.Linear(1, self.feature_num)

        # GRU over S_LEN timesteps: input per step is concatenation of six FEATURE_NUM vectors
        self.ctx = nn.GRU(self.feature_num * self.s_dim[0], self.gru_hidden, batch_first=True)
        # map GRU output to FEATURE_NUM, then to action logits
        self.fc4_actor = nn.Linear(self.gru_hidden, self.feature_num)
        self.pi_head = nn.Linear(self.feature_num, action_dim)

    def forward(self, inputs):
        # inputs: [B, 6, S_LEN]
        B, C, T = inputs.shape

        # build per-time-step encodings for each of the 6 channels
        # scalar channels use the last time slice per step; vector channels use full slice
        # channel 0: last quality (scalar)
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, :].transpose(1, 2)))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, :].transpose(1, 2)))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :].transpose(1, 2)))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :].transpose(1, 2)))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :].transpose(1, 2)))
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, :].transpose(1, 2)))

        # concatenate per-channel features along feature dim: [B, T, FEATURE_NUM*6]
        merge_seq = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], dim=2)

        # run GRU over time dimension T (S_LEN) and use the last output for policy
        out, _h = self.ctx(merge_seq)  # out: [B, T, GRU_H]
        last_out = out[:, -1, :]      # [B, GRU_H]
        pi_net = F.relu(self.fc4_actor(last_out))
        pi = F.softmax(self.pi_head(pi_net), dim=-1)
        pi = torch.clamp(pi, ACTION_EPS, 1. - ACTION_EPS)
        return pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, feature_num=FEATURE_NUM, gru_hidden=GRU_H):
        super(Critic, self).__init__()
        # Critic network
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.feature_num = feature_num
        self.gru_hidden = gru_hidden
        # same per-channel encoders as Actor (all scalar per time step)
        self.fc1_actor = nn.Linear(1, self.feature_num)
        self.fc2_actor = nn.Linear(1, self.feature_num)
        self.conv1_actor = nn.Linear(1, self.feature_num)
        self.conv2_actor = nn.Linear(1, self.feature_num)
        self.conv3_actor = nn.Linear(1, self.feature_num)
        self.fc3_actor = nn.Linear(1, self.feature_num)

        # GRU over S_LEN timesteps
        self.ctx = nn.GRU(self.feature_num * self.s_dim[0], self.gru_hidden, batch_first=True)
        self.fc4_actor = nn.Linear(self.gru_hidden, self.feature_num)
        self.val_head = nn.Linear(self.feature_num, 1)

    def forward(self, inputs):
        # inputs: [B, 6, S_LEN]
        B, C, T = inputs.shape

        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, :].transpose(1, 2)))  # [B, T, FEATURE_NUM]
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, :].transpose(1, 2)))  # [B, T, FEATURE_NUM]
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :].transpose(1, 2)))  # [B, T, FEATURE_NUM]
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :].transpose(1, 2)))  # [B, T, FEATURE_NUM]
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :].transpose(1, 2)))  # [B, T, FEATURE_NUM]
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, :].transpose(1, 2)))  # [B, T, FEATURE_NUM]

        merge_seq = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], dim=2)  # [B, T, FEATURE_NUM*6]

        out, _h = self.ctx(merge_seq)  # [B, T, GRU_H]
        last_out = out[:, -1, :]       # [B, GRU_H]
        value_net = F.relu(self.fc4_actor(last_out))
        value = self.val_head(value_net)
        return value
    
class Network():
    def __init__(self, state_dim, action_dim, learning_rate, device=None,
                 feature_num=FEATURE_NUM, gru_hidden=GRU_H):
        self.s_dim = state_dim
        self.action_dim = action_dim
        self._entropy_weight = np.log(action_dim)
        self.H_target = 0.1
        self.PPO_TRAINING_EPO = 5

        self.feature_num = feature_num
        self.gru_hidden = gru_hidden
        self.device = device or DEVICE
        if self.device.type == 'cuda':
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(device_index)

        self.lr_rate = learning_rate
        self._build_models()

    def _build_models(self):
        self.actor = Actor(self.s_dim, self.action_dim,
                           feature_num=self.feature_num,
                           gru_hidden=self.gru_hidden).to(self.device)
        self.critic = Critic(self.s_dim, self.action_dim,
                             feature_num=self.feature_num,
                             gru_hidden=self.gru_hidden).to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + \
                                    list(self.critic.parameters()), lr=self.lr_rate)

    def get_network_params(self):
        actor_state = {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}
        critic_state = {k: v.detach().cpu() for k, v in self.critic.state_dict().items()}
        return {
            'feature_num': self.feature_num,
            'gru_hidden': self.gru_hidden,
            'actor_state': actor_state,
            'critic_state': critic_state,
        }
    
    def set_network_params(self, input_network_params):
        if isinstance(input_network_params, dict):
            feature_num = input_network_params.get('feature_num', self.feature_num)
            gru_hidden = input_network_params.get('gru_hidden', self.gru_hidden)
            actor_net_params = input_network_params.get('actor_state')
            critic_net_params = input_network_params.get('critic_state')
        else:
            actor_net_params, critic_net_params = input_network_params
            feature_num = self._infer_feature_num(actor_net_params)
            gru_hidden = self._infer_gru_hidden(actor_net_params)

        if feature_num != self.feature_num or gru_hidden != self.gru_hidden:
            self.feature_num = feature_num
            self.gru_hidden = gru_hidden
            self._build_models()

        self.actor.load_state_dict(actor_net_params)
        self.critic.load_state_dict(critic_net_params)
        self.actor.to(self.device)
        self.critic.to(self.device)

    @staticmethod
    def _infer_feature_num(actor_state_dict):
        weight = actor_state_dict.get('fc1_actor.weight')
        if weight is not None:
            return weight.shape[0]
        return FEATURE_NUM

    @staticmethod
    def _infer_gru_hidden(actor_state_dict):
        weight = actor_state_dict.get('ctx.weight_ih_l0')
        if weight is not None:
            return weight.shape[0] // 3
        return GRU_H

    def r(self, pi_new, pi_old, acts):
        return torch.sum(pi_new * acts, dim=1, keepdim=True) / \
               torch.sum(pi_old * acts, dim=1, keepdim=True)

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        s_batch = torch.from_numpy(s_batch).to(self.device, dtype=torch.float32)
        a_batch = torch.from_numpy(a_batch).to(self.device, dtype=torch.float32)
        p_batch = torch.from_numpy(p_batch).to(self.device, dtype=torch.float32)
        v_batch = torch.from_numpy(v_batch).to(self.device, dtype=torch.float32)

        for _ in range(self.PPO_TRAINING_EPO):
            pi = self.actor.forward(s_batch)
            val = self.critic.forward(s_batch)

            # loss
            adv = v_batch - val.detach()
            ratio = self.r(pi, p_batch, a_batch)
            ppo2loss = torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv)
            # Dual-PPO
            dual_loss = torch.where(adv < 0, torch.max(ppo2loss, 3. * adv), ppo2loss)
            loss_entropy = torch.sum(-pi * torch.log(pi), dim=1, keepdim=True)

            loss = -dual_loss.mean() + 10. * F.mse_loss(val, v_batch) - self._entropy_weight * loss_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update entropy weight
        _H = (-(torch.log(p_batch) * p_batch).sum(dim=1)).mean().item()
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
        self._entropy_weight = max(self._entropy_weight, 1e-2)

    def predict(self, input):
        with torch.no_grad():
            input_t = torch.from_numpy(input).to(self.device, dtype=torch.float32)
            pi = self.actor.forward(input_t)[0]
            return pi.cpu().numpy()

    def load_model(self, nn_model):
        payload = torch.load(nn_model, map_location='cpu')
        self.set_network_params(payload)

    def save_model(self, nn_model):
        model_params = self.get_network_params()
        torch.save(model_params, nn_model)

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        R_batch = np.zeros_like(r_batch)

        if terminal:
            # in this case, the terminal reward will be assigned as r_batch[-1]
            R_batch[-1] = r_batch[-1]  # terminal state
        else:
            s_t = torch.from_numpy(np.array(s_batch)).to(self.device, dtype=torch.float32)
            val = self.critic.forward(s_t)
            R_batch[-1] = val[-1].detach().cpu().item()  # bootstrap from last state

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t] = r_batch[t] + GAMMA * R_batch[t + 1]

        return list(R_batch)
           

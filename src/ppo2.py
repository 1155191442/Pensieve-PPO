import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import inspect

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_NUM = 64
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

        self._last_loaded_epoch = None
        self._last_trained_epoch = None
        self.lr_rate = learning_rate
        self._build_models()

    def _build_models(self):
        self.actor = Actor(self.s_dim, self.action_dim,
                           feature_num=self.feature_num,
                           gru_hidden=self.gru_hidden).to(self.device)
        self.critic = Critic(self.s_dim, self.action_dim,
                             feature_num=self.feature_num,
                             gru_hidden=self.gru_hidden).to(self.device)
        self._compile_optimizer()

    def _compile_optimizer(self, params=None, lr=None):
        if params is None:
            params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=lr or self.lr_rate)

    @staticmethod
    def _move_optimizer_state(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    @staticmethod
    def _apply_trainable_filter(model, allowed_modules):
        for name, param in model.named_parameters():
            module_name = name.split('.')[0]
            trainable = True if allowed_modules is None else module_name in allowed_modules
            param.requires_grad_(trainable)

    @staticmethod
    def _is_actor_state_dict(state_dict):
        return isinstance(state_dict, dict) and any(k.startswith('pi_head') for k in state_dict.keys())

    @staticmethod
    def _is_critic_state_dict(state_dict):
        return isinstance(state_dict, dict) and any(k.startswith('val_head') for k in state_dict.keys())

    def configure_trainable_params(self, actor_modules=None, critic_modules=None, lr=None):
        self._apply_trainable_filter(self.actor, actor_modules)
        self._apply_trainable_filter(self.critic, critic_modules)

        params = [p for p in self.actor.parameters() if p.requires_grad]
        params += [p for p in self.critic.parameters() if p.requires_grad]
        if not params:
            raise ValueError('No trainable parameters selected for optimizer.')
        self._compile_optimizer(params=params, lr=lr or self.lr_rate)

    def get_network_params(self):
        actor_state = {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}
        critic_state = {k: v.detach().cpu() for k, v in self.critic.state_dict().items()}
        return {
            'feature_num': self.feature_num,
            'gru_hidden': self.gru_hidden,
            'actor_state': actor_state,
            'critic_state': critic_state,
        }
    
    def _parse_checkpoint_payload(self, payload):
        feature_num = None
        gru_hidden = None
        actor_state = None
        critic_state = None

        if isinstance(payload, dict):
            # newest format
            if 'actor_state' in payload or 'critic_state' in payload:
                actor_state = payload.get('actor_state')
                critic_state = payload.get('critic_state')
                feature_num = payload.get('feature_num')
                gru_hidden = payload.get('gru_hidden')
            # older dict-style checkpoints
            elif 'actor' in payload and 'critic' in payload:
                actor_state = payload['actor']
                critic_state = payload['critic']
            # nested containers such as {'state_dict': {...}}
            elif 'state_dict' in payload:
                nested = payload['state_dict']
                if isinstance(nested, dict):
                    actor_state = nested.get('actor_state') or nested.get('actor')
                    critic_state = nested.get('critic_state') or nested.get('critic')

            if (actor_state is None or critic_state is None):
                for value in payload.values():
                    if actor_state is None and self._is_actor_state_dict(value):
                        actor_state = value
                        continue
                    if critic_state is None and self._is_critic_state_dict(value):
                        critic_state = value
        elif isinstance(payload, (list, tuple)) and len(payload) == 2:
            actor_state, critic_state = payload

        if actor_state is None or critic_state is None:
            raise ValueError('Unrecognized checkpoint payload; unable to locate actor/critic weights.')

        feature_num = feature_num or self._infer_feature_num(actor_state)
        gru_hidden = gru_hidden or self._infer_gru_hidden(actor_state)
        return feature_num, gru_hidden, actor_state, critic_state

    def set_network_params(self, input_network_params):
        feature_num, gru_hidden, actor_net_params, critic_net_params = \
            self._parse_checkpoint_payload(input_network_params)

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
        self._last_trained_epoch = epoch

    def online_adaptation_step(self, state, action, old_action_prob, reward,
                               next_state, terminal, num_updates=1):
        if not isinstance(num_updates, int) or num_updates < 1:
            num_updates = 1

        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
        action_t = torch.from_numpy(action).unsqueeze(0).to(self.device, dtype=torch.float32)
        old_prob_t = torch.from_numpy(old_action_prob).unsqueeze(0).to(self.device, dtype=torch.float32)
        old_prob_t = torch.clamp(old_prob_t, ACTION_EPS, 1. - ACTION_EPS)
        reward_t = torch.tensor([[reward]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if terminal:
                target_value = reward_t
            else:
                next_state_t = torch.from_numpy(next_state).unsqueeze(0).to(self.device, dtype=torch.float32)
                next_val = self.critic.forward(next_state_t)
                target_value = reward_t + GAMMA * next_val

        for _ in range(num_updates):
            pi = self.actor.forward(state_t)
            val = self.critic.forward(state_t)

            adv = target_value - val.detach()
            ratio = self.r(pi, old_prob_t, action_t)
            ppo2loss = torch.min(ratio * adv,
                                 torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv)
            dual_loss = torch.where(adv < 0, torch.max(ppo2loss, 3. * adv), ppo2loss)
            loss_entropy = torch.sum(-pi * torch.log(pi), dim=1, keepdim=True)
            value_loss = F.mse_loss(val, target_value)

            loss = -dual_loss.mean() + 10. * value_loss - self._entropy_weight * loss_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        lr_scale = self.optimizer.param_groups[0]['lr'] if self.optimizer else self.lr_rate
        _H = (-(pi.detach() * torch.log(pi.detach())).sum(dim=1)).mean().item()
        _g = _H - self.H_target
        self._entropy_weight -= lr_scale * _g * 0.1 * num_updates
        self._entropy_weight = max(self._entropy_weight, 1e-2)

    def predict(self, input):
        with torch.no_grad():
            input_t = torch.from_numpy(input).to(self.device, dtype=torch.float32)
            pi = self.actor.forward(input_t)[0]
            return pi.cpu().numpy()

    def load_model(self, nn_model):
        try:
            add_safe_globals = torch.serialization.add_safe_globals  # type: ignore[attr-defined]
        except AttributeError:
            add_safe_globals = None
        if add_safe_globals is not None:
            add_safe_globals([np.core.multiarray.scalar])

        load_kwargs = {'map_location': 'cpu'}
        try:
            supports_weights_only = 'weights_only' in inspect.signature(torch.load).parameters
        except (ValueError, TypeError):
            supports_weights_only = False
        if supports_weights_only:
            load_kwargs['weights_only'] = False

        try:
            payload = torch.load(nn_model, **load_kwargs)
        except TypeError:
            load_kwargs.pop('weights_only', None)
            payload = torch.load(nn_model, **load_kwargs)
        optimizer_state = None
        entropy_weight = None
        h_target = None
        lr_rate = None
        epoch = None
        if isinstance(payload, dict):
            optimizer_state = payload.get('optimizer_state')
            entropy_weight = payload.get('entropy_weight')
            h_target = payload.get('H_target')
            lr_rate = payload.get('lr_rate')
            epoch = payload.get('epoch')

        self.set_network_params(payload)

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state(self.optimizer, self.device)
        if lr_rate is not None:
            self.lr_rate = lr_rate
            for group in self.optimizer.param_groups:
                group['lr'] = self.lr_rate
        if entropy_weight is not None:
            self._entropy_weight = entropy_weight
        if h_target is not None:
            self.H_target = h_target

        self._last_loaded_epoch = epoch
        self._last_trained_epoch = epoch
        return epoch

    def save_model(self, nn_model, epoch=None):
        payload = self.get_network_params()
        payload['optimizer_state'] = self.optimizer.state_dict()
        payload['entropy_weight'] = self._entropy_weight
        payload['H_target'] = self.H_target
        payload['lr_rate'] = self.lr_rate

        if epoch is not None:
            payload['epoch'] = epoch
            self._last_trained_epoch = epoch
        elif self._last_trained_epoch is not None:
            payload['epoch'] = self._last_trained_epoch

        torch.save(payload, nn_model)

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
           

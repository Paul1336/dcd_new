import torch
import torch.nn as nn
import numpy as np
from .distributions import Categorical
from .common import *


def create_model_for_iphyre_agent(args, env, name):
    

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


    class IphyreNetwork(DeviceAwareModule):
        def __init__(
            self,
            observation_space,
            action_space,
            should_freeze_embedding=False,
        ):
            super(IphyreNetwork, self).__init__()

            self.rnn = None
            
            should_use_deep_set = True
            embedding_dim = 16
            embedding_hidden_dim = 32
            embedding_hidden_action_dim = 8

            print('should_freeze_embedding', should_freeze_embedding)
            
            self.should_use_deep_set = should_use_deep_set
            self.should_freeze_embedding = should_freeze_embedding

            if observation_space.shape == (224, 224, 3):
                shape = np.array([512])
            elif self.should_use_deep_set:
                self.block_embed_layer = nn.Sequential(
                    layer_init(nn.Linear(9, embedding_hidden_dim)),
                    nn.Tanh(),
                    layer_init(nn.Linear(embedding_hidden_dim, embedding_dim)),
                )
                self.action_embed_layer = nn.Sequential(
                    layer_init(nn.Linear(2, embedding_hidden_action_dim)),
                    nn.Tanh(),
                    layer_init(nn.Linear(embedding_hidden_action_dim, embedding_dim)),
                )

                shape = np.array([embedding_dim])
            else:
                shape = np.array(observation_space.shape)

            self.shape = shape

            self.action_num = action_space.n

            print('Action num:', self.action_num, 'Observation space:', observation_space.shape)
            print('Action space:', action_space)

            self.critic = nn.Sequential(
                layer_init(nn.Linear(shape.prod(), 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear(shape.prod(), 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                # layer_init(nn.Linear(256, action_space.n), std=0.01),
                Categorical(256, action_space.n)
            )

        def reset_critic(self):
            self.critic = nn.Sequential(
                layer_init(nn.Linear(self.shape.prod(), 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 1), std=1.0),
            )

        def get_embedding(self, x):
            if self.should_use_deep_set:
                blocks = x[:, : 12 * 9].reshape(-1, 12, 9)
                actions = x[:, 12 * 9 : 12 * 9 + 7 * 2].reshape(-1, 7, 2)
                # task_encoding is not used for now
                # task_encoding = x[:, 12 * 9 + 7 * 2 :].reshape(-1, self.num_envs)
                if self.should_freeze_embedding:
                    with torch.no_grad():
                        block_embeddings = self.block_embed_layer(blocks)
                        action_embeddings = self.action_embed_layer(actions)
                        aggregated_embedding = torch.sum(
                            block_embeddings, dim=1
                        ) + torch.sum(action_embeddings, dim=1)
                else:
                    block_embeddings = self.block_embed_layer(blocks)
                    action_embeddings = self.action_embed_layer(actions)
                    aggregated_embedding = torch.sum(block_embeddings, dim=1) + torch.sum(
                        action_embeddings, dim=1
                    )

                
                return aggregated_embedding
            else:
                return x

        @property
        def is_recurrent(self):
            return self.rnn is not None

        @property
        def recurrent_hidden_state_size(self):
            # """Size of rnn_hx."""
            if self.rnn is not None:
                return self.rnn.recurrent_hidden_state_size
            else:
                return 0

        def get_value(self, x, rnn_hxs, masks):
            embedding = self.get_embedding(x)
            return self.critic(embedding)

        def get_logits(self, x):
            embedding = self.get_embedding(x)
            logits = embedding
            for index, layer in enumerate(self.actor):
                logits = layer(logits)
                if index == 3:
                    break
            return logits



        def act(self, x, rnn_hxs, masks, deterministic=False):
            embedding = self.get_embedding(x)
            
            dist = self.actor(embedding)
            value = self.critic(embedding)

            action = dist.sample()
            if deterministic:
                action = dist.mode()

            actor_log_dist = dist.logits

            return (
                value,
                action,
                actor_log_dist,
                rnn_hxs

                # action,
                # probs.log_prob(action),
                # probs.entropy(),
                # self.get_value_logits(embedding),
                # probs,
            )

        def evaluate_actions(self, x, rnn_hxs, masks, action, return_policy_logits=False):
            embedding = self.get_embedding(x)
            
            dist = self.actor(embedding)
            value = self.critic(embedding)

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

            if return_policy_logits:
                return value, action_log_probs, dist_entropy, rnn_hxs, dist
            
            return value, action_log_probs, dist_entropy, rnn_hxs
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import sys
import os


def weight_init_keras_default(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.zeros_(m.bias.data)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight.data, - 0.05, 0.05)


def weight_init_uniform(m):
    init.uniform_(m.weight.data, - 0.05, 0.05)
    init.zeros_(m.bias.data)

def set_not_requires_grad(ms):
    if not isinstance(ms, list):
        ms = [ms]
    for m in ms:
        for param in m.parameters():
            param.requires_grad = False


class create_user_network(nn.Module):
    def __init__(self, user_input_dim, user_embedding_size, dropout = 0, momentum = False) -> None:
        super(create_user_network, self).__init__()
        self.user_input_dim = user_input_dim
        self.user_embedding_size = user_embedding_size
        self.momentum = momentum

        self.user_embeddings = nn.Embedding(self.user_input_dim, self.user_embedding_size)
        weight_init_keras_default(self.user_embeddings)
        self.Linear1 = nn.Linear(self.user_embedding_size, 48)
        weight_init_uniform(self.Linear1)
        self.dpo = nn.Dropout(p=dropout)
        self.Linear2 = nn.Linear(48, 32)
        weight_init_keras_default(self.Linear2)
        self.dpo2 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.Bn = nn.BatchNorm1d(32, momentum=0.01, eps = 0.001)
        if self.momentum:
            set_not_requires_grad([self.user_embeddings, self.Linear1, self.Linear2])

    def forward(self, user_input):
        user_embedding = self.user_embeddings(user_input)
        user_part = self.Linear1(user_embedding)
        user_part = self.dpo(user_part)
        user_part = self.relu(self.Linear2(user_part))
        user_part = self.Bn(user_part)
        user_part = self.dpo2(user_part)
        return user_part

class create_item_network(nn.Module):
    def __init__(self, item_input_dim, item_embedding_size, dropout = 0, momentum = False) -> None:
        super(create_item_network, self).__init__()
        self.item_input_dim = item_input_dim
        self.item_embedding_size = item_embedding_size
        self.momentum = momentum

        self.item_embeddings = nn.Embedding(self.item_input_dim, self.item_embedding_size)
        weight_init_keras_default(self.item_embeddings)
        self.Linear1 = nn.Linear(self.item_embedding_size, 48)
        weight_init_uniform(self.Linear1)
        self.dpo = nn.Dropout(p=dropout)
        self.Linear2 = nn.Linear(48, 32)
        weight_init_keras_default(self.Linear2)
        self.dpo2 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.Bn = nn.BatchNorm1d(32, momentum=0.01, eps = 0.001)
        if self.momentum:
            set_not_requires_grad([self.item_embeddings, self.Linear1, self.Linear2])

    def forward(self, item_input):
        item_embedding = self.item_embeddings(item_input)
        item_part = self.Linear1(item_embedding)
        item_part = self.dpo(item_part)
        item_part = self.relu(self.Linear2(item_part))
        item_part = self.Bn(item_part)
        item_part = self.dpo2(item_part)
        return item_part


class ASLNetwork(nn.Module):
    def __init__(self, user_input_dim, item_input_dim, dropout = 0, user_embedding_size=50, item_embedding_size=50) -> None:
        super(ASLNetwork, self).__init__()
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        
        self.user_embedding_size = user_embedding_size
        self.item_embedding_size = item_embedding_size

        self.user_vanilla_network = create_user_network(self.user_input_dim, self.user_embedding_size, dropout = dropout)
        self.user_momentum_network = create_user_network(self.user_input_dim, self.user_embedding_size, momentum = True)
        self.user_momentum_network.load_state_dict(self.user_vanilla_network.state_dict())
        
        self.item_vanilla_network = create_item_network(self.item_input_dim, self.item_embedding_size, dropout = dropout)
        self.item_momentum_network = create_item_network(self.item_input_dim, self.item_embedding_size, momentum = True)
        self.item_momentum_network.load_state_dict(self.item_vanilla_network.state_dict())

        self.predictor1 = self.create_predictor(32)
        weight_init_keras_default(self.predictor1)
        self.predictor2 = self.create_predictor(32)
        weight_init_keras_default(self.predictor2)

        self.predictor1_item_m = self.create_predictor(32)
        weight_init_keras_default(self.predictor1_item_m)
        self.predictor2_item_m = self.create_predictor(32)
        weight_init_keras_default(self.predictor2_item_m)

        self.predictor1_user_m = self.create_predictor(32)
        weight_init_keras_default(self.predictor1_user_m)
        self.predictor2_user_m = self.create_predictor(32)
        weight_init_keras_default(self.predictor2_user_m)

        self.pair_predictor = self.create_predictor(32)
        weight_init_keras_default(self.pair_predictor)
    
    def create_predictor(self, input_size):
        return nn.Sequential(
            nn.Linear(input_size, 1), 
            nn.Sigmoid()
        )

    def forward(self, input):
        user_input_left = input['user_left']
        user_input_right = input['user_right']
        item_input_left = input['item_left']
        item_input_right = input['item_right']

        user_vanilla_left = self.user_vanilla_network(user_input_left)
        user_momentum_left = self.user_momentum_network(user_input_left)
        user_vanilla_right = self.user_vanilla_network(user_input_right)
        user_momentum_right = self.user_momentum_network(user_input_right)

        item_vanilla_left = self.item_vanilla_network(item_input_left)
        item_momentum_left = self.item_momentum_network(item_input_left)
        item_vanilla_right = self.item_vanilla_network(item_input_right)
        item_momentum_right = self.item_momentum_network(item_input_right)

        vanilla_part_left = torch.multiply(user_vanilla_left, item_vanilla_left)
        momentum_part_left = torch.multiply(user_momentum_left, item_momentum_left)
        it_m_us_v_part_left = torch.multiply(user_vanilla_left, item_momentum_left)
        it_v_us_m_part_left = torch.multiply(user_momentum_left, item_vanilla_left)

        vanilla_part_right = torch.multiply(user_vanilla_right, item_vanilla_right)
        momentum_part_right = torch.multiply(user_momentum_right, item_momentum_right)
        it_m_us_v_part_right = torch.multiply(user_vanilla_right, item_momentum_right)
        it_v_us_m_part_right = torch.multiply(user_momentum_right, item_vanilla_right)

        preds_y1 = self.predictor1(momentum_part_left)
        preds_y2 = self.predictor2(momentum_part_right)
        preds_y1_item_m = self.predictor1_item_m(it_m_us_v_part_left)
        preds_y2_item_m = self.predictor2_item_m(it_m_us_v_part_right)
        preds_y1_user_m = self.predictor1_user_m(it_v_us_m_part_left)
        preds_y2_user_m = self.predictor2_user_m(it_v_us_m_part_right)

        preds_pair = self.pair_predictor(vanilla_part_left - vanilla_part_right)

        # MP4 Discrepancy M-M
        dist_part_left = torch.abs(vanilla_part_left - momentum_part_left)
        dist_part_right = torch.abs(vanilla_part_right - momentum_part_right)

        discrepancy_left = torch.exp(torch.mean(dist_part_left, dim=1, keepdims=True))
        discrepancy_right = torch.exp(torch.mean(dist_part_right, dim=1, keepdims=True))

        discrepancy_weights = torch.multiply(discrepancy_left, discrepancy_right)
        discrepancy_weights = 1. / discrepancy_weights

        # MP2 Discrepancy M-V
        dist_part_left_it_m_us_v = torch.abs(item_vanilla_left - it_m_us_v_part_left)
        dist_part_right_it_m_us_v = torch.abs(item_vanilla_right - it_m_us_v_part_right)

        discrepancy_left_it_m_us_v = torch.exp(torch.mean(dist_part_left_it_m_us_v, dim=1, keepdims=True))
        discrepancy_right_it_m_us_v = torch.exp(torch.mean(dist_part_right_it_m_us_v, dim=1, keepdims=True))

        discrepancy_weights_it_m_us_v = torch.multiply(discrepancy_left_it_m_us_v, discrepancy_right_it_m_us_v)
        discrepancy_weights_item_m = 1. / discrepancy_weights_it_m_us_v

        # MP2-Mirror Discrepancy V-M
        dist_part_left_it_v_us_m = torch.abs(user_vanilla_left - it_v_us_m_part_left)
        dist_part_right_it_v_us_m = torch.abs(user_vanilla_right - it_v_us_m_part_right)

        discrepancy_left_it_v_us_m = torch.exp(torch.mean(dist_part_left_it_v_us_m, dim=1, keepdims=True))
        discrepancy_right_it_v_us_m = torch.exp(torch.mean(dist_part_right_it_v_us_m, dim=1, keepdims=True))

        discrepancy_weights_it_v_us_m = torch.multiply(discrepancy_left_it_v_us_m, discrepancy_right_it_v_us_m)
        discrepancy_weights_user_m = 1. / discrepancy_weights_it_v_us_m
        return preds_y1, preds_y2, preds_y1_item_m, preds_y2_item_m, preds_y1_user_m, preds_y2_user_m, preds_pair, discrepancy_weights, discrepancy_weights_item_m, discrepancy_weights_user_m


class Structure_Alpha(nn.Module):
    def __init__(self, temperature = 1) -> None:
        super(Structure_Alpha, self).__init__()
        self.alpha_1 = nn.Parameter(torch.Tensor([0.5]))
        self.alpha_2 = nn.Parameter(torch.Tensor([0.5]))
        self.alpha_3 = nn.Parameter(torch.Tensor([0.5]))
        self.temperature = temperature

    def forward(self):
        alpha1_ = self.alpha_1 / self.temperature
        alpha2_ = self.alpha_2 / self.temperature
        alpha3_ = self.alpha_3 / self.temperature

        exp_sum = torch.exp(alpha1_) + torch.exp(alpha2_) + torch.exp(alpha3_)
        
        alpha1_ = torch.exp(alpha1_) / exp_sum
        alpha2_ = torch.exp(alpha2_) / exp_sum
        alpha3_ = torch.exp(alpha3_) / exp_sum
        return alpha1_, alpha2_, alpha3_



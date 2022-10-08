# -*- coding: utf-8 -*-

"""
@Time ： 2022/10/8 2:14 PM
@Author ： Jinming Yang
@File ： main.py

"""

import os
import pandas as pd
import argparse
from datetime import datetime
import pickle
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import logging
from collections import OrderedDict

from utils import DataProcessor, InputProducer, logger, Dataset, auto_select_gpu, build_optimizer, weight_init, plot_curve
from utils import build_score_dataframe, ndcg_score_at_k, hit_rate_at_k
from subparsers import add_ml100k_subparser, add_ml1m_subparser, add_beauty_subparser, add_office_subparser

from models.ASL import ASLNetwork, Structure_Alpha

def parquet_collate_fn(batch):
    df = pd.DataFrame(batch)
    tensor_data = dict()
    # print(df.columns)
    # print(df)
    for col in df.columns:
        # print(col)
        tensor_data[col] = torch.tensor(df[col])
    return tensor_data

class StreamingDataset(IterableDataset):
    def __init__(self, df):
        self.df = df

    def process_parquet_file(self):
        for row in self.df.itertuples(index=False):
            yield row
        
    def __iter__(self):
        return self.process_parquet_file()

def regularization_norm(ten, weight = 0.01, p = 2):
    if p == 1:
        return ten.abs().sum() * weight
    elif p == 2:
        return ten.square().sum() * weight
    else:
        raise NotImplementedError('{}-norm not implemented yet'.format(p))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--reduction', type=str, default='mean')
    parser.add_argument('--torch_seed', type=int, default=20)
    parser.add_argument('--numpy_seed', type=int, default=20)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--loss_weight_pair', type=float, default=0.6)
    parser.add_argument('--loss_weight_point1', type=float, default=0.1)
    parser.add_argument('--loss_weight_point2', type=float, default=0.1)
    
    subparsers = parser.add_subparsers()
    add_ml100k_subparser(subparsers)
    add_ml1m_subparser(subparsers)
    add_beauty_subparser(subparsers)
    add_office_subparser(subparsers)
    args = parser.parse_args()

    # Create Log Path
    model_data_name = 'ASL' + '_' + args.data + '_' + str(args.index).zfill(2)
    log_path = './res/' + model_data_name + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    handler = logger.FileHandler(log_path + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_'+ args.data + '.log')
    logger=logger.getLogger() 
    logger.addHandler(handler)
    logger.info(args)

    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        logger.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:{}'.format(cuda))
    elif torch.backends.mps.is_available():
        logger.info('Using MPS')
        device = torch.device('mps')
    else:
        logger.info('Using CPU')
        device = torch.device('cpu')

    if args.data == 'ml-100k':
        file_set = {'train': 'data/ml-100k/u1.base.data', 'validation': 'data/ml-100k/u1.valid.data', 'test': 'data/ml-100k/u1.test.data'}
        ip = InputProducer(user_vocab_size=944,
                        item_vocab_size=1683,
                        chunksize=1000000)
    elif args.data == 'beauty':
        file_set = {'train': 'data/beauty/base.data', 'validation': 'data/beauty/valid.data', 'test': 'data/beauty/test.data'}
        ip = InputProducer(user_vocab_size=50000,
                            item_vocab_size=20000,
                            chunksize=1000000
                            )
    elif args.data == 'ml-1m':
        file_set = {'train': 'data/ml-1m/base.data', 'validation': 'data/ml-1m/valid.data', 'test': 'data/ml-1m/test.data'}
        ip = InputProducer(user_vocab_size=6041,
                            item_vocab_size=3707,
                            chunksize=1000000)
    elif args.data == 'office-products':
        file_set = {'train': 'data/office-products/base.data', 'validation': 'data/office-products/valid.data', 'test': 'data/office-products/test.data'}
        ip = InputProducer(user_vocab_size=4905,
                            item_vocab_size=2420,
                            chunksize=1000000 
                            )
    else:
        raise Exception("Dataset can only be chosen from {}".format(['ml-100k', 'ml-1m', 'beauty', 'office-products']))

    # Initialize Four tower model
    torch.manual_seed(args.torch_seed)
    model = ASLNetwork(ip.user_vocab_size,
                                ip.item_vocab_size,
                                user_embedding_size=args.embedding_size,
                                item_embedding_size=args.embedding_size).to(device)

    trainable_parameters = list(model.user_vanilla_network.parameters()) \
                                + list(model.item_vanilla_network.parameters()) \
                                    + list(model.predictor1.parameters()) \
                                        + list(model.predictor2.parameters()) \
                                            + list(model.pair_predictor.parameters())

    trainable_parameters += list(model.predictor1_item_m.parameters()) \
                                + list(model.predictor2_item_m.parameters()) \
                                    + list(model.predictor1_user_m.parameters()) \
                                        + list(model.predictor2_user_m.parameters())

    opt = build_optimizer(args.opt, trainable_parameters, lr = args.lr, weight_decay = args.weight_decay)

    # Initialize tower searching module
    torch.manual_seed(args.torch_seed)
    np.random.seed(args.numpy_seed)
    str_alpha = Structure_Alpha().to(device)
    trainable_parameters_alpha = list(str_alpha.parameters())
    opt_alpha = build_optimizer(args.opt, trainable_parameters_alpha, lr = args.lr_str)


    logger.info('Number of Trainable Parameter Tensors: {}'.format(len(trainable_parameters) + len(trainable_parameters_alpha)))

    ip.produce_chunk_input(file_set, sep='\t')

    chunked_train_dfs = ip.datasets['train']
    validation_dfs = ip.datasets['validation']
    test_dfs = ip.datasets['test']

    counter = 1
    alpha_mms = []
    alpha_vms = []
    alpha_mvs = []

    logger.info(f'Training ASL model:')

    for train_df in chunked_train_dfs:
        logger.info(f'chunk - [{counter}]')
        train_df = train_df.sample(frac=1.0, random_state=42)
        train_input = ip.produce_input_df_from_dataframe('train', train_df)
        counter += 1
        data_stream = StreamingDataset(train_input)
        dataloader = DataLoader(data_stream, batch_size=args.batch_size, shuffle=False,
                                collate_fn=parquet_collate_fn, drop_last=True)

        # data_iterator = iter(dataloader)
        for epoch in range(args.epochs):
            model.train()
            str_alpha.eval()
            overall_losses = []
            pair_losses = []
            point1_loss_mms = []
            point2_loss_mms = []
            reg_losses = []
            point1_loss_vms = []
            point2_loss_vms = []
            point1_loss_mvs = []
            point2_loss_mvs = []
            batch_num = len(train_df) // args.batch_size
            with tqdm(dataloader, total = batch_num, unit="batch") as tepoch:
                for batch_data in tepoch:
                    opt.zero_grad()
                    tepoch.set_description('Epoch: {}'.format(epoch + 1))
                    for key in batch_data.keys():
                        if batch_data[key].dtype in [torch.int32, torch.int64]:
                            batch_data[key] = batch_data[key].clone().detach().to(device = device, dtype=torch.int32)
                        if batch_data[key].dtype in [torch.float32, torch.float64]:
                            batch_data[key] = batch_data[key].clone().detach().to(device = device, dtype=torch.float32)

                    preds_y1_mm, preds_y2_mm, preds_y1_item_m, preds_y2_item_m, preds_y1_user_m, preds_y2_user_m, preds_pair, discrepancy_weights_mm, discrepancy_weights_vm, discrepancy_weights_mv = model(batch_data)
                    
                    # Pairwise Loss
                    y_pair = batch_data['y'].float()
                    y_1 = batch_data['y1'].float()
                    y_2 = batch_data['y2'].float()
                    pair_loss = F.binary_cross_entropy(preds_pair.squeeze(), y_pair, reduction = args.reduction) * args.loss_weight_pair

                    # Pointwise Losses of M_uM_u, V_uM_i, M_uV_i tower combinarions
                    discrepancy_weights_mm = discrepancy_weights_mm.clone().detach()
                    point1_loss_mm = F.binary_cross_entropy(preds_y1_mm.squeeze(), y_1, discrepancy_weights_mm.squeeze(), reduction = args.reduction) * args.loss_weight_point1
                    point2_loss_mm = F.binary_cross_entropy(preds_y2_mm.squeeze(), y_2, discrepancy_weights_mm.squeeze(), reduction = args.reduction) * args.loss_weight_point2

                    discrepancy_weights_vm = discrepancy_weights_vm.clone().detach()
                    point1_loss_vm = F.binary_cross_entropy(preds_y1_item_m.squeeze(), y_1, discrepancy_weights_vm.squeeze(), reduction = args.reduction) * args.loss_weight_point1
                    point2_loss_vm = F.binary_cross_entropy(preds_y2_item_m.squeeze(), y_2, discrepancy_weights_vm.squeeze(), reduction = args.reduction) * args.loss_weight_point2

                    discrepancy_weights_mv = discrepancy_weights_mv.clone().detach()
                    point1_loss_mv = F.binary_cross_entropy(preds_y1_user_m.squeeze(), y_1, discrepancy_weights_mv.squeeze(), reduction = args.reduction) * args.loss_weight_point1
                    point2_loss_mv = F.binary_cross_entropy(preds_y2_user_m.squeeze(), y_2, discrepancy_weights_mv.squeeze(), reduction = args.reduction) * args.loss_weight_point2
                    
                    ## Regularization
                    regularization_loss = regularization_norm(model.user_vanilla_network.Linear1.weight, weight = args.loss_weight_reg, p = 1) + regularization_norm(model.item_vanilla_network.Linear1.weight, weight = args.loss_weight_reg, p = 1)

                    ## Overall loss
                    alpha_mm, alpha_vm, alpha_mv = str_alpha()
                    loss = pair_loss + alpha_mm * point1_loss_mm + alpha_vm * point1_loss_vm + alpha_mv * point1_loss_mv + alpha_mm * point2_loss_mm + alpha_vm * point2_loss_vm + alpha_mv * point2_loss_mv + regularization_loss

                    ## Gradient based update for vanilla towers and predictors
                    loss.backward()
                    opt.step()

                    overall_losses.append(loss.item())
                    pair_losses.append(pair_loss.item()) 
                    point1_loss_mms.append(point1_loss_mm.item())
                    point2_loss_mms.append(point2_loss_mm.item())
                    point1_loss_vms.append(point1_loss_vm.item())
                    point2_loss_vms.append(point2_loss_vm.item())
                    point1_loss_mvs.append(point1_loss_mv.item())
                    point2_loss_mvs.append(point2_loss_mv.item())
                    reg_losses.append(regularization_loss.item())
                        
                    ## Momentum Update for item momentum tower
                    item_vanilla_network_state = model.item_vanilla_network.state_dict()
                    item_momentum_network_state = model.item_momentum_network.state_dict()
                    item_state_names = item_vanilla_network_state.keys()
                    new_item_momentum_network_state = OrderedDict({key: (item_momentum_network_state[key] * args.beta + item_vanilla_network_state[key] * (1 - args.beta)).clone().detach() for key in item_state_names})
                    model.item_momentum_network.load_state_dict(new_item_momentum_network_state)
                    
                    ## Momentum Update for user momentum tower
                    user_vanilla_network_state = model.user_vanilla_network.state_dict()
                    user_momentum_network_state = model.user_momentum_network.state_dict()
                    user_state_names = user_vanilla_network_state.keys()
                    new_user_momentum_network_state = OrderedDict({key: (user_momentum_network_state[key] * args.beta + user_vanilla_network_state[key] * (1 - args.beta)).clone().detach() for key in user_state_names})
                    model.user_momentum_network.load_state_dict(new_user_momentum_network_state)

                    tepoch.set_postfix(loss="{:.4f}".format(loss.item()), pair_loss="{:.4f}".format(pair_loss.item()), point1_loss="{:.4f}".format(point1_loss_mm.item()),
                                            point1_loss_vm="{:.4f}".format(point1_loss_vm.item()),
                                            point1_loss_mv="{:.4f}".format(point1_loss_mv.item()),
                                            reg_loss="{:.4f}".format(regularization_loss.item()))

            logger.info(f'Epoch: {epoch + 1}. Overall loss: {np.nanmean(overall_losses)}, pairwise loss: {np.nanmean(pair_losses)}, item1 pointwise loss: {np.nanmean(point1_loss_mms)}, item2 pointwise loss: {np.nanmean(point2_loss_mms)}, item1 pointwise loss (item momentum): {np.nanmean(point1_loss_vms)}, item2 pointwise loss (item momentum): {np.nanmean(point2_loss_vms)}, item1 pointwise loss (user momentum): {np.nanmean(point1_loss_mvs)}, item2 pointwise loss (user momentum): {np.nanmean(point2_loss_mvs)}, regularization loss: {np.nanmean(reg_losses)}')
            logger.info(f'Epoch: {epoch + 1}. Alpha1: {alpha_mm.item()}, Alpha2: {alpha_vm.item()}, Alpha3: {alpha_mv.item()}')
            
            # =========== Validation ============
            # Validation Set Training for tower combination searching
            logger.info('Start tower combination searching on the validation set')
            validation_df = validation_dfs.sample(frac=1.0, random_state=42)
            validation_input = ip.produce_input_df_from_dataframe('validation', validation_df)
            validation_data_stream = StreamingDataset(validation_input)
            validation_dataloader = DataLoader(validation_data_stream, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=parquet_collate_fn, drop_last=True)

            model.eval()
            str_alpha.train()
            overall_losses = []
            point1_loss_mms = []
            point2_loss_mms = []
            point1_loss_vms = []
            point2_loss_vms = []
            point1_loss_mvs = []
            point2_loss_mvs = []

            batch_num = len(validation_df) // args.batch_size
            with tqdm(validation_dataloader, total = batch_num, unit="batch") as tepoch:
                for batch_data in tepoch:
                    opt_alpha.zero_grad()
                    tepoch.set_description('Validation Epoch: {}'.format(epoch + 1))
                    for key in batch_data.keys():
                        if batch_data[key].dtype in [torch.int32, torch.int64]:
                            batch_data[key] = batch_data[key].clone().detach().to(device = device, dtype=torch.int32)
                        if batch_data[key].dtype in [torch.float32, torch.float64]:
                            batch_data[key] = batch_data[key].clone().detach().to(device = device, dtype=torch.float32)
                    preds_y1_mm, preds_y2_mm, preds_y1_item_m, preds_y2_item_m, preds_y1_user_m, preds_y2_user_m, preds_pair, discrepancy_weights_mm, discrepancy_weights_vm, discrepancy_weights_mv = model(batch_data)
                    
                    y_pair = batch_data['y'].float()
                    y_1 = batch_data['y1'].float()
                    y_2 = batch_data['y2'].float()
                    
                    discrepancy_weights_mm = discrepancy_weights_mm.clone().detach()
                    # pair_loss = F.binary_cross_entropy(preds_pair.squeeze(), y_pair, reduction = args.reduction) * args.loss_weight_pair
                    point1_loss_mm = F.binary_cross_entropy(preds_y1_mm.squeeze(), y_1, discrepancy_weights_mm.squeeze(), reduction = args.reduction) * args.loss_weight_point1
                    point2_loss_mm = F.binary_cross_entropy(preds_y2_mm.squeeze(), y_2, discrepancy_weights_mm.squeeze(), reduction = args.reduction) * args.loss_weight_point2

                    discrepancy_weights_vm = discrepancy_weights_vm.clone().detach()
                    point1_loss_vm = F.binary_cross_entropy(preds_y1_item_m.squeeze(), y_1, discrepancy_weights_vm.squeeze(), reduction = args.reduction) * args.loss_weight_point1
                    point2_loss_vm = F.binary_cross_entropy(preds_y2_item_m.squeeze(), y_2, discrepancy_weights_vm.squeeze(), reduction = args.reduction) * args.loss_weight_point2

                    discrepancy_weights_mv = discrepancy_weights_mv.clone().detach()
                    point1_loss_mv = F.binary_cross_entropy(preds_y1_user_m.squeeze(), y_1, discrepancy_weights_mv.squeeze(), reduction = args.reduction) * args.loss_weight_point1
                    point2_loss_mv = F.binary_cross_entropy(preds_y2_user_m.squeeze(), y_2, discrepancy_weights_mv.squeeze(), reduction = args.reduction) * args.loss_weight_point2
                    
                    alpha_mm, alpha_vm, alpha_mv = str_alpha()
                    loss_alpha = alpha_mm * point1_loss_mm.clone().detach() + alpha_vm * point1_loss_vm.clone().detach() + alpha_mv * point1_loss_mv.clone().detach() + alpha_mm * point2_loss_mm.clone().detach() + alpha_vm * point2_loss_vm.clone().detach() + alpha_mv * point2_loss_mv.clone().detach() 

                    loss_alpha.backward()
                    opt_alpha.step()

                    overall_losses.append(loss_alpha.item())
                    alpha_mms.append(alpha_mm.clone().detach().cpu().numpy())
                    alpha_vms.append(alpha_vm.clone().detach().cpu().numpy())
                    alpha_mvs.append(alpha_mv.clone().detach().cpu().numpy())

                    point1_loss_mms.append(point1_loss_mm.item())
                    point2_loss_mms.append(point2_loss_mm.item())    
                    point1_loss_vms.append(point1_loss_vm.item())
                    point2_loss_vms.append(point2_loss_vm.item())
                    point1_loss_mvs.append(point1_loss_mv.item())
                    point2_loss_mvs.append(point2_loss_mv.item())
                
                tepoch.set_postfix(loss="{:.4f}".format(loss.item()), alpha_mm="{:.4f}".format(alpha_mm.item()), alpha_vm="{:.4f}".format(alpha_vm.item()), alpha_mv="{:.4f}".format(alpha_mv.item()))

            logger.info(f'Epoch: {epoch + 1}. Validation overall loss: {np.nanmean(overall_losses)}, item1 pointwise loss (mm): {np.nanmean(point1_loss_mms)}, item2 pointwise loss (mm): {np.nanmean(point2_loss_mms)}, item1 pointwise loss (vm): {np.nanmean(point1_loss_vms)}, item2 pointwise loss (vm): {np.nanmean(point2_loss_vms)}, item1 pointwise loss (mv): {np.nanmean(point1_loss_mvs)}, item2 pointwise loss (mv): {np.nanmean(point2_loss_mvs)}')
            logger.info(f'Epoch: {epoch + 1}. Validation Alpha_mm: {alpha_mm.item()}, Alpha_vm: {alpha_vm.item()}, Alpha_mv: {alpha_mv.item()}')
    
            torch.save(model, log_path + 'model.pt')
            torch.save(str_alpha, log_path + 'alpha.pt')
            out = {'alpha_mms': alpha_mms, 'alpha_vms': alpha_vms, 'alpha_mvs': alpha_mvs}
            pickle.dump(out, open(log_path + 'result.pkl', "wb"))
            plot_curve(out, log_path + 'alphas.png')
            print()
                
            # =========== Test ============
            # Test Model
            test_df = test_dfs.sample(frac=1.0, random_state=42)
            test_input = ip.produce_input_df_from_dataframe('test', test_df)
            counter += 1
            test_data_stream = StreamingDataset(test_input)
            test_dataloader = DataLoader(test_data_stream, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=parquet_collate_fn, drop_last=True)

            batch_num = len(test_df) // args.batch_size

            # data_iterator = iter(dataloader)
            test_pred1s_mm = []
            test_pred1s_vm = []
            test_pred1s_mv = []
            test_pred1s_comb = []
            test_y1s = []
            test_uids = []
            test_iid1s = []
            model.eval()
            str_alpha.eval()
            # with tqdm(test_dataloader, total = batch_num, unit="batch") as tepoch:
            for batch_data in test_dataloader:
                # tepoch.set_description('Testing: {}'.format(epoch + 1))
                for key in batch_data.keys():
                    if batch_data[key].dtype in [torch.int32, torch.int64]:
                        batch_data[key] = batch_data[key].clone().detach().to(device = device, dtype=torch.int32)
                    if batch_data[key].dtype in [torch.float32, torch.float64]:
                        batch_data[key] = batch_data[key].clone().detach().to(device = device, dtype=torch.float32)

                preds_y1, preds_y2, preds_y1_item_m, preds_y2_item_m, preds_y1_user_m, preds_y2_user_m, preds_pair, discrepancy_weights, discrepancy_weights_item_m, discrepancy_weights_user_m = model(batch_data)
                alpha_mm, alpha_vm, alpha_mv = str_alpha()

                y_1 = batch_data['y1'].float()
                
                test_uids.append(batch_data['uid'])
                test_iid1s.append(batch_data['item_left'])
                test_y1s.append(y_1)
                test_pred1s_mm.append(preds_y1.squeeze())
                test_pred1s_vm.append(preds_y1_item_m.squeeze())
                test_pred1s_mv.append(preds_y1_user_m.squeeze())
                test_pred1_comb = preds_y1 * alpha_mm + preds_y1_item_m * alpha_vm + preds_y1_user_m * alpha_mv
                test_pred1s_comb.append(test_pred1_comb.squeeze())

            test_uids = torch.cat(test_uids, dim = 0).clone().detach().to('cpu').numpy()
            test_iid1s = torch.cat(test_iid1s, dim = 0).clone().detach().to('cpu').numpy()
            test_y1s = torch.cat(test_y1s, dim = 0).clone().detach().to('cpu').numpy()

            test_pred1s_mm = torch.cat(test_pred1s_mm, dim = 0).clone().detach().to('cpu').numpy()
            test_pred1s_vm = torch.cat(test_pred1s_vm, dim = 0).clone().detach().to('cpu').numpy()
            test_pred1s_mv = torch.cat(test_pred1s_mv, dim = 0).clone().detach().to('cpu').numpy()
            test_pred1s_comb = torch.cat(test_pred1s_comb, dim = 0).clone().detach().to('cpu').numpy()


            logger.info('Metrics')
            logger.info('User momentum - item momentum')
            score_df = build_score_dataframe(group_key=test_uids,
                                            y_true=test_y1s,
                                            y_preds=test_pred1s_mm,
                                            remove_dupes=True,
                                            dup_indicator=test_iid1s)
            # print(score_df)
            k_list = [5, 10]
            for k in k_list:
                ndcg_score = ndcg_score_at_k(score_df=score_df, k=k)
                hit_rate = hit_rate_at_k(score_df=score_df, k=k)
                logger.info(f'ndcg score at {k} (mean): {ndcg_score}')
                logger.info(f'hit rate at {k} (mean): {hit_rate}')

            logger.info('User vanilla - item momentum')
            score_df = build_score_dataframe(group_key=test_uids,
                                            y_true=test_y1s,
                                            y_preds=test_pred1s_vm,
                                            remove_dupes=True,
                                            dup_indicator=test_iid1s)
            # print(score_df)
            k_list = [5, 10]
            for k in k_list:
                ndcg_score = ndcg_score_at_k(score_df=score_df, k=k)
                hit_rate = hit_rate_at_k(score_df=score_df, k=k)
                logger.info(f'ndcg score at {k} (mean): {ndcg_score}')
                logger.info(f'hit rate at {k} (mean): {hit_rate}')

            logger.info('User momentum - item vanilla')
            score_df = build_score_dataframe(group_key=test_uids,
                                            y_true=test_y1s,
                                            y_preds=test_pred1s_mv,
                                            remove_dupes=True,
                                            dup_indicator=test_iid1s)
            # print(score_df)
            k_list = [5, 10]
            for k in k_list:
                ndcg_score = ndcg_score_at_k(score_df=score_df, k=k)
                hit_rate = hit_rate_at_k(score_df=score_df, k=k)
                logger.info(f'ndcg score at {k} (mean): {ndcg_score}')
                logger.info(f'hit rate at {k} (mean): {hit_rate}')

            logger.info('Weighted Sum')
            score_df = build_score_dataframe(group_key=test_uids,
                                            y_true=test_y1s,
                                            y_preds=test_pred1s_comb,
                                            remove_dupes=True,
                                            dup_indicator=test_iid1s)
            # print(score_df)
            k_list = [5, 10]
            for k in k_list:
                ndcg_score = ndcg_score_at_k(score_df=score_df, k=k)
                hit_rate = hit_rate_at_k(score_df=score_df, k=k)
                logger.info(f'Combine ndcg score at {k} (mean): {ndcg_score}')
                logger.info(f'Combine hit rate at {k} (mean): {hit_rate}')



# -*- coding: utf-8 -*-

"""
@Time ： 2021/2/7 3:04 PM
@Author ： Anonymity
@File ： utils.py

"""

import os
import random
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
from typing import List, Dict
from tqdm import tqdm
import logging
import subprocess
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn
import matplotlib.pyplot as plt

logger = logging
logger.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def plot_curve(data, plot_file):
    plt.plot(data['alpha_mms'], label='alpha_mm')
    plt.plot(data['alpha_vms'], label='alpha_vm')
    plt.plot(data['alpha_mvs'], label='alpha_mv')
    plt.legend()
    plt.savefig(plot_file)
    plt.close()

def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Embedding):
        init.xavier_normal_(m.weight.data)
        if m.padding_idx is not None:
            m._fill_padding_idx_with_zero()
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def build_optimizer(opt, params, lr = 1e-4, weight_decay = 0):
    filter_fn = filter(lambda p : p.requires_grad, params)
    if opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay, eps=1e-07)
    elif opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=lr, momentum=0.95, weight_decay=weight_decay)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=lr, weight_decay=weight_decay)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=lr, weight_decay=weight_decay)
    return optimizer

# get gpu usage
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def auto_select_gpu(memory_threshold = 7000, smooth_ratio=200, strategy='greedy'):
    gpu_memory_raw = get_gpu_memory_map() + 10
    if strategy=='random':
        gpu_memory = gpu_memory_raw/smooth_ratio
        gpu_memory = gpu_memory.sum() / (gpu_memory+10)
        gpu_memory[gpu_memory_raw>memory_threshold] = 0
        gpu_prob = gpu_memory / gpu_memory.sum()
        cuda = str(np.random.choice(len(gpu_prob), p=gpu_prob))
        print('GPU select prob: {}, Select GPU {}'.format(gpu_prob, cuda))
    elif strategy == 'greedy':
        cuda = np.argmin(gpu_memory_raw)
        print('GPU mem: {}, Select GPU {}'.format(gpu_memory_raw[cuda], cuda))
    return cuda

# column names in raw data, e.g. u1.base
class Columns:
    uid = 'uid'
    iid = 'iid'
    rating = 'rating'
    timestamp = 'timestamp'
    names = [uid, iid, rating, timestamp]


class Pair:
    def __init__(self, iid1: int, iid2: int, genre1: str, genre2: str, label1: int, label2: int, pairwise_label: int):
        self.iid1 = iid1
        self.iid2 = iid2
        self.genre1 = genre1
        self.genre2 = genre2
        self.label1 = label1
        self.label2 = label2
        self.pairwise_label = pairwise_label


class Dataset:
    header = ['uid', 'iid1', 'iid2', 'genre1', 'genre2', 'label1', 'label2', 'pair_label']
    group_key = 'group_key'
    uid = 'uid'
    iid1 = 'iid1'
    iid2 = 'iid2'
    label1 = 'label1'
    label2 = 'label2'
    pair_label = 'pair_label'

    y1 = 'y1'
    y2 = 'y2'
    y = 'y'


class Score:
    user = 'user'
    item = 'item'
    label = 'label'
    preds = 'preds'
    group_key = 'group_key'


class DataProcessor:
    def __init__(self, base_path: str, key: str, sep: str = '\t', threshold: int = 5,
                 n: int = 5, seed: int = None, extra_negative_sampling: bool = False, sampling_rate: float = 0.2,
                 split_rate: float = 0.8):
        """
        :param base_path:
        :param key: e.g. u1
        :param sep:
        :param threshold: threshold for binarizing the rating
        :param n: choose n samples randomly from each user
        :param seed: to reproduce the result
        :param extra_negative_sampling: flag to indicate whether to add unseen items as negative samples
        :param sampling_rate:
        """
        self.key = key
        self.item_file = f'{base_path}/u.item'
        self.rating_file = f'{base_path}/u.data'
        self.file_prefix = os.path.join(base_path, key)
        self.data_dir = self.file_prefix[:self.file_prefix.index('data') + 5]
        self.base_file = f'{self.file_prefix}.base'
        self.test_file = f'{self.file_prefix}.test'
        self.columns = Columns
        self.names = self.columns.names
        self.sep = sep
        self.threshold = threshold
        self.n = n
        self.seed = seed
        self.extra_negative_sampling = extra_negative_sampling
        self.sampling_rate = sampling_rate
        self.split_rate = split_rate

        # customized attribute
        self.rating_df = pd.read_csv(self.rating_file, sep=self.sep, names=self.names)
        self.user_encoder = self.categorical_encoder(self.rating_df[self.columns.uid])
        self.item_encoder = self.categorical_encoder(self.rating_df[self.columns.iid])
        self.item_feature_dict = self.load_item_feature()
        self.item_ids = list(self.item_feature_dict.keys())
        self.rating_info_dict = self.load_rating_info()
        random.seed(self.seed)

    # binarize rating to binary labels
    @staticmethod
    def binarizer(rating, threshold) -> int:
        if rating >= threshold:
            return 1
        return 0

    @staticmethod
    def pairwise_labeler(r1: int, r2: int) -> int:
        if r1 > r2:
            return 1
        return 0

    @staticmethod
    def categorical_encoder(x):
        le = LabelEncoder()
        le.fit(x)
        return le

    def split_data(self):
        train_writer = open(f'{self.base_file}.enc', 'w', encoding='utf-8')
        test_writer = open(f'{self.test_file}.enc', 'w', encoding='utf-8')

        df = self.rating_df
        df['u_enc'] = self.user_encoder.transform(df[self.columns.uid])
        df['i_enc'] = self.item_encoder.transform(df[self.columns.iid])

        logger.info(f'number of users: {len(set(self.rating_df[self.columns.uid]))}')
        logger.info(f'number of items: {len(set(self.rating_df[self.columns.iid]))}')

        group_df = df.groupby(self.columns.uid)
        for user_id, info in tqdm(group_df):
            limit = int(len(info) * self.split_rate)
            # train samples
            for i in range(limit):
                row = info.iloc[i]
                train_writer.write(
                    ('{}\t' * 3 + '{}\n').format(row['u_enc'],
                                                 row['i_enc'],
                                                 row[self.columns.rating],
                                                 row[self.columns.timestamp]))
            # test samples
            for j in range(limit, len(info)):
                row = info.iloc[j]
                test_writer.write(
                    ('{}\t' * 3 + '{}\n').format(row['u_enc'],
                                                 row['i_enc'],
                                                 row[self.columns.rating],
                                                 row[self.columns.timestamp]))
        train_writer.close()
        test_writer.close()

    def make_pairs(self, user_id: int, data: pd.DataFrame) -> List[Pair]:
        size = min(self.n, len(data))
        samples = data.sample(n=size, random_state=self.seed)
        pairs, iids = [], []
        threshold = self.threshold
        for i in range(size):
            # info of item1
            idx1 = samples.index[i]
            iid1 = data.loc[idx1][self.columns.iid]
            rating1 = data.loc[idx1][self.columns.rating]
            label1 = self.binarizer(rating1, threshold)

            for j in range(i + 1, size):
                # info of item2
                idx2 = samples.index[j]
                iid2 = data.loc[idx2][self.columns.iid]
                rating2 = data.loc[idx2][self.columns.rating]
                label2 = self.binarizer(rating2, threshold)

                pairs.append(
                    Pair(iid1=iid1,
                         iid2=iid2,
                         label1=label1,
                         label2=label2,
                         genre1=self.item_feature_dict[iid1],
                         genre2=self.item_feature_dict[iid2],
                         pairwise_label=self.pairwise_labeler(rating1, rating2))
                )
            # collect positive item id
            if label1 == 1:
                iids.append(iid1)
        # add extra negative samples
        if self.extra_negative_sampling and len(iids) > 0:
            unseen_items = random.sample(self.item_ids, self.n)
            for unseen_item in unseen_items:
                if unseen_item not in self.rating_info_dict[user_id]:
                    for iid in iids:
                        bar = random.uniform(0, 1)
                        if bar < self.sampling_rate:
                            if bar < self.sampling_rate / 2:
                                pairs.append(
                                    Pair(
                                        iid1=unseen_item,
                                        iid2=iid,
                                        label1=0,
                                        label2=1,
                                        genre1=self.item_feature_dict[unseen_item],
                                        genre2=self.item_feature_dict[iid],
                                        pairwise_label=0
                                    )
                                )
                            else:
                                pairs.append(
                                    Pair(
                                        iid1=iid,
                                        iid2=unseen_item,
                                        label1=1,
                                        label2=0,
                                        genre1=self.item_feature_dict[iid],
                                        genre2=self.item_feature_dict[unseen_item],
                                        pairwise_label=1
                                    )
                                )
        return pairs

    def process(self, file_name: str):
        # judge train or test
        if file_name == self.base_file:
            writer = open(f'{self.data_dir}/{self.key}.base.data', 'w', encoding='utf-8')
        else:
            writer = open(f'{self.data_dir}/{self.key}.valid.data', 'w', encoding='utf-8')
        writer.write('\t'.join(Dataset.header) + '\n')

        df = pd.read_csv(f'{file_name}.enc', sep=self.sep, names=self.names)
        grouped_df = df.groupby(self.columns.uid)
        tqdm.pandas()
        for user_id, info in tqdm(grouped_df, desc=file_name.split('.')[-1]):
            if len(info) < 2:  # no pairs
                continue
            pairs = self.make_pairs(user_id, info)
            for pair in pairs:
                writer.write(('{}\t' * (len(Dataset.header) - 1) + '{}\n').format(
                    user_id,
                    pair.iid1, pair.iid2,
                    pair.genre1, pair.genre2,
                    pair.label1, pair.label2,
                    pair.pairwise_label))
        writer.close()

    def process_test(self):
        # generate testing data file
        writer = open(f'{self.data_dir}/{self.key}.test.data', 'w', encoding='utf-8')
        writer.write('\t'.join(Dataset.header) + '\n')

        df = pd.read_csv(f'{self.test_file}.enc', sep=self.sep, names=self.names)
        for idx, row in df.iterrows():
            user_id = row[self.columns.uid]
            iid = row[self.columns.iid]
            label = self.binarizer(row[self.columns.rating], threshold=self.threshold)
            writer.write(('{}\t' * (len(Dataset.header) - 1) + '{}\n').format(
                user_id,
                iid, iid,
                self.item_feature_dict[iid], self.item_feature_dict[iid],
                label, label,
                0
            ))

        writer.close()

    def load_item_feature(self) -> Dict[int, str]:
        feature_dict = {}
        handler = open(self.item_file, 'r', encoding='ISO-8859-1')
        for line in handler:
            item_features = line.strip().split('|', 5)
            iid, genre = int(item_features[0]), item_features[5]
            iid_enc = self.item_encoder.transform([iid])
            feature_dict[iid_enc[0]] = genre
        handler.close()
        return feature_dict

    def load_rating_info(self) -> Dict[int, list]:
        """
        Map user -> seen item list
        :return:
        """
        df = self.rating_df
        df['u_enc'] = self.user_encoder.transform(df[self.columns.uid])
        df['i_enc'] = self.item_encoder.transform(df[self.columns.iid])
        rating_info = df.groupby('u_enc')['i_enc'].apply(list).to_dict()
        return rating_info


class InputProducer:
    def __init__(self, user_vocab_size=943, item_vocab_size=1682, random_state=42,
                 chunksize: int = 10 ** 5):
        """
        :param user_vocab_size:
        :param item_vocab_size:
        :param random_state: shuffle the input data
        """
        self.user_vocab_size = user_vocab_size
        self.item_vocab_size = item_vocab_size
        self.random_state = random_state
        self.chunksize = chunksize
        self.datasets = None

    def load_chunk_data(self, file_names=None, sep='\t'):
        from collections import defaultdict
        if file_names is None:
            file_names = defaultdict(str)
        datasets = dict()
        for key in file_names:
            logger.info(f'load data from {file_names[key]}')
            if key == 'train':
                dataframe = pd.read_csv(os.path.join(file_names[key]), sep=sep,
                                        chunksize=self.chunksize)
            else:
                dataframe = pd.read_csv(os.path.join(file_names[key]), sep=sep)
            datasets[key] = dataframe
        return datasets

    def produce_input_from_dataframe(self, key, df: pd.DataFrame):
        user_input = np.asarray(df[Dataset.uid].apply(lambda x: [int(x), ]).tolist())
        logger.info(f'{key}: number of users - {len(set(df[Dataset.uid]))}')

        item1_input = np.asarray(df[Dataset.iid1].apply(lambda x: [int(x), ]).tolist())
        item2_input = np.asarray(df[Dataset.iid2].apply(lambda x: [int(x), ]).tolist())
        logger.info(f'{key}: number of items - {len(set(df[Dataset.iid1]))}')

        label1 = np.asarray(df[Dataset.label1], dtype=np.float32)
        label2 = np.asarray(df[Dataset.label2], dtype=np.float32)
        pairwise_label = np.asarray(df[Dataset.pair_label], dtype=np.float32)
        logger.info(f'{key}: number of interactions - {len(df[[Dataset.uid, Dataset.iid1]].drop_duplicates())}')
        logger.info(f'{key}: number of pairs - {len(df)}')

        data_map = {
            Dataset.uid: df[Dataset.uid], Dataset.iid1: df[Dataset.iid1],
            "left_input": [user_input, item1_input], "right_input": [user_input, item2_input],
            Dataset.y1: label1, Dataset.y2: label2, Dataset.y: pairwise_label
        }
        if Dataset.group_key in df.columns:
            data_map[Dataset.group_key] = df[Dataset.group_key]
        del df
        gc.collect()
        return data_map

    def produce_input_df_from_dataframe(self, key, df: pd.DataFrame):
        user_input = np.asarray(df[Dataset.uid].apply(lambda x: [int(x), ]).tolist())
        logger.info(f'{key}: number of users - {len(set(df[Dataset.uid]))}')

        item1_input = np.asarray(df[Dataset.iid1].apply(lambda x: [int(x), ]).tolist())
        item2_input = np.asarray(df[Dataset.iid2].apply(lambda x: [int(x), ]).tolist())
        logger.info(f'{key}: number of items - {len(set(df[Dataset.iid1]))}')

        label1 = np.asarray(df[Dataset.label1], dtype=np.float32)
        label2 = np.asarray(df[Dataset.label2], dtype=np.float32)
        pairwise_label = np.asarray(df[Dataset.pair_label], dtype=np.float32)
        logger.info(f'{key}: number of interactions - {len(df[[Dataset.uid, Dataset.iid1]].drop_duplicates())}')
        logger.info(f'{key}: number of pairs - {len(df)}')

        data_map = {
            Dataset.uid: df[Dataset.uid], Dataset.iid1: df[Dataset.iid1],
            "user_left": user_input.squeeze(), "item_left": item1_input.squeeze(), "user_right": user_input.squeeze(), "item_right": item2_input.squeeze(),
            Dataset.y1: label1, Dataset.y2: label2, Dataset.y: pairwise_label
        }
        # print([m.shape for key, m in data_map.items()])
        if Dataset.group_key in df.columns:
            data_map[Dataset.group_key] = df[Dataset.group_key]
        data_df = pd.DataFrame(data_map)
        del df
        gc.collect()
        return data_df

    def produce_chunk_input(self, file_names, sep='\t'):
        datasets = self.load_chunk_data(file_names, sep)
        for key, df in datasets.items():
            if key == 'train':
                continue
            # datasets[key] = self.produce_input_from_dataframe(key, df)
        self.datasets = datasets
        return datasets

def build_score_dataframe(group_key: List[str], y_true: List[float], y_preds: List[float],
                          remove_dupes=True, dup_indicator=None, enable_log=True):
    assert len(y_true) == len(group_key)

    score_df = pd.DataFrame({
        Score.group_key: group_key,
        Score.label: y_true,
        Score.preds: y_preds
    })
    # if enable_log:
    #     logger.info(f'shape of score dataframe: {score_df.shape}')

    if remove_dupes:
        score_df["dup_indicator"] = dup_indicator
        score_df = score_df.drop_duplicates(subset=["group_key", "dup_indicator"])
        # logger.info(f'shape of score dataframe without dupes: {score_df.shape}')
    return score_df


# evaluation metrics
def ndcg_score_at_k(score_df: pd.DataFrame, k=5) -> float:
    ndcg_scores = []
    grouped_score_df = score_df.groupby(Score.group_key)
    for q, key in grouped_score_df:
        y1 = np.asarray([list(key[Score.label])])
        y2 = np.asarray([list(key[Score.preds])])
        if len(y1[0]) > 1 and len(y2[0]) > 1:
            score = ndcg_score(y1, y2, k=k)
            ndcg_scores.append(score)
    return sum(ndcg_scores) / len(ndcg_scores)


def hit_rate_at_k(score_df: pd.DataFrame, k=5) -> float:
    hit_rates = []
    grouped_score_df = score_df.groupby(Score.group_key)
    for q, key in grouped_score_df:
        ordered_key = key.sort_values(by=[Score.preds], ascending=False)
        total_positive = min(k, sum(ordered_key[Score.label]))
        if total_positive == 0:
            continue
        hit_rates.append(sum(ordered_key[Score.label][:k]) * 1.0 / total_positive)
    return sum(hit_rates) / len(hit_rates)

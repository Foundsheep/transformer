# dataset source : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71265

import os
import time
import json
from pathlib import Path
import pandas as pd
import tensorflow as tf
from konlpy.tag import Okt
from nltk.tokenize import wordpunct_tokenize

# TODO : argparse

class DataLoader:
    def __init__(self):
        self.root_folder = Path().cwd()/'dataset'/'01.데이터'
        self.path_enkr_train = self.root_folder/'1.Training'/'원천데이터'/'일상생활및구어체_영한_train_set.json'
        self.path_enkr_val = self.root_folder/'2.Validation'/'원천데이터'/'VS1'/'일상생활및구어체_영한_valid_set.json'
        self.save_path_ko_dict = self.root_folder.parent/'ko_dict.json'
        self.save_path_en_dict = self.root_folder.parent/'en_dict.json'
        self.save_path_train_pickle = self.root_folder.parent/'df_train.pkl'
        self.save_path_val_pickle = self.root_folder.parent/'df_val.pkl'

    def load_dataset(self, num_cut=100000):
        print("--- 1. load_dataset started")
        if not os.path.exists(self.save_path_ko_dict):
            print("--- 1.1. No data exists")
            with open(self.path_enkr_train, 'r') as f:
                print(f"--- 1.1.1 path : {self.path_enkr_train}")
                enkr_train = json.load(f)

            with open(self.path_enkr_val, 'r') as f:
                enkr_val = json.load(f)

            train = [(datum['en'], datum['ko']) for datum in enkr_train['data']]
            val = [(datum['en'], datum['ko']) for datum in enkr_val['data']]
            df_train = pd.DataFrame(train).rename({0: 'en', 1: 'ko'}, axis=1)
            df_val = pd.DataFrame(val).rename({0: 'en', 1: 'ko'}, axis=1)

            # shorten the data for now
            if num_cut > 0:
                df_train = df_train.iloc[:num_cut, :]
                df_val = df_val.iloc[:int(num_cut * 0.2), :]

            df_concat = pd.concat([df_train, df_val], axis=0)
            self.en_dict, self.en_dict_inv, self.ko_dict, self.ko_dict_inv = self.get_token_dict(df_concat)

            df_train['ko_tokenized'] = self.do_tokenize_and_pad(df_train, 'ko', self.ko_dict, max_token=50)
            df_train['en_tokenized'] = self.do_tokenize_and_pad(df_train, 'en', self.en_dict, max_token=50)
            df_val['ko_tokenized'] = self.do_tokenize_and_pad(df_val, 'ko', self.ko_dict, max_token=50)
            df_val['en_tokenized'] = self.do_tokenize_and_pad(df_val, 'en', self.en_dict, max_token=50)

            self.df_train = df_train
            self.df_val = df_val
            return self.ko_dict, \
                   self.ko_dict_inv, \
                   self.en_dict, \
                   self.en_dict_inv, \
                   self.df_train, \
                   self.df_val

        else:
            print("--- 1.2. Data exists")
            with open(self.save_path_ko_dict, 'r') as f:
                self.ko_dict = json.load(f)
                self.ko_dict_inv = {v: k for k, v in self.ko_dict.items()}
            with open(self.save_path_en_dict, 'r') as f:
                self.en_dict = json.load(f)
                self.en_dict_inv = {v: k for k, v in self.en_dict.items()}

            self.df_train = pd.read_pickle(self.save_path_train_pickle)
            self.df_val = pd.read_pickle(self.save_path_val_pickle)
            return self.ko_dict, \
                   self.ko_dict_inv, \
                   self.en_dict, \
                   self.en_dict_inv, \
                   self.df_train, \
                   self.df_val

    def get_token_dict(self, df):
        print("--- 2. Tokenizing started")
        # Korean
        okt = Okt()

        start = time.time()
        ko_tokens = df['ko'].map(okt.morphs).agg(sum)
        ko_tokens_series = pd.Series(ko_tokens)
        ko_counts = ko_tokens_series.value_counts()
        ko_dict = {k: v for k, v in zip(ko_counts.keys(), range(3, len(ko_counts) + 3))}
        ko_dict['[SOS]'] = 1
        ko_dict['[EOS]'] = 2
        ko_dict_inv = {v: k for k, v in ko_dict.items()}
        end = time.time()
        print(f"--- Korean token and dict made --- {end-start :.2f} seconds")

        # English
        start = time.time()
        en_tokens = df['en'].map(wordpunct_tokenize).agg(sum)
        en_tokens_series = pd.Series(en_tokens)
        en_counts = en_tokens_series.value_counts()
        en_dict = {k: v for k, v in zip(en_counts.keys(), range(3, len(en_counts) + 3))}
        en_dict['[SOS]'] = 1
        en_dict['[EOS]'] = 2
        en_dict_inv = {v: k for k, v in en_dict.items()}
        end = time.time()
        print(f"--- English token and dict made --- {end-start :.2f} seconds")

        return en_dict, en_dict_inv, ko_dict, ko_dict_inv

    def do_tokenize_and_pad(self, df, col_name, col_dict, max_token=50):
        """
        Receives pd.DataFrame with a string in each cell, and
        returns the cell value to tokenized sentence(a list with integers)
        """
        print(f"--- 3. Tokenizing goes further ...")
        if col_name == 'ko':
            tokenize_func = Okt().morphs

        elif col_name == 'en':
            tokenize_func = wordpunct_tokenize

        SOS = [1]
        EOS = [2]
        return df[col_name].map(lambda xx: SOS + [col_dict[x] for x in tokenize_func(xx)] + EOS \
            if len(tokenize_func(xx)) == max_token \
            else SOS + [col_dict[x] for x in tokenize_func(xx)] + EOS + [0] * (max_token - len(tokenize_func(xx))))

    def save_data(self):
        print("--- 4. Save the data")
        with open(self.save_path_ko_dict, 'w') as f:
            f.write(json.dumps(self.ko_dict))
            print(f"ko_dict saved at {self.save_path_ko_dict}")

        with open(self.save_path_en_dict, 'w') as f:
            f.write(json.dumps(self.en_dict))
            print(f"en_dict saved at {self.save_path_en_dict}")

        pd.to_pickle(self.df_train, self.save_path_train_pickle)
        print(f"df_train saved at {self.save_path_train_pickle}")

        pd.to_pickle(self.df_val, self.save_path_val_pickle)
        print(f"df_val saved at {self.save_path_val_pickle}")


if __name__ == '__main__':
    dl = DataLoader()
    _1, _2, _3, _4, _5, _6 = dl.load_dataset()
    dl.save_data()

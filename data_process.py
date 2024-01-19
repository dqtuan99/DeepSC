# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:30:16 2024

@author: Tuan
"""

import numpy as np
import pickle
import json
import tensorflow as tf
import torch

class CorpusData(torch.utils.data.Dataset):
    def __init__(self, data_path, dict_path):
        self.data_as_id = self.read_dataset_from_path(data_path)
        self.max_sentence_length = self.data_as_id.shape[1]
        self.word2id_dict = self.read_dict_from_path(dict_path)
        self.id2word_dict = {v: k for k, v in self.word2id_dict.items()}
        self.num_classes = list(self.word2id_dict.values())[-1] + 1
        
    def __len__(self):
        return len(self.data_as_id)

    def __getitem__(self, index):
        post_padding_length = np.where(self.data_as_id[index] == 0)[0].shape[0]
        sentence_length = self.max_sentence_length - post_padding_length

        return self.data_as_id[index], sentence_length
    
    def get_sentence_as_word(self, index):
        sentence_as_word = []
               
        sentence_as_id, sentence_length = self.__getitem__(index)
        for i in range(1, sentence_length - 1):
            word = self.id2word_dict[sentence_as_id[i].item()]
            sentence_as_word.append(word)
        
        return ' '.join(sentence_as_word)
    
    def convert_id_sentence_to_word(self, sentence_as_id, sentence_length):
        sentence_as_word = []
        
        for i in range(0, sentence_length):
            word = self.id2word_dict[sentence_as_id[i].item()]
            sentence_as_word.append(word)
            
        return ' '.join(sentence_as_word)
    
    def read_dataset_from_path(self, path, read_length=-1):
        raw_data = pickle.load(open(path, 'rb'))
        data = raw_data[:read_length]
        data = tf.keras.preprocessing.sequence.pad_sequences(data, padding='post')
        data = torch.tensor(data)
        
        return data
    
    def read_dict_from_path(self, path):
        f = open(path)
        word2id_dict = json.load(f)
        
        return word2id_dict['token_to_idx']
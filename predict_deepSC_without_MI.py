"""
it's used to validate model trained from train.py
attention it's wired that argument snr failed to be transported into model and fading channel cannot update it's argument
so i put them into the same file instead of importing from outside
"""

import torch
import os
import pickle
from model import calBLEU, SemanticCommunicationSystem
from data_process import CorpusData
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Using ' + str(device).upper())

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

data_test_path = './dataset/test_data.pkl'
data_dict_path = './dataset/vocab.json'

corpus_data = CorpusData(data_test_path, data_dict_path)
max_sentence_length = corpus_data.max_sentence_length - 2
input_size = corpus_data.num_classes


# model_path = './trainedModel/deepSC_without_MI.pth'
# net = SemanticCommunicationSystem(input_size=input_size)
# net.load_state_dict(torch.load(model_path, map_location = device))
# net.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)


# dataloader = DataLoader(corpus_data, batch_size=batch_size, shuffle=False)


snr_range = np.arange(-10, 20, 4) # -10 to 20, step_size = 4
K_range = np.array([2, 4, 6, 10])
test_snr_range = np.arange(-10, 21, 2)


save_path = './trainedModel/'


snr_range = np.array([12])
K_range = np.array([8])

batch_size = 512

test_info = {}

for K in K_range:
    
    K_info = {}
    
    for snr in snr_range:
            
        model_name = 'deepSC_K_' + str(K) + '_snr_' + str(snr) + '.pth'
        
        model_path = os.path.join(save_path, model_name)
        
        net = SemanticCommunicationSystem(input_size=input_size, snr=snr, K=K)
        net.load_state_dict(torch.load(model_path, map_location = device))
        net.to(device)
        
        print('Current DeepSC model setting (K, snr) = ({}, {})'.format(K, snr))
            
        BLEU_1_SS_per_testSNR = []
        BLEU_2_SS_per_testSNR = []
        BLEU_3_SS_per_testSNR = []
        BLEU_4_SS_per_testSNR = []
        SS_per_testSNR = []
        
        for test_snr in test_snr_range:
            
            net.snr = test_snr
            print('Testing current DeepSC model with setting (K, snr) = ({}, {}) for test_snr = {}'.format(K, snr, net.snr))
            
            dataloader = DataLoader(corpus_data, batch_size=batch_size, shuffle=False)  
            
            BLEU_1_list = []
            BLEU_2_list = []
            BLEU_3_list = []
            BLEU_4_list = []
            semantic_similarity_list = []
            
            train_bar = tqdm(dataloader)
            
            for batch_idx, data in enumerate(train_bar):
                
                if batch_idx >= 4:
                    break
                
                inputs = np.zeros((batch_size, max_sentence_length))  # store every id of corresponding word inside the sentence into the matrix
                
                sentence_list = []
                tokenized_sentence_list = []
                sentence_length_list = []
                
                for i in range(batch_size):
                    sentence_idx = batch_idx * batch_size + i                    
                    sentence = corpus_data.get_sentence_as_word(index=sentence_idx)
                    tokenized_sentence = word_tokenize(sentence)
                    sentence_length = len(tokenized_sentence)
                    sentence_as_id = np.zeros(max_sentence_length)
                    
                    for j in range(sentence_length):
                        sentence_as_id[j] = corpus_data.word2id_dict[tokenized_sentence[j]]
                    
                    inputs[i, :] = sentence_as_id
                    
                    sentence_list.append(sentence)
                    tokenized_sentence_list.append(tokenized_sentence)
                    sentence_length_list.append(sentence_length)
            
                inputs = torch.LongTensor(inputs).to(device)
                label = F.one_hot(inputs, num_classes = input_size).float().to(device)
            
                s_predicted = net(inputs)
                s_predicted = torch.argmax(s_predicted, dim=2)
            
                for i in range(batch_size):
                    sentence = sentence_list[i]
                    tokenized_sentence = tokenized_sentence_list[i]
                    sentence_length = sentence_length_list[i]
                    
                    output_as_id = s_predicted[i, :]  # get the id list of most possible word
                    origin_sentence_as_id = inputs[i, :]
            
                    BLEU1 = calBLEU(1, output_as_id.cpu().detach().numpy(), origin_sentence_as_id.cpu().detach().numpy(), sentence_length)
                    BLEU_1_list.append(BLEU1)
                    
                    if sentence_length >= 2:
                        BLEU2 = calBLEU(2, output_as_id.cpu().detach().numpy(), origin_sentence_as_id.cpu().detach().numpy(), sentence_length)
                        BLEU_2_list.append(BLEU2)
                        
                        if sentence_length >= 3:
                            BLEU3 = calBLEU(3, output_as_id.cpu().detach().numpy(), origin_sentence_as_id.cpu().detach().numpy(), sentence_length)
                            BLEU_3_list.append(BLEU3)
                            
                            if sentence_length >= 4:
                                BLEU4 = calBLEU(4, output_as_id.cpu().detach().numpy(), origin_sentence_as_id.cpu().detach().numpy(), sentence_length)  # calculate BLEU
                                BLEU_4_list.append(BLEU4)
                    
                    output_sentence = corpus_data.convert_id_sentence_to_word(sentence_as_id=output_as_id, sentence_length=sentence_length)
                        
                    encoded_input = tokenizer(sentence, return_tensors='pt')  # encode sentence to fit bert model
                    bert_input = bert_model(**encoded_input).pooler_output  # get semantic meaning of the sentence
                    encoded_input = tokenizer(output_sentence, return_tensors='pt')
                    bert_output = bert_model(**encoded_input).pooler_output
                    semantic_similarity = torch.sum(bert_input * bert_output) / (torch.sqrt(torch.sum(bert_input * bert_input))
                                                                            * torch.sqrt(torch.sum(bert_output * bert_output)))
                    semantic_similarity_list.append(semantic_similarity.cpu().detach().numpy())
                            
            avg_BLEU_1 = np.mean(BLEU_1_list)
            avg_BLEU_2 = np.mean(BLEU_2_list)
            avg_BLEU_3 = np.mean(BLEU_3_list)
            avg_BLEU_4 = np.mean(BLEU_4_list)
            avg_SS = np.mean(semantic_similarity_list)
            
            BLEU_1_SS_per_testSNR.append(avg_BLEU_1)
            BLEU_2_SS_per_testSNR.append(avg_BLEU_2)
            BLEU_3_SS_per_testSNR.append(avg_BLEU_3)
            BLEU_4_SS_per_testSNR.append(avg_BLEU_4)  
            SS_per_testSNR.append(avg_SS)
            
            print('Finished testing current DeepSC model with setting (K, snr) = ({}, {}) for test_snr = {}'.format(K, snr, net.snr))
            print('BLEU 1 = {}'.format(avg_BLEU_1))
            print('BLEU 2 = {}'.format(avg_BLEU_2))
            print('BLEU 3 = {}'.format(avg_BLEU_3))
            print('BLEU 4 = {}'.format(avg_BLEU_4))
            print('Semantic Similarity = {}'.format(avg_SS))
            
        current_info = {}
        current_info['BLEU_1'] = BLEU_1_SS_per_testSNR
        current_info['BLEU_2'] = BLEU_2_SS_per_testSNR
        current_info['BLEU_3'] = BLEU_3_SS_per_testSNR
        current_info['BLEU_4'] = BLEU_4_SS_per_testSNR
        current_info['SS'] = SS_per_testSNR
        
        K_info[snr] = current_info
        
    test_info[K] = K_info
        
    x = test_snr_range
    y1 = BLEU_1_SS_per_testSNR
    y2 = BLEU_2_SS_per_testSNR
    y3 = BLEU_3_SS_per_testSNR
    y4 = BLEU_4_SS_per_testSNR
    y5 = SS_per_testSNR
    plt.figure(figsize=(6.4, 9.6))
    plt.suptitle("deepSC without MI")
    plt.subplot(2, 1, 1)
    plt.xlabel("SNR")
    plt.ylabel("BLEU")
    plt.plot(x, y1, marker='D', label='1-gram')
    plt.plot(x, y2, marker='D', label='2-gram')
    plt.plot(x, y3, marker='D', label='3-gram')
    plt.plot(x, y4, marker='D', label='4-gram')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.xlabel("SNR")
    plt.ylabel("Sentence Similarity")
    plt.plot(x, y5, marker='D')
    plt.show()
    
with open('./test_result/test_info.pkl', 'wb') as f:
    pickle.dump(test_info, f)
    
print("All done!")


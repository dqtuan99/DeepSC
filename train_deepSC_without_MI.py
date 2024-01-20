"""
it's used to train a semantic communication system without mutual information model
"""
import os
import numpy as np

from data_process import CorpusData
from model import SemanticCommunicationSystem, LossFn
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if not os.path.exists("./trainedModel"):
  os.makedirs("./trainedModel")

if not os.path.exists("./dataset"):
  os.makedirs("./dataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ' + str(device).upper())

batch_size = 256
num_epoch = 2
lr = 0.0005

data_train_path = './dataset/train_data.pkl'
data_dict_path = './dataset/vocab.json'

corpus_data = CorpusData(data_train_path, data_dict_path)

dataloader = DataLoader(corpus_data, batch_size=batch_size, shuffle=True)

input_size = corpus_data.num_classes

snr_range = np.arange(-10, 20, 4) # -10 to 20, step_size = 4
K_range = np.array([2, 4, 6, 10])


save_path = './trainedModel/'

lossFn = LossFn()

for K in K_range:
  for snr in snr_range:

    net = SemanticCommunicationSystem(input_size=input_size, snr=snr, K=K).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    print("Current setting (K, snr) = ({}, {})".format(K, snr))

    for epoch in range(num_epoch):
        train_bar = tqdm(dataloader)
        for i, data in enumerate(train_bar):
            [inputs, sentence_length] = data  # get length of sentence without padding
            num_sample = inputs.size()[0]  # get how much sentence the system get
            inputs = inputs.long().to(device)

            label = F.one_hot(inputs, num_classes=input_size).float().to(device)

            s_predicted = net(inputs)

            loss = lossFn(s_predicted, label, sentence_length, num_sample, batch_size)
            loss.backward()
            optim.step()
            optim.zero_grad()

            print('  loss: ', loss.cpu().detach().numpy())

    model_name = 'deepSC_K_' + str(K) + '_snr_' + str(snr) + '.pth'
    model_path = os.path.join(save_path, model_name)

    print("Saving model with current setting (K, snr) = ({}, {})".format(K, snr))
    torch.save(net.state_dict(), model_path)
    
print("All done!")
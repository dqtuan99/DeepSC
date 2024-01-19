"""
it's used to train a model guided by deepSC and mutual information system
attention that it won't modify mutual info model, only deepSC's improvement will be stored
"""

import torch
from torch.utils.data import DataLoader
import modelModifiedForMI
from tqdm import tqdm
from data_process import CorpusData
import torch.nn.functional as F


batch_size = 256
num_epoch = 2
lamda = 0.05  # it's used to control how much the muInfo will affect deepSC model
save_path = './trainedModel/deepSC_with_MI.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ' + str(device).upper())

#==============================================================================

train_path = './dataset/train_data.pkl'
dict_path = './dataset/vocab.json'

corpus_data = CorpusData(train_path, dict_path)

dataloader = DataLoader(corpus_data, batch_size=batch_size, shuffle=True)

input_size = corpus_data.num_wordID

scNet = modelModifiedForMI.SemanticCommunicationSystem(input_size=input_size).to(device)
muInfoNet = modelModifiedForMI.MutualInfoSystem().to(device)

load_pretrained = False

if load_pretrained:
    deepSC_path = './trainedModel/deepSC_without_MI.pth'
    muInfo_path = './trainedModel/MutualInfoSystem.pth'
    scNet.load_state_dict(torch.load(deepSC_path, map_location=device))
    muInfoNet.load_state_dict(torch.load(muInfo_path, map_location=device))    

#==============================================================================

# dataloader = DataLoader(CorpusData(), batch_size= batch_size, shuffle=True)
# scNet = modelModifiedForMI.SemanticCommunicationSystem()
# scNet.load_state_dict(torch.load(deepSC_path, map_location=device))
# scNet.to(device)
# muInfoNet = modelModifiedForMI.MutualInfoSystem()
# muInfoNet.load_state_dict(torch.load(muInfo_path, map_location=device))
# muInfoNet.to(device)

optim = torch.optim.Adam(scNet.parameters(), lr=0.0005)
lossFn = modelModifiedForMI.LossFn()

for epoch in range(num_epoch):
    train_bar = tqdm(dataloader)
    for i, data in enumerate(train_bar):
        [inputs, sentence_length] = data  # get length of sentence without padding
        num_sample = inputs.size()[0]  # get how much sentence the system get
        # inputs = inputs[:, 0, :].clone().detach().requires_grad_(True).long()  # .long used to convert the tensor to long format
        # in order to fit one_hot function
        inputs = inputs.long().to(device)
        # inputs = inputs.to(device)

        label = F.one_hot(inputs, num_classes=input_size).float().to(device)
        # label = label.to(device)

        [s_predicted, codeSent, codeWithNoise] = scNet(inputs)

        x = torch.reshape(codeSent, (-1, 16))  # get intermediate variables to train mutual info sys
        y = torch.reshape(codeWithNoise, (-1, 16))

        batch_joint = modelModifiedForMI.sample_batch(5, 'joint', x, y).to(device)
        batch_marginal = modelModifiedForMI.sample_batch(5, 'marginal', x, y).to(device)

        t = muInfoNet(batch_joint)
        et = torch.exp(muInfoNet(batch_marginal))
        MI_loss = torch.mean(t) - torch.log(torch.mean(et))
        SC_loss = lossFn(s_predicted, label, sentence_length, num_sample, batch_size)

        loss = SC_loss + torch.exp(-MI_loss) * lamda
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        print()
        print("Total Loss: {}, Mutual Loss: {}, SC Loss: {}".format(loss.cpu().detach().numpy(),
                                                                    -MI_loss.cpu().detach().numpy(),
                                                                    SC_loss.cpu().detach().numpy()))
        break
torch.save(scNet.state_dict(), save_path)
print("All done!")
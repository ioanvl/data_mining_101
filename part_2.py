import pandas as pd
import numpy as np
import nltk
import math
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

#pytorch dataset object gia training/testing - dhmiourgei ta samples
class onion_words(Dataset):
    def __init__(self, dataframe, max_len):
        self.data = dataframe
        self.total = len(dataframe)
        self.max_len = max_len
    def __len__(self):
        return self.total
    def __getitem__(self, ind):
        label = torch.FloatTensor(1).zero_()
        if self.data.iloc[ind]['label'] == 1:
            label += 1
        x = torch.FloatTensor(self.data.iloc[ind]['tf_idf'])
        if int(x.size()[0]) > self.max_len:
            x = torch.narrow(x, 0, 0, self.max_len)
        else:
            s_pad = int(self.max_len - x.size())
            x = F.pad(x, (0, s_pad))
        return x.view(1, -1), label


class Mish(nn.Module):
    # a self regulating activation function
    # credits:
    # Diganta Misra - https://arxiv.org/ftp/arxiv/papers/1908/1908.08681.pdf
    # Less Wright - https://github.com/lessw2020/mish
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

#pytorch network object
class onion_net(nn.Module):
    def __init__(self, max_len, num_channels=24, depth=4, kernel_s=3, num_linear=64):
        super().__init__()
        self.max_len = max_len
        self.num_ch = num_channels
        
        layers = []
        norms = []
        self.initial_conv = nn.Conv1d(1, self.num_ch, 1)
        
        for i in range(depth):
            layer = nn.Conv1d(self.num_ch, self.num_ch,kernel_s)
            layers.append(layer)
            
            norm = nn.BatchNorm1d(self.num_ch)
            norms.append(norm)
            
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        
        self._to_linear = 0
        #mock pass me kapoio random sample gia na doume to mege8os tou sample meta ta comvolutions
        #mias kai to pytorch den exei kapoia volikh 'flatten' function
        #kai prepei na 3eroume to input size gia ena linear layer
        x_temp = torch.randn(self.max_len * self.num_ch).view(-1, self.num_ch, self.max_len)
        x_temp = self.conv_forward(x_temp)
        self._to_linear = x_temp[0].shape[0] * x_temp[0].shape[1]
        
        self.activ_0 = Mish()
        self.fc_1 = nn.Linear(self._to_linear, num_linear)
        self.activ_1 = Mish()
        self.drop = nn.Dropout(0.2)
        self.fc_out = nn.Linear(num_linear, 1)
        self.sig = nn.Sigmoid()
        print(f"Network initialised,\n{depth} layers depth with {self.num_ch} channels\nKernel size: {kernel_s}")
        
        
    def conv_forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
        return x
    
    def forward(self,x):
        x = self.initial_conv(x)
        x = self.conv_forward(x)
        x = x.view(-1, self._to_linear)
        
        x = self.activ_0(x)
        x = self.activ_1(self.fc_1(x))
        x = self.drop(x)
        x = self.fc_out(x)
        
        return self.sig(x)


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)




if __name__ == "__main__":

    nltk.download('punkt')
    nltk.download('stopwords')

    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"

    print("\n\nRaw data:")
    print("==============================")
    onot = pd.read_csv("onion-or-not.csv")
    print(onot)

    def prep_stent(sentence):    
        tokens = word_tokenize(sentence)
        
        stops = set(stopwords.words('english'))
        clean_tokens = [x for x in tokens if ((x not in stops) and (x not in symbols))]
        
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(x) for x in clean_tokens]
        return stemmed_tokens


    print("\n\nStemming and tokenization/cleanup")
    print("==============================")
    onot['prep_text'] = onot['text'].apply(prep_stent)
    print(onot['prep_text'])



    local_corpus = {}
    def tf_and_learn_df(tokens):
        uniques, count = np.unique(tokens, return_counts=True)
        u_dict = dict(zip(uniques, count))
        for x in uniques:
            if x not in local_corpus:
                local_corpus.update({x: 1})
            else:
                local_corpus[x] += 1
        tf_s = [(u_dict[x] / len(tokens)) for x in tokens]
        return tf_s

    print("\n\nApplying tf and recording corpus for idf")
    print("==============================")
    onot['tf'] = onot['prep_text'].apply(tf_and_learn_df)
    print(onot['tf'])

    data_len = len(onot)

    def apply_tf_idf(row):
        idf = [math.log(data_len / (local_corpus[x] + 1)) for x in row['prep_text']]
        tf_idf = [(x * y) for x,y in zip(row['tf'], idf)]
        return tf_idf

    print("\n\nApplying tf-idf")
    print("==============================")
    onot['tf_idf'] = onot.apply(apply_tf_idf, axis=1)
    print(onot['tf_idf'])


    def get_len(tokens):
        return len(tokens)

    print("\n\nRecording sentece length")
    print("==============================")
    onot['len'] = onot['tf_idf'].apply(get_len)
    print(onot['len'])



    temp_len = onot['len'].sort_values()
    temp_ind = int(np.floor(len(temp_len) * 0.9))
    max_len = temp_len.iloc[temp_ind]
    print(f"\n---\nTop 90% sent.length: {max_len}\nWe'll use that to pad/cut the setences accordingly")





    print("---\nSplitting dataset 75-25")
    ran_seed = (int(round(time.time()))) % 5000
    np.random.seed(ran_seed)

    index = onot.index.values
    np.random.shuffle(index)

    cut = int(np.floor(len(onot) * 0.25))
    train_index, test_index = index[cut:], index[:cut]
    train = onot.iloc[train_index]
    test = onot.iloc[test_index]


    # default / init values
    learning_rate = 1e-3
    batch_size = 64
    scale = 1
    lr = learning_rate * scale
    bat_s = batch_size * scale
    epochs = 20


    # datasets
    train_set = onion_words(train[['tf_idf', 'label']], max_len)
    test_set = onion_words(test[['tf_idf', 'label']], max_len)

    #loaders for batching (also shuffling but we already took care of that)
    train_loader = DataLoader(train_set, batch_size=bat_s)
    test_loader = DataLoader(test_set, batch_size=bat_s)

    #iterator for intermitent testing each eapoch
    test_iter = iter(test_loader)


    channels = 24
    depth = 5
    kernel = 3

    print(f"\n=======\n\nlr: {learning_rate}\nBatch size: {batch_size}\n{epochs} Epochs\n---")
    network = onion_net(max_len, num_channels=channels, depth=depth, kernel_s=kernel)
    network.apply(weights_init)



    loss_f = nn.BCELoss()
    optim = Adam(network.parameters(), lr=lr)
    sched = ReduceLROnPlateau(optim, mode='max', factor=0.1, patience=max((epochs // 6), 1), verbose=True)
    target_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    # training loop
    print("\n-------------------------\nTraining..")
    network.to(target_dev)
    for i in range(epochs):
        network.train()
        print(f"Epoch: {i + 1}")
        for data in tqdm(train_loader, desc="Train: "):
            sample, label = data
            sample, label = sample.to(target_dev), label.to(target_dev)
            
            optim.zero_grad()
            output = network(sample)
            loss = loss_f(output, label)
            loss.backward()
            optim.step()
        
        inter_matches = []
        for _ in tqdm(range(len(test_loader) // 10), desc="test"):
            try:
                sample, label = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                sample, label = next(test_iter)
            sample, label = sample.to(target_dev), label.to(target_dev)
            
            output = network(sample)
            inter_matches += [((float(i) >= 0.5 and float(j) == 1) or (float(i) < 0.5 and float(j) == 0)) for i,j, in zip(output, label)]
        
        temp_acc = inter_matches.count(True) / len(inter_matches)
        sched.step(temp_acc)
        print(f"done.  =  Acc: {round(temp_acc, 2)}\n")

    print("\n------------------------------\n")
    print("Complete validation on unseen data..")
    results = pd.DataFrame(columns=['pred', 'truth'])
    network.eval()
    network.to(target_dev)
    for data in tqdm(test_loader, desc="Validation: "):
        sample, label = data
        sample, label = sample.to(target_dev), label.to(target_dev)
        
        output = network(sample)
        matches = [[float(i), float(j)] for i,j, in zip(output, label)]
        
        temp_res = pd.DataFrame(matches, columns=list(results.columns.values))
        results = results.append(temp_res, ignore_index=True)
        
    print("Pass finished")

    acc = len(results[(((results['pred'] >= 0.5) & (results['truth'] == 1)) | ((results['pred'] < 0.5) & (results['truth'] == 0)))]) / len(results)
    precision = len(results[((results['pred'] >= 0.5) & (results['truth'] ==1))]) / len(results[results['pred'] >= 0.5])
    recall = len(results[((results['pred'] >= 0.5) & (results['truth'] ==1))]) / len(results[results['truth'] == 1])
    f_one = ((precision * recall) / (precision + recall)) * 2

    print("\n==========================\n")
    print(f"Accuracy: {round(acc * 100, 2)}%")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1: {round(f_one, 4)}")






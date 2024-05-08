#!/usr/bin/env python
# coding: utf-8

# # import packages

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import gzip
import shutil
import os


# # data processing

# In[2]:


with gzip.open('glove.6B.100d.gz', 'rb') as f_in:
    with open('glove.6B.100d', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[3]:


def word_type(word:str):
    if word.isdigit(): #Is a digit
        return "unk_num"
    elif word.islower(): 
        return "unk_all_lower"    
    elif word.isupper(): 
        return "unk_all_upper"              
    elif any(char.isdigit() for char in word):
        return "unk_contain_num"    
    else:
        return "unk"


# In[4]:


def read_in_data(filename):
    vocab = dict()
    sentences =[] # generating original output
    tags = set()
    tmp_sentence = []
    targets = []
    tmp_target = []
    with open(filename, "r") as file:
        for line in file.readlines():
            if len(line) > 1:
                _, word, tag = line.strip().split(" ")
                if word not in vocab.keys():
                    vocab[word] = 1
                else:
                    vocab[word] += 1
                tags.add(tag)
                tmp_sentence.append(word)
                tmp_target.append(tag)
            else:
                sentences.append(tmp_sentence)
                targets.append(tmp_target)
                tmp_sentence = []
                tmp_target = []
        if len(tmp_sentence)>0:
            sentences.append(tmp_sentence)
            targets.append(tmp_target)
    return vocab, sentences, tags, targets

def remove_low_frequency_word(vocab, occurences):
    candidates = set()
    for word in vocab.keys():
        if vocab[word] >= occurences:
            if any(w.isdigit() for w in word):
                candidates.add(word_type(word))
            else:
                candidates.add(word)
        else:
            candidates.add(word_type(word))
    return candidates


# In[5]:


vocab_OCC, sentences, tags, targets = read_in_data("./data/train")


# In[6]:


vocab = remove_low_frequency_word(vocab_OCC, occurences = 2)


# In[7]:


def get_idx_vocab_tags(vocab, tags):
    word_to_idx = {}
    tag_to_idx = {}
    # word_to_idx["unk"] = 1
    word_to_idx["PAD"] = 0
    start_word = 1
    for word in vocab:
        word_to_idx[word] = start_word
        start_word += 1
    start_tag = 0
    for tag in tags:
        tag_to_idx[tag] = start_tag
        start_tag += 1
    return word_to_idx, tag_to_idx


# In[8]:


word_to_idx, tag_to_idx = get_idx_vocab_tags(vocab, tags)


# In[9]:


len(word_to_idx)


# In[10]:


print(word_to_idx)


# In[11]:


def get_vocab_tags_idx(word_to_idx, tag_to_idx):
    idx_to_word, idx_to_tag = {}, {}
    for word, idx in word_to_idx.items():
        idx_to_word[idx] = word
    for tag, idx in tag_to_idx.items():
        idx_to_tag[idx] = tag
    return idx_to_word, idx_to_tag


# In[12]:


idx_to_word, idx_to_tag = get_vocab_tags_idx(word_to_idx, tag_to_idx)


# In[13]:


def process_sentences(sentences:list, word_to_idx:dict):
    data = []
    for sentence in sentences:
        tmp_sentence = []
        for word in sentence:
            if word in word_to_idx.keys():
                tmp_sentence.append(word_to_idx[word])
            else:
                tmp_sentence.append(word_to_idx[word_type(word)])
        data.append(tmp_sentence)
    return data

def process_targets(targets:list, tag_to_idx:dict):
    data = []
    for target in targets:
        tmp_target = []
        for tag in target:
            tmp_target.append(tag_to_idx[tag])
        data.append(tmp_target)
    return data
    


# In[14]:


data_X = process_sentences(sentences, word_to_idx)


# In[15]:


data_y = process_targets(targets, tag_to_idx)


# In[16]:


# get the maximum length of sentence in train
maximum_length = 0
for sample in data_X:
    if len(sample) > maximum_length:
        maximum_length = len(sample)


# In[17]:


maximum_length


# In[18]:


def padding_sentence(data: list, maximum_length=125):
    for i in range(len(data)):
        if len(data[i]) > maximum_length:  # Truncating
            data[i] = data[i][:maximum_length]
        elif len(data[i]) < maximum_length:  # Padding 
            data[i] = data[i] + [0] * (maximum_length - len(data[i]))
    return data

def padding_tags(data: list, maximum_length=125):
    for i in range(len(data)):
        if len(data[i]) > maximum_length:  # Truncating
            data[i] = data[i][:maximum_length]
        elif len(data[i]) < maximum_length:  # Padding with penalty score
            data[i] = data[i] + [-100] * (maximum_length - len(data[i]))
    return data


# In[19]:


data_X = padding_sentence(data_X)
data_y = padding_tags(data_y)


# In[20]:


X_train = torch.LongTensor(data_X)
Y_train = torch.LongTensor(data_y)


# In[21]:


print(X_train)


# In[22]:


ds_train = TensorDataset(X_train, Y_train)
loader_train = DataLoader(ds_train, batch_size=16, shuffle=False)


# In[23]:


embedding_weight = dict()
f = open(os.path.join('glove.6B.100d'), encoding='utf-8')
for line in f:
    word_vector = line.split()
    word = word_vector[0]
    word_emb = np.asarray(word_vector[1:], dtype='float32') 
    embedding_weight[word] = word_emb

embedding_dim = 100
embedding_input = np.zeros((len(word_to_idx), embedding_dim))


# In[24]:


for word, idx in word_to_idx.items():
    embedding_vec = embedding_weight.get(word.lower())
    if embedding_vec is not None:
        embedding_input[idx] = embedding_vec


# In[29]:


print(embedding_input)


# In[30]:


#embedding_matrix_model = torch.LongTensor(embedding_matrix)
embedding_matrix_model = torch.Tensor(embedding_input)


# # model structure define

# In[31]:


class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, target_size, embedding_dim=100, lstm_hidden_dim=256, lstm_layers=1, lstm_dropout=0.33, linear_dim=128):
        super().__init__()        
        self.dropout = nn.Dropout(0.33)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim,num_layers=lstm_layers,batch_first=True, bidirectional=True) #dropout=lstm_dropout)
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, target_size)

    def forward(self, sentence):
        embeds = self.dropout(self.embedding(sentence))
        lstm_out,_ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        output = self.classifier(elu_out)
        return output


# In[32]:


model = BiLSTMNER(vocab_size = len(word_to_idx), target_size = len(tag_to_idx))


# In[33]:


def read_in_dev_data(filename):
    sentences = []
    tmp_sentence = []
    targets = []
    tmp_target = []
    with open(filename) as file:
        for line in file.readlines():
            if len(line) > 1:
                _, word, tag = line.strip().split(" ")
                tmp_sentence.append(word)
                tmp_target.append(tag)
            else:
                sentences.append(tmp_sentence)
                targets.append(tmp_target)
                tmp_sentence = []
                tmp_target = []
        if len(tmp_sentence)>0:
            sentences.append(tmp_sentence)
            targets.append(tmp_target)
    return sentences, targets


# # dev dataset

# In[34]:


dev_sentences, dev_targets = read_in_dev_data("./data/dev")


# In[35]:


data_X_dev = process_sentences(dev_sentences, word_to_idx)
data_y_dev = process_targets(dev_targets, tag_to_idx)


# In[36]:


data_X_dev = padding_sentence(data_X_dev)
data_y_dev = padding_tags(data_y_dev)


# In[37]:


X_dev = torch.LongTensor(data_X_dev)
Y_dev = torch.LongTensor(data_y_dev)
ds_dev = TensorDataset(X_dev, Y_dev)
loader_dev = DataLoader(ds_dev, batch_size=16, shuffle=False)


# # test dataset

# In[38]:


def read_in_test_data(filename):
    sentences = []
    tmp_sentence = []
    with open(filename) as file:
        for line in file.readlines():
            if len(line) > 1:
                _, word = line.strip().split(" ")
                tmp_sentence.append(word)
            else:
                sentences.append(tmp_sentence)
                tmp_sentence = []
        if len(tmp_sentence)>0:
            sentences.append(tmp_sentence)
    return sentences

test_sentences = read_in_test_data("./data/test")


# In[39]:


data_test = process_sentences(test_sentences, word_to_idx)


# In[40]:


data_test = process_sentences(test_sentences, word_to_idx)
data_X_test = padding_sentence(data_test)
X_test = torch.LongTensor(data_X_test)
loader_test = DataLoader(X_test, batch_size=16, shuffle=False)


# # Using GPU

# In[41]:


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU mode")
else:
    device = torch.device("cpu")
    print("CPU mode")


# # train and evaluate model

# In[42]:


def train_evaluate(model, train_data, dev_data, epoch_num=50, tag_pad_idx = -100):
    optimizer = optim.SGD(model.parameters(), lr=0.23, momentum=0.9, nesterov=True) # Set hyperparameter
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)
    criterion = nn.CrossEntropyLoss(ignore_index= -100)
    best_loss = float('inf')
    predict_table = []
    for epoch in range(epoch_num):
        # training
        train_result = []
        train_loss = 0
        train_acc = 0
        train_total = 0
        model.train()
        for train_sentence, train_target in train_data:
            optimizer.zero_grad()
            train_sentence = train_sentence.to(device)
            train_target = train_target.to(device)
            train_pred = model(train_sentence)
            train_pred = train_pred.view(-1, train_pred.shape[-1])
            train_target = train_target.view(-1)
            train_tmp_loss = criterion(train_pred, train_target)

            train_tmp_total = 0
            train_tmp_correct = 0
            max_pred = train_pred.argmax(dim=1, keepdim = True)
            for tmp_pred, tmp_target, tmp_word in zip(max_pred, train_target, train_sentence.view(-1)):
                if tmp_word != 0:
                    train_result.append((tmp_word.item(), tmp_pred.item(), tmp_target.item()))
                    if tmp_target.item() == tmp_pred.item():
                        train_tmp_correct += 1
                    train_total += 1
            train_tmp_loss.backward()
            optimizer.step()
            train_loss += train_tmp_loss.item()
            train_acc += train_tmp_correct
            train_total += train_tmp_total
        print('Epoch ', epoch, ' :')
        print(f'\tTrain Loss: {train_loss/len(train_data):.6f} | Train Acc: {(train_acc/train_total)*100:.2f}%')

        # evaluating
        dev_result = []
        dev_loss = 0
        dev_acc = 0
        dev_total = 0
        model.eval()
        with torch.no_grad():
            for dev_sentence, dev_target in dev_data:
                dev_sentence = dev_sentence.to(device)
                dev_target = dev_target.to(device)
                dev_pred = model(dev_sentence)
                dev_pred = dev_pred.view(-1, dev_pred.shape[-1])
                dev_target = dev_target.view(-1)
                dev_tmp_loss = criterion(dev_pred, dev_target)

                dev_tmp_total = 0
                dev_tmp_correct = 0
                max_pred = dev_pred.argmax(dim=1, keepdim=True)
                for tmp_pred, tmp_target, tmp_word in zip(max_pred, dev_target, dev_sentence.view(-1)):
                    if tmp_word != 0:
                        dev_result.append((tmp_word.item(), tmp_pred.item(), tmp_target.item()))
                        if tmp_target.item() == tmp_pred.item():
                            dev_tmp_correct += 1
                        dev_total += 1
                dev_loss += dev_tmp_loss.item()
                dev_acc += dev_tmp_correct
                dev_total += dev_tmp_total
        print(f'\tDev Loss: {dev_loss / len(dev_data):.6f} | Dev Acc: {(dev_acc / dev_total) * 100:.2f}%')
        if dev_loss <= best_loss:
            best_loss = dev_loss
            predict_table = dev_result
            torch.save(model.state_dict(), './model/blstm2.pt')
    return predict_table


# In[43]:


model.to(device)


# In[44]:


model.embedding.weight.data.copy_(embedding_matrix_model)


# In[45]:


predict_table = train_evaluate(model,loader_train,loader_dev)


# In[46]:


y_pred = [int(x[1]) for x in predict_table]
i=0
new_file = open('./dev2.out', "w")
with open('./data/dev', "r") as file:
    for line in file:
        if len(line) > 1:
            idx, word, tag = line.strip().split(" ")
            new_file.write(str(idx)+' '+str(word)+' '+str(idx_to_tag[y_pred[i]])+'\n')
            i+=1
        else:
            new_file.write('\n')
file.close()
new_file.close()


# # test predict

# In[47]:


output = []
model.to(device)
model.eval()
for test_sentence in loader_test:
    test_sentence = test_sentence.to(device)
    test_pred = model(test_sentence)
    test_pred = test_pred.view(-1, test_pred.shape[-1])
    max_pred = test_pred.argmax(dim=1, keepdim = True)
    for tmp_pred, word in zip(max_pred, test_sentence.view(-1)):
        if word != 0:
            output.append(tmp_pred.item())


# In[48]:


length = 0
with open('./data/test', "r") as file:
    for line in file:
        if len(line) > 1: 
            length+=1
file.close()
print(length)
print(len(output))


# In[49]:


i=0
new_file = open('./test2.out', "w")
with open('./data/test', "r") as file:
    for line in file:
        if len(line) > 1:
            idx, word = line.strip().split(" ")
            new_file.write(str(idx)+' '+str(word)+' '+str(idx_to_tag[output[i]])+'\n')
            i+=1
        else:
            new_file.write('\n')
file.close()
new_file.close()


# In[ ]:





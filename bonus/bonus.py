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


char_vocab = set()
for word in vocab:
    for char in word:
        if char not in char_vocab:
            char_vocab.add(char)


# In[8]:


char_to_idx = {}
i = 0
for char in char_vocab:
    char_to_idx[char] = i
    i += 1


# In[9]:


char_to_idx


# In[10]:


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


# In[11]:


word_to_idx, tag_to_idx = get_idx_vocab_tags(vocab, tags)


# In[12]:


len(word_to_idx)


# In[13]:


print(word_to_idx)


# In[14]:


def get_vocab_tags_idx(word_to_idx, tag_to_idx):
    idx_to_word, idx_to_tag = {}, {}
    for word, idx in word_to_idx.items():
        idx_to_word[idx] = word
    for tag, idx in tag_to_idx.items():
        idx_to_tag[idx] = tag
    return idx_to_word, idx_to_tag


# In[15]:


idx_to_word, idx_to_tag = get_vocab_tags_idx(word_to_idx, tag_to_idx)


# In[16]:


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
    
def process_sentences_with_char(sentences:list, word_to_idx:dict, char_to_idx:dict):
    data = []
    char_res = []
    for sentence in sentences:
        tmp_sentence = []
        tmp_word = []
        for word in sentence:
            tmp_char = []
            if word in word_to_idx.keys():
                tmp_sentence.append(word_to_idx[word])
                for char in word:
                    tmp_char.append(char_to_idx[char])
            else:
                tmp_sentence.append(word_to_idx[word_type(word)])
                for char in word_type(word):
                    tmp_char.append(char_to_idx[char])
            tmp_word.append(tmp_char)
        data.append(tmp_sentence)
        char_res.append(tmp_word)
    return data, char_res


# In[17]:


data_X, data_X_char = process_sentences_with_char(sentences, word_to_idx, char_to_idx)


# In[18]:


data_X_char


# In[19]:


data_y = process_targets(targets, tag_to_idx)


# In[20]:


# get the maximum length of sentence in train
maximum_length = 0
for sample in data_X:
    if len(sample) > maximum_length:
        maximum_length = len(sample)


# In[21]:


maximum_length


# In[22]:


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


# In[23]:


def padding_char(data:list, maximum_length = 10, max_str_length = 125):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if len(data[i][j]) > maximum_length:  # Truncating
                data[i][j] = data[i][j][:maximum_length]
            elif len(data[i][j]) < maximum_length:  # Padding 
                data[i][j] = data[i][j] + [0] * (maximum_length - len(data[i][j]))
        
        if len(data[i]) > max_str_length:
            data[i] = data[i][:max_str_length]
        elif len(data[i]) < max_str_length:
            for _ in range(max_str_length - len(data[i])):
                data[i].append([[0] * maximum_length][0])
    return data


# In[24]:


data_X_char = padding_char(data_X_char)


# In[25]:


len(data_X_char)


# In[26]:


len(data_X)


# In[27]:


data_X = padding_sentence(data_X)
data_y = padding_tags(data_y)


# In[28]:


X_train = torch.LongTensor(data_X)
Y_train = torch.LongTensor(data_y)


# In[29]:


X_train_char = torch.LongTensor(data_X_char)


# In[30]:


print(X_train)


# In[31]:


ds_train = TensorDataset(X_train, Y_train, X_train_char)
loader_train = DataLoader(ds_train, batch_size=16, shuffle=False)


# In[32]:


embedding_weight = dict()
f = open(os.path.join('glove.6B.100d'), encoding='utf-8')
for line in f:
    word_vector = line.split()
    word = word_vector[0]
    word_emb = np.asarray(word_vector[1:], dtype='float32') 
    embedding_weight[word] = word_emb

embedding_dim = 100
embedding_input = np.zeros((len(word_to_idx), embedding_dim))


# In[33]:


for word, idx in word_to_idx.items():
    embedding_vec = embedding_weight.get(word.lower())
    if embedding_vec is not None:
        embedding_input[idx] = embedding_vec


# In[34]:


print(embedding_input)


# In[35]:


#embedding_matrix_model = torch.LongTensor(embedding_matrix)
embedding_matrix_model = torch.Tensor(embedding_input)


# # model structure define

# In[36]:


# class BiLSTMNER(nn.Module):
#     def __init__(self, vocab_size, target_size, embedding_dim=100, lstm_hidden_dim=256, lstm_layers=1, lstm_dropout=0.33, linear_dim=128):
#         super().__init__()        
#         self.dropout = nn.Dropout(0.33)
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim,num_layers=lstm_layers,batch_first=True, bidirectional=True) #dropout=lstm_dropout)
#         self.linear = nn.Linear(lstm_hidden_dim * 2, linear_dim)
#         self.elu = nn.ELU()
#         self.classifier = nn.Linear(linear_dim, target_size)

#     def forward(self, sentence):
#         embeds = self.dropout(self.embedding(sentence))
#         lstm_out,_ = self.lstm(embeds)
#         lstm_out = self.dropout(lstm_out)
#         linear_out = self.linear(lstm_out)
#         elu_out = self.elu(linear_out)
#         output = self.classifier(elu_out)
#         return output
import torch.nn.functional as F
class BiLSTMNERwithCNN(nn.Module):
    def __init__(self, vocab_size, target_size, embedding_dim=100, lstm_hidden_dim=256, lstm_layers=1, lstm_dropout=0.33, linear_dim=128):
        super().__init__()        
        self.dropout = nn.Dropout(0.33)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim, 30)
        self.lstm = nn.LSTM(130, lstm_hidden_dim,num_layers=lstm_layers,batch_first=True, bidirectional=True) #dropout=lstm_dropout)
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, target_size)
        self.conv1d = nn.Conv1d(in_channels=100, out_channels=30, kernel_size=3, padding=1)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        
    def forward(self, sentence):
        embeds = self.dropout(self.embedding(sentence))
        batch_size, sequence_length, embedding_dim = embeds.size()
        
        cnn_out = self.conv1d(embeds.permute(0, 2, 1))
        pool_out = self.maxpool(cnn_out)
        embeds_2 = pool_out.permute(0, 2, 1)
        embeds_2 = embeds_2.repeat(1, 125, 1)
        concatenated_tensor = torch.cat((embeds, embeds_2), dim=2)
        lstm_out, _ = self.lstm(concatenated_tensor)
        lstm_out = self.dropout(lstm_out)
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        output = self.classifier(elu_out)
        return output

class CharacterCNN(nn.Module):
    def __init__(self, input_dim, output_dim, max_char_length, num_filters=100, kernel_sizes=(3, 4, 5)):
        super(CharacterCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (kernel_size, input_dim)) for kernel_size in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.max_char_length = max_char_length

    def forward(self, x):
        # x: (batch_size, sequence_length, max_char_length, input_dim)
        batch_size = x.size(0)
        x = x.view(-1, 1, self.max_char_length, x.size(3))  # reshape for Conv2d
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # conv and relu
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]  # max pooling
        x = torch.cat(x, 1)  # concatenate all pooled features
        x = x.view(batch_size, -1, x.size(1))  # reshape to original size
        return x

class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, target_size, char_vocab_size, char_embedding_dim=30, char_hidden_dim=30, embedding_dim=100, lstm_hidden_dim=256, lstm_layers=1, lstm_dropout=0.33, linear_dim=128):
        super().__init__()        
        self.dropout = nn.Dropout(0.33)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.char_cnn = CharacterCNN(input_dim=char_embedding_dim, output_dim=char_hidden_dim, max_char_length=10)
        self.lstm = nn.LSTM(400, lstm_hidden_dim,num_layers=lstm_layers,batch_first=True, bidirectional=True) #dropout=lstm_dropout)
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, target_size)

    def forward(self, words, chars):
        word_embeds = self.dropout(self.embedding(words))
        char_embeds = self.dropout(self.char_embedding(chars))
        char_cnn_out = self.char_cnn(char_embeds)
        word_char_embeds = torch.cat((word_embeds, char_cnn_out), dim=2)
        
        lstm_out,_ = self.lstm(word_char_embeds)
        lstm_out = self.dropout(lstm_out)
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        output = self.classifier(elu_out)
        return output


# In[37]:


model = BiLSTMNER(vocab_size = len(word_to_idx),char_vocab_size=len(char_to_idx), target_size = len(tag_to_idx))


# In[38]:


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

# In[39]:


dev_sentences, dev_targets = read_in_dev_data("./data/dev")


# In[40]:


data_X_dev,data_X_dev_char = process_sentences_with_char(dev_sentences, word_to_idx, char_to_idx)
data_y_dev = process_targets(dev_targets, tag_to_idx)


# In[41]:


data_X_dev = padding_sentence(data_X_dev)
data_y_dev = padding_tags(data_y_dev)


# In[42]:


data_X_dev_char = padding_char(data_X_dev_char)


# In[43]:


X_dev = torch.LongTensor(data_X_dev)
Y_dev = torch.LongTensor(data_y_dev)
Y_dev_char = torch.LongTensor(data_X_dev_char)
ds_dev = TensorDataset(X_dev, Y_dev, Y_dev_char)
loader_dev = DataLoader(ds_dev, batch_size=16, shuffle=False)


# # test dataset

# In[44]:


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


# In[48]:


data_test, data_test_char = process_sentences_with_char(test_sentences, word_to_idx, char_to_idx)
data_X_test = padding_sentence(data_test)
data_X_test_char = padding_char(data_test_char)
X_test = torch.LongTensor(data_X_test)
X_test_char = torch.LongTensor(data_X_test_char)
test_dev = TensorDataset(X_test, X_test_char)
loader_test = DataLoader(test_dev, batch_size=16, shuffle=False)


# # Using GPU

# In[49]:


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU mode")
else:
    device = torch.device("cpu")
    print("CPU mode")


# # train and evaluate model

# In[50]:


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
        for train_sentence, train_target, train_char in train_data:
            optimizer.zero_grad()
            train_sentence = train_sentence.to(device)
            train_target = train_target.to(device)
            train_char = train_char.to(device)
            train_pred = model(train_sentence,train_char)
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
            for dev_sentence, dev_target, dev_char in dev_data:
                dev_sentence = dev_sentence.to(device)
                dev_target = dev_target.to(device)
                dev_char = dev_char.to(device)
                dev_pred = model(dev_sentence,dev_char)
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
#         if dev_loss <= best_loss:
#             best_loss = dev_loss
#             predict_table = dev_result
#             torch.save(model.state_dict(), './model/bonus.pt')
        if dev_acc >= best_acc: # using acc as index maybe better?
            best_acc = dev_acc
            predict_table = dev_result
            torch.save(model.state_dict(), './model/bonus.pt')
    return predict_table


# In[51]:


model.to(device)


# In[52]:


model.embedding.weight.data.copy_(embedding_matrix_model)


# In[53]:


predict_table = train_evaluate(model,loader_train,loader_dev)


# In[54]:


y_pred = [int(x[1]) for x in predict_table]
i=0
new_file = open('./bonus.out', "w")
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

# In[55]:


output = []
model.to(device)
model.eval()
for test_sentence, test_char in loader_test:
    test_sentence = test_sentence.to(device)
    test_char = test_char.to(device)
    test_pred = model(test_sentence, test_char)
    test_pred = test_pred.view(-1, test_pred.shape[-1])
    max_pred = test_pred.argmax(dim=1, keepdim = True)
    for tmp_pred, word in zip(max_pred, test_sentence.view(-1)):
        if word != 0:
            output.append(tmp_pred.item())


# In[56]:


length = 0
with open('./data/test', "r") as file:
    for line in file:
        if len(line) > 1: 
            length+=1
file.close()
print(length)
print(len(output))


# In[57]:


i=0
new_file = open('./pred.out', "w")
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


import os, sys
import pandas as pd
import scipy as sp
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import movie_reviews
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader, dataset
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from SmilesPE.tokenizer import *
from smiles_pair_encoders_functions import *
from itertools import chain, repeat, islice
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
'''
Initialize tokenizer
'''
vocab_file = 'VocabFiles/vocab_spe.txt'
spe_file = 'VocabFiles/SPE_ChEMBL.txt'
tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def tokenize_function(examples):
    #print(examples[0])
    return np.array(list(pad(tokenizer(examples)['input_ids'], 25, 0)))

def training_data(raw_data):
    smiles_data_frame = pd.DataFrame(data = {'text': raw_data['Canonical SMILES'], 'labels': raw_data['Toxicity Value']})
    print(smiles_data_frame['text'])
    smiles_data_frame['text'] = smiles_data_frame['text'].map(tokenize_function)#, batched=True)
    print(smiles_data_frame['text'].values)
    target = smiles_data_frame['labels'].values
    features = np.stack([tok_dat for tok_dat in smiles_data_frame['text']])
    print(target)
    #train = data_utils.TensorDataset(features, target)
    #train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
    feature_tensor = torch.tensor(features)
    label_tensor = torch.tensor(smiles_data_frame['labels'])
    print(feature_tensor)
    print(label_tensor)
    dataset = TensorDataset(feature_tensor, label_tensor)
    print(len(dataset[0][0]))
    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)

    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    #print(training_data.shape)
    print(len(test_data))
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader, test_data


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(3132, d_model)
        self.d_model = d_model

        self.dropout1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(3200, 1024)
        self.act1 = nn.ReLU()

        self.dropout2 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(1024, 256)
        self.act2 = nn.ReLU()

        self.dropout3 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(256, 64)
        self.act3 = nn.ReLU()

        self.dropout4 = nn.Dropout(0.1)
        self.linear4 = nn.Linear(64, 16)
        self.act4 = nn.ReLU()

        self.dropout5 = nn.Dropout(0.1)
        self.linear5 = nn.Linear(16, 1)
        self.act5 = nn.Softmax()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.zero_()
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear4.bias.data.zero_()
        self.linear4.weight.data.uniform_(-initrange, initrange)
        self.linear5.bias.data.zero_()
        self.linear5.weight.data.uniform_(-initrange, initrange)
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src)* math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.dropout1(output)
        output = torch.reshape(output, (len(output),len(output[0])*len(output[0][0])))
        output = self.linear1(output)
        output = self.act1(output)
        output = self.dropout2(output)
        output = self.linear2(output)
        output = self.act2(output)
        output = self.dropout3(output)
        output = self.linear3(output)
        output = self.act3(output)
        output = self.dropout4(output)
        output = self.linear4(output)
        output = self.act4(output)
        output = self.dropout5(output)
        output = self.linear5(output)
        output = self.act5(output)
        return torch.reshape(output, (-1,))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
directory = 'data'
dir_list = os.listdir(directory)

for fil in dir_list:
    raw = pd.read_csv(f'{directory}/{fil}')
    base = Path(f'{fil}').stem

    train_loader, test_loader, test_data = training_data(raw)
    
    model = TransformerModel(ntoken=25, d_model=128, nhead=8, d_hid=128,
                     nlayers=4, dropout= 0.1)
    
    
    # define optimizer. Specify the parameters of the model to be trainable. Learning rate of .001
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
    loss_fn = nn.NLLLoss() #nn.CrossEntropyLoss
    
    # some extra variables to track performance during training
    f1_history = []
    trainstep = 0
    running_loss = 0.
    
    for i in tqdm(range(100)):
        for j, (batch_X, batch_y) in enumerate(train_loader):
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if False:
                # Compute the loss and its gradients
                # Gather data and report
                running_loss += loss.item()
                last_loss = running_loss # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
    
        for k, (batch_Xt, batch_yt) in enumerate(test_loader):
            y_hat = model(batch_Xt).detach().numpy() >=.5
            y_grnd = batch_yt.detach().numpy()==1
            f1_history.append({'epoch' : i, 'minibatch' : k, 'trainstep' : trainstep,
                                      'task' : 'tox', 'f1' : f1_score(y_grnd, y_hat)})
            trainstep += 1
        
        if i % 10 == 0:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'models/{base}.pt')

    f1_df = pd.DataFrame(f1_history)
    f1_df.to_csv(f'f1_{base}.csv')

sys.exit()









if False:

    class SingleTask_Network(nn.Module):
        def __init__(self, input_dim : int, 
                     output_dim : int = 1, 
                     hidden_dim : int = 300):
            super(SingleTask_Network, self).__init__()
            
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim
            
            self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
            self.final = nn.Linear(self.hidden_dim, self.output_dim)

        def forward(self, x : torch.Tensor):
            x = self.hidden(x)
            x = torch.sigmoid(x)
            x = self.final(x)
            return x

    def make_preds_single(model, X):
        # helper function to make predictions for a model
        with torch.no_grad():
            y_hat = model(torch.from_numpy(X.astype(np.int8).todense()).float())
        return y_hat
    
    '''
    Implement multi_task network
    '''
    
    class MultiTask_Network(nn.Module):
        """
          Multi-task NN model. Shared input and single hidden layer for both tasks.
          Individual final layers for each task. During training and test, `forward`
          method takes a `task_id` argument which specifies which task is the current one
          and therefore which final layer to use.
    
        """
        def __init__(self, input_dim,
                     output_dim_0 : int = 1, output_dim_1 : int = 1,
                     hidden_dim : int = 100):
            super(MultiTask_Network, self).__init__()
            self.input_dim = input_dim
            self.output_dim_0 = output_dim_0
            self.output_dim_1 = output_dim_1
            self.hidden_dim = hidden_dim
    
            # define a single hidden layer
            self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
            # define final layers for each task
            self.final_0 = nn.Linear(self.hidden_dim, self.output_dim_0)
            self.final_1 = nn.Linear(self.hidden_dim, self.output_dim_1)
    
        def forward(self, x : torch.Tensor, task_id : int):
            x = self.hidden(x)
            x = torch.sigmoid(x)
            # final output is determined by which task_id is passed in and
            # which final layer is used to transform the penultimate values
            if task_id == 0:
                x = self.final_0(x)
            elif task_id == 1:
                x = self.final_1(x)
            else:
                assert False, 'Bad Task ID passed'
    
            return x
    
    def make_preds_multi(model, X, task_id):
        # helper function for multi-task model
        # will automatically apply sigmoid or softmax depending on task
        with torch.no_grad():
            if task_id == 0:
                y_hat = torch.sigmoid(model(torch.from_numpy(X.astype(np.int8).todense()).float(),
                  task_id = task_id).squeeze()).detach().numpy()
                y_hat = (y_hat > .5).astype(np.int8)
            elif task_id == 1:
                y_hat = torch.softmax(model(torch.from_numpy(X.astype(np.int8).todense()).float(),
                  task_id = task_id).squeeze(), dim = 1).detach().numpy()
        return y_hat

    def make_preds_single(model, X):
        # helper function to make predictions for a model
        with torch.no_grad():
            y_hat = model(torch.from_numpy(X.astype(np.int8).todense()).float())
        return y_hat








    '''
    Load Datasets
    '''
    
    def load_data():
        """
          Loads labeled movie reviews from NLTK. 
          Returns a pandas dataframe with `text` column and `is_positive` for labels.
        """
        nltk.download('movie_reviews', quiet = True)
        
        negative_review_ids = movie_reviews.fileids(categories=["neg"])
        positive_review_ids = movie_reviews.fileids(categories=["pos"])
            
        neg_reviews = [movie_reviews.raw(movie_id) for movie_id in movie_reviews.fileids(categories = ["neg"])]
        pos_reviews = [movie_reviews.raw(movie_id) for movie_id in movie_reviews.fileids(categories = ["pos"])]
        labels = [0 for _ in range(len(neg_reviews))] + [1 for _ in range(len(pos_reviews))]
    
        movie_df = pd.DataFrame(data = {'text' : neg_reviews + pos_reviews, 'is_positive' : labels})
        movie_df.to_csv('data/movies.csv', index=False)
        return movie_df
    
    def load_yelp_data():
        """
          Loads in a sample of the larger Yelp dataset of annotated restaurant reviews. Reviews are from 1 to 5 stars.
          Removes 2 and 4 star reviews, leaving negative (1 star), neutral (3 stars) and positive (5 stars) reviews
          Returns a pandas dataframe with `text` column and `is_negative`, `is_neutral`, `is_positive` columns
        """
        full_df = pd.read_csv('data/yelp.csv')
        df = full_df[full_df['stars'].isin([1, 3, 5])].copy().reset_index(drop = True)
        df['is_positive'] = (df['stars'] == 5).astype(int)
        df['is_neutral'] = (df['stars'] == 3).astype(int)
        df['is_negative'] = (df['stars'] == 1).astype(int)
        return df
    
    def prep_data(downsample_yelp = True):
        """
          Loads both datasets and creates a bag-of-words representation for both. Because Yelp is larger than
          the movie dataset, downsample to be equivalent size.
          Returns a dictionary which contain the input and output variables for either set.
        """
        movie_df = load_movie_data()
        yelp_df = load_yelp_data()
        
        # downsample to be equal sizes
        if downsample_yelp:
            yelp_df = yelp_df.sample(len(movie_df))
        
        cv_hyperparams = {'min_df' : 3, 'max_df' : .33, 'stop_words' : 'english'}
        yelp_cv = CountVectorizer(**cv_hyperparams).fit(yelp_df['text'].tolist())
        movie_cv = CountVectorizer(**cv_hyperparams).fit(movie_df['text'].tolist())
        
        # so that the model doesn't overfit on specific terms unique to either dataset, CountVectorizer's vocab is words
        # shared by both
        shared_vocab = set(yelp_cv.vocabulary_.keys()).intersection(set(movie_cv.vocabulary_.keys()))
        
        cv = CountVectorizer(vocabulary = shared_vocab).fit(movie_df['text'].tolist() + yelp_df['text'].tolist())
        data = {'movie' : 
                   {'X' : cv.transform(movie_df['text']).astype(np.int8),
                   'y' : movie_df['is_positive'].astype(np.int8).values.reshape(-1, 1)},
               'yelp' : 
                   {'X' : cv.transform(yelp_df['text']).astype(np.int8),
                   'y' : yelp_df[['is_positive', 'is_neutral', 'is_negative']].astype(np.int8).values}
               }
        return data
    
    
    '''
    Test using single task network
    '''
    
    class Task_Dataset(Dataset):
        def __init__(self, X : sp.sparse.csr.csr_matrix, 
                           y : np.ndarray):
            self.X = X
            self.y = torch.from_numpy(y).float()
            assert self.X.shape[0] == self.y.shape[0]
            
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):    
            X = torch.from_numpy(self.X[idx].astype(np.int8).todense()).float().squeeze()
            y = self.y[idx]
            return X, y





    data = prep_data()
    
    movie_X_train, movie_X_test, movie_y_train, movie_y_test =  train_test_split(data['movie']['X'], data['movie']['y'], 
                                                                                 random_state = 1225)
    
    yelp_X_train, yelp_X_test, yelp_y_train, yelp_y_test =  train_test_split(data['yelp']['X'], data['yelp']['y'], 
                                                                                 random_state = 1225)
    
    
    # create Dataset and Dataloaders for each. Batch size 64
    
    movie_ds = Task_Dataset(movie_X_train, movie_y_train)
    movie_dl = DataLoader(movie_ds, batch_size = 64, shuffle = True)
    
    yelp_ds = Task_Dataset(yelp_X_train, yelp_y_train)
    yelp_dl = DataLoader(yelp_ds, batch_size = 64, shuffle = True)
    
    model = SingleTask_Network(movie_ds.X.shape[1], movie_ds.y.shape[1])
    # define optimizer. Specify the parameters of the model to be trainable. Learning rate of .001
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # some extra variables to track performance during training
    f1_history = []
    trainstep = 0
    for i in tqdm(range(6)):
        for j, (batch_X, batch_y) in enumerate(movie_dl):
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            y_hat = torch.sigmoid(make_preds_single(model, movie_X_test)) >= .5
            f1_history.append({'epoch' : i, 'minibatch' : j, 'trainstep' : trainstep,
                              'task' : 'movie', 'f1' : f1_score(movie_y_test, y_hat)})
            trainstep += 1
    
    # create Dataset and Dataloaders for each. Batch size 64
    
    movie_ds = Task_Dataset(movie_X_train, movie_y_train)
    movie_dl = DataLoader(movie_ds, batch_size = 64, shuffle = True)
    
    yelp_ds = Task_Dataset(yelp_X_train, yelp_y_train)
    yelp_dl = DataLoader(yelp_ds, batch_size = 64, shuffle = True)
    
    
    
    
    
    
    
    model = SingleTask_Network(movie_ds.X.shape[1], movie_ds.y.shape[1])
    # define optimizer. Specify the parameters of the model to be trainable. Learning rate of .001
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # some extra variables to track performance during training 
    f1_history = []
    trainstep = 0
    for i in tqdm(range(6)):
        for j, (batch_X, batch_y) in enumerate(movie_dl):
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_hat = torch.sigmoid(make_preds_single(model, movie_X_test)) >= .5
            f1_history.append({'epoch' : i, 'minibatch' : j, 'trainstep' : trainstep,
                              'task' : 'movie', 'f1' : f1_score(movie_y_test, y_hat)})
            trainstep += 1
    
    # overall F1 of fit model. ~.82 F1 score. Not too bad
    y_hat = torch.sigmoid(make_preds_single(model, movie_X_test)) >= .5
    print(classification_report(movie_y_test, y_hat, 
                                target_names=['negative', 'positive']))
    
    st_movie_graph_df = pd.DataFrame(f1_history)
    
    model = SingleTask_Network(yelp_ds.X.shape[1], yelp_ds.y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # because Yelp data is multilabel, not binary, use CrossEntropy as loss function
    loss_fn = nn.CrossEntropyLoss()
    
    f1_history = []
    trainstep = 0
    for i in tqdm(range(6)):
        for j, (batch_X, batch_y) in enumerate(yelp_dl):
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_hat = torch.softmax(make_preds_single(model, yelp_X_test), axis = 0)
            f1_history.append({'epoch' : i, 'minibatch' : j, 'trainstep' : trainstep,
                              'task' : 'yelp', 'f1' : f1_score(yelp_y_test.argmax(axis = 1), y_hat.argmax(axis = 1),
                                                                average = 'micro')})
            trainstep += 1
    
    # performance of fit Yelp model. ~.66, not as good as movie data but it is a slightly harder task w/ 3 labels
    y_hat = torch.softmax(make_preds_single(model, yelp_X_test), axis = 0)
    print(classification_report(yelp_y_test.argmax(axis = 1), y_hat.argmax(axis = 1), 
                                target_names=['negative', 'neutral', 'positive']))
    
    
    
    model = MultiTask_Network(movie_ds.X.shape[1], 
                               output_dim_0 = movie_ds.y.shape[1], 
                              output_dim_1 = yelp_ds.y.shape[1])
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # define the loss functions for each task
    movie_loss_fn = nn.BCEWithLogitsLoss()
    yelp_loss_fn = nn.CrossEntropyLoss()
    
    f1_history = []
    grad_history = []
    trainstep = 0
    for i in tqdm(range(6)):
        losses_per_epoch = []
        grads_per_epoch = []
        # zip the dataloaders anew for each epoch (learned from trial and error :D)
        zipped_dls = zip(movie_dl, yelp_dl)
        for j, ((movie_batch_X, movie_batch_y), (yelp_batch_X, yelp_batch_y)) in enumerate(zipped_dls):
            # forward pass and loss calculation for task 0
            movie_preds = model(movie_batch_X, task_id = 0)
            movie_loss = movie_loss_fn(movie_preds, movie_batch_y)
            
            # forward pass and loss calculation for task 1
            yelp_preds = model(yelp_batch_X, task_id = 1)
            yelp_loss = yelp_loss_fn(yelp_preds, yelp_batch_y)
            
            # sum the loss for each
            # in theory, the weight for each task could be weighed differently, e.g.,
            # loss = wgt_0 * loss_0 + wgt_1 * loss_1
            loss = movie_loss + yelp_loss
            losses_per_epoch.append(loss.item())
            
            # just like the single-task model, zero gradient, backprop and update
            # weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_hat = make_preds_multi(model, movie_X_test, task_id = 0)
            f1_history.append({'epoch' : i, 'minibatch' : j, 'trainstep' : trainstep,
                              'task' : 'movie', 'f1' : f1_score(movie_y_test, y_hat)})
            y_hat = make_preds_multi(model, yelp_X_test, task_id = 1)
            f1_history.append({'epoch' : i, 'minibatch' : j, 'trainstep' : trainstep,
                              'task' : 'yelp', 'f1' : f1_score(yelp_y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'micro')})
            
            grad_history.append({'epoch' : i, 'minibatch' : j, 'trainstep' : trainstep,
                                 'layer' : 'hidden_layer', 'grad_norm' : model.hidden._parameters['weight'].grad.norm().item()})
            grad_history.append({'epoch' : i, 'minibatch' : j, 'trainstep' : trainstep,
                                 'layer' : 'movie_final_layer', 'grad_norm' : model.final_0._parameters['weight'].grad.norm().item()})
            grad_history.append({'epoch' : i, 'minibatch' : j, 'trainstep' : trainstep,
                                 'layer' : 'yelp_final_layer', 'grad_norm' : model.final_1._parameters['weight'].grad.norm().item()})
            
            trainstep += 1
    
    # performance of movie task for multi-task model. Similar performance to single-task at ~.82 F1
    y_hat = make_preds_multi(model, movie_X_test, task_id = 0)
    print(classification_report(movie_y_test, y_hat, target_names=['negative', 'positive']))
    
    # performance of yelp data. Teensy boost of single-task :)
    y_hat = make_preds_multi(model, yelp_X_test, task_id = 1)
    print(classification_report(yelp_y_test.argmax(axis = 1), y_hat.argmax(axis = 1), target_names=['negative', 'neutral', 'positive']))

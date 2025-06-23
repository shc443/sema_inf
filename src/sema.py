# !pip install --update openpyxl

# from pl_bolts.callbacks import PrintTableMetricsCallback

# import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MultilabelF1Score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
# from keras.utils.data_utils import pad_sequences

import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd
import os

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW, BertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import RobertaConfig, RobertaModel
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForMaskedLM, DebertaV2Config, DebertaV2ForSequenceClassification
from transformers import RobertaConfig, RobertaModel
import numpy as np
import random
import time
import datetime
import pickle 
from tqdm import tqdm
tqdm.pandas()

# from konlpy.tag import Mecab
from tqdm import tqdm
from collections import Counter
# mecab = Mecab()

from os import listdir
from os.path import isfile, join

class VOC_Dataset2(Dataset):

  def __init__(self, data, tokenizer, max_token_len=512):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    data_row = self.data.iloc[index]

    voc_text   = data_row.VOC
    voc_labels = data_row.label

    encoding = self.tokenizer.encode_plus(voc_text,
                                          add_special_tokens=True,
                                          max_length=self.max_token_len,
                                          return_token_type_ids=False,
                                          padding="max_length",
                                          truncation=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')

    return dict(voc_text=voc_text,
                input_ids=encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten(),
                labels=torch.FloatTensor(voc_labels))

class VOC_DataModule:
  def __init__(self, train_df, test_df, tokenizer, batch_size=4, max_token_len=200):
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
    self.test_dataset  = VOC_Dataset2(self.test_df, self.tokenizer, self.max_token_len)

  def setup(self, stage=None):
    self.train_dataset = VOC_Dataset2(self.train_df, self.tokenizer, self.max_token_len)         
    self.test_dataset  = VOC_Dataset2(self.test_df, self.tokenizer, self.max_token_len)
    
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
  def val_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size)
  
  def predict_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size)

class VOC_TopicLabeler(nn.Module):
  def __init__(self, n_classes=None, n_training_steps=None, n_warmup_steps=10000, model = None):
    super().__init__()
    # self.config = config

    self.config = AutoConfig.from_pretrained('{model}'.format(model = model), output_hidden_states=True)
    self.model = AutoModelForMaskedLM.from_pretrained('{model}'.format(model = model), config=self.config)
    
    self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCEWithLogitsLoss() #nn.BCELoss() with sigmoid layer 
    self.dropout = nn.Dropout(self.config.hidden_dropout_prob) 
    self.dense = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
    self.activation = nn.Tanh()
    self.acc = F1Score(task='multilabel', num_labels=n_classes)
    self.training_step_outputs = []


  def forward(self, input_ids, attention_mask, labels=None):
    output = self.model(input_ids, attention_mask=attention_mask)
    last_hidden_state = output.hidden_states[-1]
    pooled_output = self.classifier(self.dropout(self.activation(self.dense(last_hidden_state[:,0]))))
    loss = 0
    if labels is not None:
        loss = self.criterion(pooled_output, labels)
    return loss, torch.sigmoid(pooled_output)

  def predict(self, dataloader, device='cuda'):
    """Run inference on a dataloader"""
    self.eval()
    predictions = []
    
    with torch.no_grad():
      for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        loss, outputs = self(input_ids, attention_mask)
        predictions.append(outputs.cpu())
    
    return predictions
  
  @classmethod
  def load_from_checkpoint(cls, checkpoint_path, **kwargs):
    """Load model from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model instance
    model = cls(**kwargs)
    
    # Load state dict
    if 'state_dict' in checkpoint:
      model.load_state_dict(checkpoint['state_dict'])
    else:
      model.load_state_dict(checkpoint)
    
    return model

def findall_vec(key,voc):
  try:
    return re.findall(key, voc)[0]
  except:
    return ''

def findall_vec2(df):
  return findall_vec(df['keyword'],df['VOC'])

def filter_etc(df):
  voc_col = df['VOC'].apply(lambda x: re.sub('[^A-Za-z0-9가-힣 ]', '', x))
  filt0 = (voc_col.str.len() < 2).astype(int)
  filt1 = voc_col.apply(lambda x : bool(re.match(r'^[_\W]+$', str(x).replace(' ','')))).astype(int)
  filt2 = voc_col.apply(lambda x : bool(re.match(r'[\d/-]+$', str(x).replace(' ','')))).astype(int)
  filt3 = voc_col.str.replace(' ','').str.split('').fillna('').apply(set).str.len() == 2
  # filt4 = voc_col.progress_apply(lambda x : tuple(Counter(mecab.morphs(x)).keys())).isin(voc_etc.apply(lambda x : tuple(x.keys())))
  return filt0+filt1+filt2+filt3#+filt4
    
def tokenize_text(txt, voc_labels):
  encoding = tokenizer.encode_plus(txt, add_special_tokens=True,
                                        max_length=MAX_LEN,
                                        return_token_type_ids=False,
                                        padding="max_length",
                                        truncation=True,
                                        return_attention_mask=True,
                                        return_tensors='pt')
  
  return encoding["input_ids"].flatten(), encoding["attention_mask"].flatten(), torch.FloatTensor(voc_labels)
pd.options.mode.chained_assignment = None  # default='warn'

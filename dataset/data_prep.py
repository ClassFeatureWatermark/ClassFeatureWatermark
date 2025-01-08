'''
Prepare the AG News dataset as tokenized torch
ids tensor and attention mask
'''

import torch
import torch.nn as nn
from transformers import ElectraTokenizer, BertTokenizer, RobertaTokenizer
# from transformers.models.clap.convert_clap_original_pytorch_to_hf import processor

from datasets import load_dataset

def get_data_dbpedia(data):
    texts = data['content']
    labels = data['label']

    # Tokenize and prep input tensors
    # if arch == 'electra':
    #     tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    # elif arch == 'bert':
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # elif arch == 'roberta':
    #     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoded_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=512,  return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    labels = torch.LongTensor(labels)

    return ids, mask, labels

def get_data(data):
    texts = data['text_utils']
    labels = data['label']

    # Tokenize and prep input tensors
    # if arch == 'electra':
    #     tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    # elif arch == 'bert':
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # elif arch == 'roberta':
    #     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoded_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    labels = torch.LongTensor(labels)

    return ids, mask, labels

class DataTensorLoader():
    def __init__(self, dataname):

        allowed_arch = ['electra', 'bert', 'roberta']
        self.arch = 'bert'
        #
        self.dataset = load_dataset(dataname)
    def get_train(self):
        return get_data(self.dataset['train'])

    def get_test(self):
        return get_data(self.dataset['test'])

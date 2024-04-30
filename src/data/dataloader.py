# a file with all the preprocessing functions for the data

from datasets import Dataset
from functools import partial, reduce
from transformers import AutoTokenizer
from pandas import read_json
import os
from tqdm import tqdm




def encode_labels(example, label2id):
    """
    to be used with datasets.map() with batched=False
    
    Encodes the labels into integers.
    
    """
    labels = example['labels']
    encoded = [label2id[label] for label in labels]
    return {'labels': encoded}


def tokenize_and_align(example, tokenizer, overlap_size = 0):
    """
    To be used with datasets.map() with batched=False

    Takes in 
        - example : an example from the datasets class
        - overlap_size: the number of tokens that overlap between two consecutive chunks
        
    outputs:
        - a Dict[]->List with columns:
            - of the bert tokenizer output
            - encoded labels
    """

    org_labels = example['labels']
    tokenized_inputs = tokenizer(example['tokens'], is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding='max_length', max_length=512, return_overflowing_tokens=True, stride=overlap_size, return_tensors='pt')
    tokenized_inputs.pop('overflow_to_sample_mapping')
    tokenized_inputs.pop('offset_mapping')
    
    new_labels = []
    org_word_ids_list = []
    document_id = []
    #iterating over chunks
    for i, chunk in enumerate(tokenized_inputs['input_ids']):
        ids_of_tokens = tokenized_inputs.word_ids(i)
        
        org_word_ids_list.append(ids_of_tokens)
        document_id.append(example['document'])
        #iterating over ids of tokens
        chunk_labels = []
        for id in ids_of_tokens:
            #if id=None, then it means it's some BERT token (CLS, SEP or PAD)
            if id is None:
                chunk_labels.append(-100)
            else:
                chunk_labels.append(org_labels[id])
        new_labels.append(chunk_labels)

    tokenized_inputs['labels'] = new_labels
    tokenized_inputs['org_word_ids'] = org_word_ids_list
    tokenized_inputs['document'] = document_id

    return tokenized_inputs

def flatten_data(data, keys_to_flatten):
    """
    Takes in:
        - data: a dataset object
        - keys_to_flatten: a list with the keys to flatten
    Outputs:
        - a dataset object with the keys_to_flatten columns
    """

    data_flat = {}


    for key in tqdm(keys_to_flatten):
        data_flat[key] = reduce(lambda x,y: x+y, data[key])

    return Dataset.from_dict(data_flat)


def preprocess_data(data_path, tokenizer, label2id, overlap_size=0, keys_to_keep=[]):
    """
    Preprocesses the data
    
    Takes in 
        - data: a string with the path to the data
        - tokenizer: a tokenizer object
        - label2id: a dictionary with the labels and their corresponding ids
        - overlap_size: the number of tokens that overlap between two consecutive chunks
        - keys_to_keep: a list with the columns to keep
        
    outputs:
        - a dataset object with 
            - the keys_to_keep columns
            - the columns from the tokenizer output 
            - the 'labels' column encoded
    """
    
    data_pd = read_json(data_path)
    data = Dataset.from_pandas(data_pd)

    print(data)

    keys_to_flatten = ['labels', 'input_ids', 'token_type_ids', 'attention_mask', 'org_word_ids'] #+ keys_to_keep

    print("encoding the labels...")
    data = data.map(partial(encode_labels, label2id = label2id), batched=False)

    print("tokenizing and aligning...")
    data = data.map(partial(tokenize_and_align, tokenizer=tokenizer, overlap_size=overlap_size), batched=False)

    print("flattening the data...")
    data = flatten_data(data, keys_to_flatten)
    
    return data


if __name__=='__main__':
    print('running dataloader.py')
    local_path = os.path.abspath(os.path.dirname(__file__))
    local_path = os.path.join(local_path, '../')
    data_path = os.path.join(local_path, '../data/raw/synthetic/mixtral.json')

    label2id = {
        'B-NAME_STUDENT': 0, 
        'B-EMAIL': 1, 
        'B-USERNAME': 2, 
        'B-ID_NUM': 3, 
        'B-PHONE_NUM': 4, 
        'B-URL_PERSONAL': 5, 
        'B-STREET_ADDRESS': 6, 
        'I-NAME_STUDENT': 7, 
        'I-EMAIL': 8, 
        'I-USERNAME': 9, 
        'I-ID_NUM': 10, 
        'I-PHONE_NUM': 11, 
        'I-URL_PERSONAL': 12, 
        'I-STREET_ADDRESS': 13, 
        'O': 14, 
        '[PAD]': -100}
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    data = preprocess_data(data_path, tokenizer, label2id, overlap_size=0)

    print(data)
    
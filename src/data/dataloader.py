# a file with all the preprocessing functions for the data

from datasets import Dataset
from functools import partial, reduce
from transformers import AutoTokenizer
from pandas import read_json, read_csv
import os
from tqdm import tqdm




def encode_labels(example, label2id):
    """
    Encodes the labels into integers
    to be used with datasets.map() with batched=False
    
    Encodes the labels into integers.
    
    """
    labels = example['labels']
    encoded = [label2id[label] for label in labels]
    return {'labels': encoded}




def tokenize_and_align(example, tokenizer, with_labels = True, overlap_size = 0):
    """
    Tokenizes the input and aligns the labels with the tokens
    To be used with datasets.map() with batched=False

    Takes in 
        - example : an example from the datasets class
        - overlap_size: the number of tokens that overlap between two consecutive chunks
        
    outputs:
        - a Dict[]->List with columns:
            - of the bert tokenizer output
            - encoded labels
    """

    if with_labels:
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

        if with_labels:
            #iterating over ids of tokens
            chunk_labels = []
            for id in ids_of_tokens:
                #if id=None, then it means it's some BERT token (CLS, SEP or PAD)
                if id is None:
                    chunk_labels.append(-100)
                else:
                    chunk_labels.append(org_labels[id])
            new_labels.append(chunk_labels)

    if with_labels:
        tokenized_inputs['labels'] = new_labels
    
    tokenized_inputs['org_word_ids'] = org_word_ids_list
    tokenized_inputs['document'] = document_id

    return tokenized_inputs

def flatten_data(data, keys_to_flatten):
    """
    Flattens the rows of the datasets object for the keys_to_flatten columns

    Takes in:
        - data: a dataset object
        - keys_to_flatten: a list with the keys to flatten
    Outputs:
        - a dataset object with the keys_to_flatten columns
    """

    data_flat = {}

    for key in tqdm(keys_to_flatten):
        data_flat[key] = reduce(lambda x, y: x + y, data[key])

    return Dataset.from_dict(data_flat)


def preprocess_data(data, tokenizer, label2id = {}, with_labels = True, overlap_size=0, keys_to_flatten=['input_ids', 'token_type_ids', 'attention_mask', 'org_word_ids', 'document']):
    """
    Preprocesses the data
    
    Takes in 
        - data: a dataset object with columns 'document', 'tokens' (if with_labels=True, also has to have 'labels')
        - tokenizer: a tokenizer object
        - label2id: a dictionary with the labels and their corresponding ids. If with_labels=True, this has to be provided. By default, it's an empty dictionary.
        - with_labels: a boolean indicating if the data has labels. By default, it's True.
        - overlap_size: the number of tokens that overlap between two consecutive chunks. By default, it's 0.
        - keys_to_flatten : a list of columns to keep in the output dataset. By default, it's ['input_ids', 'token_type_ids', 'attention_mask', 'org_word_ids', 'document']
        
    outputs:
        - a dataset object with keys_to_flatten columns
    """

    assert 'document' in data.column_names, "data has to have a 'document' column"
    assert 'tokens' in data.column_names, "data has to have a 'tokens' column"
    if with_labels:
        assert 'labels' in data.column_names, "data has to have a 'labels' column"
        assert label2id, "label2id has to be provided if with_labels=True"

    if with_labels:
        keys_to_flatten.append('labels')

        print("encoding the labels...")
        data = data.map(partial(encode_labels, label2id = label2id), batched=False)

    print("tokenizing and aligning...")
    data = data.map(partial(tokenize_and_align, tokenizer=tokenizer, overlap_size=overlap_size, with_labels = with_labels), batched=False)

    print("flattening the data...")
    data = flatten_data(data, keys_to_flatten)
    
    return data

def get_dataset_from_path(data_path):
    """
    Loads a dataset from a path and returns it as a datasets object

    Takes in 
        - data: a string with the path to the data (has to be a json or csv file)
    
    outputs:
        - a datasets object
    """

    filetype = data_path.split('.')[-1]
    data = None
    if filetype == 'json':
        data = read_json(data_path)
    elif filetype == 'csv':
        data = read_csv(data_path)
    else:
        raise ValueError('Filetype not supported. Suuported filetypes are: json, csv')
    
    data = Dataset.from_pandas(data)

    return data

def get_train_val_test_split(data, seed, val_size=0.1, test_size=0.1):
    """
    Takes in:
        - data: a dataset object
        - seed: the seed for the random split
        - val_size: the size of the validation set
        - test_size: the size of the test set
    Outputs:
        - a tuple with data_train, data_val, data_test
    """

    data = data.train_test_split(test_size=test_size, seed = seed)
    data_train_val = data['train'].train_test_split(test_size=val_size, seed = seed)

    return data_train_val['train'], data_train_val['test'], data['test']


if __name__=='__main__':
    print('running dataloader.py')
    local_path = os.path.abspath(os.path.dirname(__file__))
    local_path = os.path.join(local_path, '../')
    data_path = os.path.join(local_path, '../data/raw/test.json')

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

    data = get_dataset_from_path(data_path)
    data = preprocess_data(data, tokenizer, label2id, with_labels=False)

    print('dataset\n',data)

    # train, val, test = get_train_val_test_split(data, seed=42, val_size=0.1, test_size=0.1)
    # print('\ntrain:\n',train,'\nval:\n', val, '\ntest:\n', test)
    
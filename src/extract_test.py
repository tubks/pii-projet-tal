from preprocessing import preprocess_data, get_dataset_from_path, get_train_val_test_split
from transformers import AutoTokenizer
import os
import json

if __name__ == '__main__':
    # setting the labels and the tokenizer
    LABELS_LIST = ['B-NAME_STUDENT', 'B-EMAIL', 'B-USERNAME', 'B-ID_NUM', 'B-PHONE_NUM', 'B-URL_PERSONAL', 'B-STREET_ADDRESS', 'I-NAME_STUDENT', 'I-EMAIL', 'I-USERNAME', 'I-ID_NUM', 'I-PHONE_NUM','I-URL_PERSONAL','I-STREET_ADDRESS', 'O']
    label2id = {label: i for i, label in enumerate(LABELS_LIST)}
    id2label = {i: label for label, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    seed = 42

    # loading the data
    data_path = os.path.join('.','data', 'raw', 'train.json')
    data = get_dataset_from_path(data_path)
    data = preprocess_data(data, tokenizer, label2id = label2id)
    data_train, data_eval, data_test = get_train_val_test_split(data, seed=seed)

    # getting the document ids that are in train and val
    documents_train = set(data_train['document'])
    documents_eval = set(data_eval['document'])
    # filtering the test data to get only the chunks that are not part of documents that are in train and val
    data_test_only_in_test = data_test.filter(lambda x: (x['document'] not in documents_train) and (x['document'] not in documents_eval))
    
    # loading the unprocessed data again and extracting the essays that are ONLY in the test set
    data_test_only_doc_ids_set =set(data_test_only_in_test['document'])
    data = get_dataset_from_path(data_path)
    data_test_texts = data.filter(lambda x: x['document'] in data_test_only_doc_ids_set)

    # formatting and saving the test data
    data_test_list = [dict(sample) for sample in data_test_texts]
    with open(os.path.join('.','data', 'test_texts.json'), 'w') as f:
        json.dump(data_test_list, f, indent=4)
    data = get_dataset_from_path(os.path.join('.','data', 'test_texts.json'))
#this file contains the code for the baseline model

from data.dataloader import preprocess_data
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer



class CFG:
    LABELS_LIST = ['B-NAME_STUDENT', 'B-EMAIL', 'B-USERNAME', 'B-ID_NUM', 'B-PHONE_NUM', 'B-URL_PERSONAL', 'B-STREET_ADDRESS', 'I-NAME_STUDENT', 'I-EMAIL', 'I-USERNAME', 'I-ID_NUM', 'I-PHONE_NUM','I-URL_PERSONAL','I-STREET_ADDRESS', 'O']
    #BERT model
    model_name = 'bert-base-uncased'
    #batch size
    batch_size = 8
    #number of epochs
    epochs = 5
    #learning rate
    lr = 1e-5

    seed = 42

    label2id = {label: i for i, label in enumerate(LABELS_LIST)}
    label2id['[PAD]'] = -100
    id2label = {i: label for label, i in label2id.items()}

    local_path = os.path.abspath(os.path.dirname(__file__))
    target_dir = os.path.join(local_path,'..','models', 'baseline')

    training_args = TrainingArguments(
        output_dir=os.path.join(target_dir, 'trainer'), 
        evaluation_strategy="epoch"
        )



if __name__ == '__main__':

    data_path = os.path.join(CFG.local_path, '../data/raw/synthetic/mixtral.json')

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForTokenClassification.from_pretrained(CFG.model_name, num_labels=len(CFG.id2label), id2label=CFG.id2label, label2id=CFG.label2id)
    
    data = preprocess_data(data_path, tokenizer, CFG.label2id, keys_to_keep=['document'])

    test_dir = os.path.join(CFG.local_path,'..','models')
    list_files = os.listdir(test_dir)
    print(list_files)
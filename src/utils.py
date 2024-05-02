from transformers import Trainer, AutoModelForTokenClassification, AutoTokenizer
import os
import torch
import numpy as np
import pandas as pd
from data.dataloader import preprocess_data
from datasets import Dataset

def train_model(data_train, data_eval, model, training_args, compute_metrics, target_dir):
    """
    Trains and Saves the model to the directory specified by target_dir
    Takes in:
        - data_train: training data as a Dataset object
        - data_eval: evaluation data as a Dataset object
        - model: a model object
        - training_args: a TrainingArguments object
        - target_dir: a string representing the path to save the model
    Returns:
        - trainer: a Trainer object that has been trained
    """
    model_save_path = os.path.join(target_dir, 'model')

    # checking if the directory exists
    assert os.path.exists(target_dir), f"Directory {target_dir} does not exist"

    assert isinstance(data_train, Dataset), "data_train is not a Dataset object"
    assert isinstance(data_eval, Dataset), "data_eval is not a Dataset object"


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    
    trainer.save_model(model_save_path)
    return trainer


def _get_labeled_tokens(data, predictions):
    """
    DO NOT USE
    """
    
    document_list = []
    token_id_list = []
    label_id_list = []
    for doc, token_id, pred in zip(data['document'],data['org_word_ids'],predictions):
        for i in range(len(predictions)):
            current_word_id = token_id
            if token_id[i] != None:
                document_list.append(doc)
                token_id_list.append(token_id[i])
                label_id_list.append(pred[i])
    
    pred_df = pd.DataFrame(
        {
            "document": document_list,
            "token": token_id_list,
            "label_id": label_id_list,
        }
    )

    pred_df = pred_df.drop_duplicates(subset = ['document', 'token', 'label_id'],keep = 'first').reset_index(drop = True)

    return pred_df

def get_predictions(data, model):
    outputs = model(input_ids=data['input_ids'], token_type_ids=data['token_type_ids'], attention_mask=data['attention_mask'])
    logits = torch.nn.functional.softmax(outputs.logits, dim=-1)
    logits = logits.detach().numpy()
    predictions = np.argmax(logits, axis=-1)

    return predictions


def _compute_metrics_joined(data, model):
    """
    DO NOT USE
    """

    data.set_format(type='pt', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    preds = get_predictions(data, model)
    pred_df = _get_labeled_tokens(data, preds)
    gold_df = _get_labeled_tokens(data, data['labels'])

    for pred, gold in zip(pred_df['label_id'], gold_df['label_id']):
        if pred != gold:
            print(f'Predicted: {pred}, Gold: {gold}')
    return pred_df, gold_df




if __name__=='__main__':
    local_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(local_path, '../data/raw/train.json')
    model = AutoModelForTokenClassification.from_pretrained('zmilczarek/pii-detection-baseline-v0.1')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data = preprocess_data(data_path, tokenizer, model.config.label2id, keys_to_keep=['document'])
    data = data.select(range(10))
    preds, golds = _compute_metrics_joined(data, model)
    print(preds.head(10))
    print(golds.head(10))
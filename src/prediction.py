from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained(
    'roberta-base', add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(
    "zeinab-sheikhi/roberta-pii-detection")
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


def get_embeddings(example):
    inputs = {'input_ids': example['input_ids'],
              'attention_mask': example['attention_mask']}
    return model(**inputs)


def predict(preprocessed_data):
    print("getting the predictions...")
    outputs = preprocessed_data.map(get_embeddings, batched=True, batch_size=4)
    logits = torch.nn.functional.softmax(outputs['logits'], dim=-1)
    logits = (outputs['logits']).detach().numpy()
    predictions = np.argmax(logits, axis=-1)
    print(outputs)
    print(len(predictions))
    return predictions

from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained(
    'roberta-base', add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(
    "zeinab-sheikhi/roberta-pii-detection")


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
    return predictions

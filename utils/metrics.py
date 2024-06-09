import numpy as np
from seqeval.metrics import recall_score, precision_score, accuracy_score




def get_fbeta_score(precision, recall, beta=5.0):
    b2 = beta ** 2
    return (1 + b2) * ((precision * recall) / (b2 * precision + recall))


def compute_metrics(p, labels_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    fbeta_score = get_fbeta_score(precision, recall)

    results = {
        'recall': recall,
        'precision': precision,
        'fbeta_score': fbeta_score
    }
        
    return results
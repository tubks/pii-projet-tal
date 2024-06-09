import os
import torch

from itertools import chain
from collections import Counter






def get_default_weight(num_samples, samples_per_labels, label_ids, num_labels):
    """
    Calculates weights for each label based on the below formula:
                weight_for_class_i = total samples / (num_samples_in_class_i * num_classes)
       
        Params:
        ---------- 
        - num_samples : int
            total number of samples in the whole dataset

        - samples_per_labels : {int: int}
            number of samples for each label id

        - label_ids : list
            list of ids from label2id mapping
        
        - num_labels: int

        Returns:
        ---------- 
        `Dict`: int weight for each label id
    """
    
    weight_dict = {id: 0 for id in label_ids}
    for label, samples in samples_per_labels.items():
        weight = num_samples / (samples * num_labels)
        weight_dict[label] = weight
    return weight_dict


def get_weights_effective_num_of_samples(samples_per_labels, label_ids, beta=0.9):
    """
    Calculates weights for each label based on the below formula:
                weight_for_class_i = 1 - beta / (1 - beta**num_samples_in_class_i)
       
        Params:
        ---------- 
        - samples_per_labels : {int: int}
            number of samples for each label id

        - label_ids : list
            list of ids from label2id mapping
        
        - beta: float; ususally set to 0.9, 0.99, or 0.999

        Returns:
        ---------- 
        `Dict`: int weight for each label id
    """

    weight_dict = {id: 0 for id in label_ids}

    for label, samples in samples_per_labels.items():
        effective_number = (1 - beta ** samples) / (1 - beta)
        weight = 1 / effective_number
        weight_dict[label] = weight
    return weight_dict


def get_weights_inverse_num_samples(samples_per_labels, label_ids, power=1):
    """
    Calculates weights for each label based on the below formula:
                weight_for_class_i = 1 / (num_samples_in_class_i)**power
       
        Params:
        ---------- 
        - samples_per_labels : {int: int}
            number of samples for each label id

        - label_ids : list
            list of ids from label2id mapping
        
        - power: int; power=1 for Inverse of Number of Sample and power=2 for Inverse of Square Root of Number of Samples

        Returns:
        ---------- 
        `Dict`: int weight for each label id
    """

    weight_dict = {id: 0 for id in label_ids}
    
    for label, samples in samples_per_labels.items():
        weight = 1 / samples ** power
        weight_dict[label] = weight
    
    return weight_dict


def compute_weights(labels, label_ids, method="default", beta=0.9):
    """
    Calculates weights for each label
       
        Params:
        ---------- 
        - labels: list of sublists
        - label_ids: dict from ids to labels
        - method: str; 
            "inverse" for get_weights_inverse_num_samples with power=1
            "inverse_square" for get_weights_inverse_num_samples with power=2
            "effective" for get_weights_effective_num_of_samples with beta=beta
        - beta: float

        Returns:
        ---------- 
        `Dict`: int weight for each label id
    """
    labels = list(chain.from_iterable(labels))  # flatten list of sublists
    num_samples = len(labels)
    samples_per_labels = Counter(labels)
    
    if method == "inverse":
        weight_dict = get_weights_inverse_num_samples(samples_per_labels, label_ids, power=1)
    elif method == "inverse_square":
        weight_dict = get_weights_inverse_num_samples(samples_per_labels, label_ids, power=2)
    elif method == "effective":
        weight_dict = get_weights_effective_num_of_samples(samples_per_labels, label_ids, beta=beta)
    else: 
        weight_dict = get_default_weight(num_samples, samples_per_labels, label_ids)
    
    return {key: weight_dict[key] for key in sorted(weight_dict)}


def weight_to_tensor(weight_dict):
    """
    convert weight_dict.values to a tensor
    """
    return torch.tensor(list(weight_dict.values()))


def save_model(model_name, target_dir):
    return os.path.join(target_dir, model_name)
     

"""
A module containing methods that preprocess the data before training the model.
"""
import torch 
from transformers import BertTokenizer, BertForTokenClassification
import os
import pandas as pd


def tokenize_and_preserve_labels(text, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []
    for word, label in zip(text, text_labels):
        #tokenizes the word using BERT's subword tokenizer
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        #adds the same label to all the subwords of the word
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def chunker(tokens, labels, max_len=512, pad_text = 0.0,pad_labels="PAD", sliding_window_size=0):
    """
    goal: chunk a text and encode it using a tokenizer
    takens in:
        tokens: list of tokens 
        labels: list of labels
        max_len: int
        pad_text: any type, by default: 0.0
        pad_labels: any type, by default: "PAD" 
        sliding_window_size: int, by default: 0. 
            If 0, the function will produce chunks without overlapping.
            If Int, the function will use a sliding window of size sliding_window_size.
    outputs:
        chunked_tokens: list of chunked tokens of the text
        chunked_labels: list of chunked labels of the text
    """
    
    assert len(tokens)==len(labels)
    chunked_tokens = []
    chunked_labels = []
    for pos in range(0,len(tokens),max_len):
        pad_length = max_len - (len(tokens) % max_len)
        #moving the start position back to make the chunk overlap with the previous one
        start_pos = max(0, pos - sliding_window_size)
        tokens_chunk = tokens[start_pos:start_pos+max_len]
        labels_chunk = labels[start_pos:start_pos+max_len]
        if len(tokens_chunk) != 512:
            tokens_chunk.extend(pad_length * [pad_text])
            labels_chunk.extend(pad_length * [pad_labels])
        chunked_tokens.append(tokens_chunk)
        chunked_labels.append(labels_chunk)
    return chunked_tokens,chunked_labels


def chunk_text_and_labels(text_and_labels,sliding_window_size=0):
    """
    goal: chunk a corpus of texts and encode it using a tokenizer
    Takes in:
        text_and_labels: list of tuples (tokens,labels)
    outputs:
        chunked_tokens: list of chunked tokens of all texts
        chunked_labels: list of chunked labels of all texts
    """
    all_chunked_tokens, all_chunked_labels = [],[]
    for tokens, labels in text_and_labels:
        chunked_tokens,chunked_labels = chunker(tokens,labels,sliding_window_size=sliding_window_size)
        all_chunked_tokens.extend(chunked_tokens)
        all_chunked_labels.extend(chunked_labels)
    return all_chunked_tokens,all_chunked_labels


if __name__ == "__main__":
    model_name = 'bert-base-uncased'
    data_path = os.path.join(os.path.dirname(__file__),'..','data', 'raw', 'train.json')
    tokenizer = BertTokenizer.from_pretrained(model_name)
    data = pd.read_json(data_path)
    tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs, tokenizer) for sent, labs in zip(data['tokens'].head(100), data['labels'])]
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    chunked_tokens,chunked_labels=chunk_text_and_labels(tokenized_texts_and_labels,sliding_window_size=15)
    print(chunked_tokens[0][-15:],chunked_tokens[1][:15])
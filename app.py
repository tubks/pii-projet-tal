import streamlit as st
import spacy
import spacy_wrap
from spacy_streamlit import visualize_ner
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from src.classes.displacy_input import displacy_input
###
# ----------------------------------------------------------
#
# The purpose of thi script is to create a web application that
#   - allows the user to input an essay
#   - detects personal information in the essay
#   - highlights the detected personal information

# To use the script, run it in the terminal with the command:
#   streamlit run app.py
#
# ----------------------------------------------------------
###

# setting up spacy

# nlp = spacy.load("en_core_web_sm")

nlp = spacy.blank("en")

# specify model from the hub
config = {"model": {"name": "zmilczarek/pii-detection-roberta-v2"},
          "predictions_to": ["ents"]}  # forced to be named entity recognition, if left out it will be estimated from the labels

# add it to the pipe
nlp.add_pipe("token_classification_transformer", config=config)

# getting the right model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     "roberta-base", add_prefix_space=True)
# model_from_huggingface = AutoModelForTokenClassification.from_pretrained(
#     'zmilczarek/pii-detection-roberta-v2')

# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# example essay
example_essay = "My name is John Doe and I live in the United States of America. I am 25 years old and I work as a software engineer. My phone number is 123-456-7890 and my email address is jdoe@gmail.com"

# the field where the user can input their essay
input_text = st.text_area("Enter your essay", example_essay)

button_pressed = st.button("Submit")

# defining what happens when the button is pressed
if button_pressed:
    # inputs = tokenizer(input_text, return_tensors="pt")
    # outputs = model_from_huggingface(**inputs)
    # predictions = outputs.logits.argmax(-1)
    doc = nlp(input_text)

    print(doc.ents)
    # predictions = torch.argmax(outputs.logits, dim=2)

    # Convert indices to entity labels
    # predicted_labels = [model_from_huggingface.config.id2label[idx.item()]
    #                     for idx in predictions[0]]

    # # Decode the tokens
    # tokenized_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # # Process the tokens and labels for displaCy
    # entities = []
    # start = 0

    # for token, label in zip(tokenized_text, predicted_labels):
    #     token = token.lstrip("Ġ")
    #     end = start + len(token)

    #     if label != "O":  # 'O' means outside any named entity
    #         if entities and entities[-1]["label"] == label:
    #             entities[-1]["end"] = end
    #         else:
    #             entities.append({
    #                 "start": start,
    #                 "end": end,
    #                 "label": label
    #             })
    #     start = end
    # displacy_doc = displacy_input(input_text, entities)

    # # Render the output using displaCy
    # print("ENT", entities)
    # print("OBJECT: ", list(model_from_huggingface.config.id2label.values()))

    visualize_ner(doc, show_table=False)
    # st.write(predictions)
    # visualize_ner(nlp(input_text), labels=nlp.get_pipe(
    #     "ner").labels, show_table=False)

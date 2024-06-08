import streamlit as st
import spacy
from spacy_streamlit import visualize_ner

#from src.classes.displacy_input import displacy_input
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

#
# saving the environment in the pip format, without weird outputs : pip list --format=freeze > requirements.txt
#


# setting up spacy

nlp = spacy.blank("en")

# specify model name from the hub, type of task and list of labels
config = {"model": {"name": "zmilczarek/pii-detection-roberta-v2"},
          "predictions_to": ["ents"],
          "labels": ['B-NAME_STUDENT', 'B-EMAIL', 'B-USERNAME', 'B-ID_NUM', 'B-PHONE_NUM', 'B-URL_PERSONAL', 'B-STREET_ADDRESS', 'I-NAME_STUDENT', 'I-EMAIL', 'I-USERNAME', 'I-ID_NUM', 'I-PHONE_NUM', 'I-URL_PERSONAL', 'I-STREET_ADDRESS', 'O']}  # forced to be named entity recognition, if left out it will be estimated from the labels

# create a pipeline with this config
nlp.add_pipe("token_classification_transformer", config=config)

# example essay
example_essay = "My name is John Doe and I live in the United States of America. I am 25 years old and I work as a software engineer. My phone number is 123-456-7890 and my email address is jdoe@gmail.com"

# the field where the user can input their essay
input_text = st.text_area("Enter your essay", example_essay)

button_pressed = st.button("Submit")

# defining what happens when the button is pressed
if button_pressed:
    doc = nlp(input_text)
    # print(doc.ents)
    visualize_ner(doc, show_table=True)

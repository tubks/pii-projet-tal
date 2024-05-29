import streamlit as st
import spacy
from spacy_streamlit import visualize_ner
from transformers import AutoModelForTokenClassification, AutoTokenizer


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

nlp = spacy.load("en_core_web_sm")

# getting the right model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
model_from_huggingface = AutoModelForTokenClassification.from_pretrained('zmilczarek/pii-detection-roberta-v2')



# example essay
example_essay = "My name is John Doe and I live in the United States of America. I am 25 years old and I work as a software engineer. My phone number is 123-456-7890 and my email address is jdoe@gmail.com"

# the field where the user can input their essay
input_text = st.text_area("Enter your essay",example_essay)

button_pressed = st.button("Submit")

# defining what happens when the button is pressed
if button_pressed:
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model_from_huggingface(**inputs)
    predictions = outputs.logits.argmax(-1)
    st.write(predictions)
    visualize_ner(nlp(input_text), labels=nlp.get_pipe("ner").labels, show_table=False)


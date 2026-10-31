import streamlit as st
from transformers import pipeline

# App Title and Description
st.title("AI Grammar & Text Correction App")
st.write("Enter a sentence below to check and correct grammatical errors using your fine-tuned Hugging Face model.")

# Load the model with caching so it only loads once
@st.cache_resource
def load_model():
    # Replace 'your-username/my-grammar-model' with your actual Hugging Face repo ID
    repo_id = "shahab1/my-grammar-model"
    return pipeline("text2text-generation", model=repo_id)

with st.spinner("Loading model from Hugging Face... Please wait."):
    corrector = load_model()

# User input text box
user_input = st.text_area("Enter your sentence:", "Me and him went to the market and buyed some fresh fruits.")

# Button to trigger model inference
if st.button("Correct Sentence"):
    if user_input.strip():
        with st.spinner("Processing..."):
            # Run inference
            result = corrector(user_input)
            
            # Display output
            st.success("Completed!")
            st.subheader("Corrected Result:")
            st.write(result[0]['generated_text'])
    else:
            st.warning("Please enter some text first.")
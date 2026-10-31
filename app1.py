import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

PREFIX = "Fix grammar: "

st.title("AI Grammar & Text Correction App")
st.write(
    "Enter a sentence below to check and correct grammatical errors using"
    " your fine-tuned Hugging Face model."
)


@st.cache_resource
def load_model():
    repo_id = "shahab1/my-grammar-model"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


with st.spinner("Loading model from Hugging Face... Please wait."):
    try:
        corrector = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        corrector = None

user_input = st.text_area(
    "Enter your sentence:", "Me and him went to the store yesterday."
)

if st.button("Correct Grammar"):
    if not user_input.strip():
        st.warning("Please enter some text to correct.")
    elif corrector is None:
        st.error("Model is not loaded properly. Please check your configuration or logs.")
    else:
        with st.spinner("Correcting..."):
            results = corrector(
                PREFIX + user_input.strip(),
                max_new_tokens=128,
                num_beams=4,
            )
            corrected_text = results[0]["generated_text"]

            st.success("Corrected Sentence:")
            st.write(corrected_text)

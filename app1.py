import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

REPO_ID = "shahab1/my-grammar-model"
BASE_MODEL = "google/flan-t5-base"
PREFIX = "Fix grammar: "
MAX_LEN = 128

st.set_page_config(page_title="AI Grammar Tutor", page_icon="✍️")
st.title("AI Grammar Tutor")
st.write(
    "Enter a sentence below to check and correct grammatical errors using"
    " your fine-tuned Flan-T5 model."
)


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(REPO_ID)
    model.to(device).eval()
    return tokenizer, model, device


def correct(text, tokenizer, model, device):
    inputs = tokenizer(
        PREFIX + text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    ).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True)


with st.spinner("Loading model... Please wait."):
    try:
        tokenizer, model, device = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        tokenizer = model = device = None

user_input = st.text_area(
    "Enter your sentence:", "She don't like to eats apples for breakfast."
)

if st.button("Correct Grammar"):
    if not user_input.strip():
        st.warning("Please enter some text to correct.")
    elif model is None:
        st.error("Model is not loaded properly. Please check your configuration or logs.")
    else:
        with st.spinner("Correcting..."):
            fixed = correct(user_input.strip(), tokenizer, model, device)

        st.success("Corrected Sentence:")
        st.write(fixed)

        if fixed.strip() == user_input.strip():
            st.info("No changes needed. The sentence was already correct.")
        else:
            st.caption("The sentence was changed.")

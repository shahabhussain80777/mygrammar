import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# App title and description
st.title("AI Grammar & Text Correction App")
st.write(
    "Enter a sentence below to check and correct grammatical errors using"
    " your fine-tuned Hugging Face model."
)


@st.cache_resource
def load_model():
  # Your Hugging Face repository ID
  repo_id = "shahab1/my-grammar-model"

  # Explicitly load tokenizer and model to prevent pipeline deprecation/tokenizer crashes
  tokenizer = AutoTokenizer.from_pretrained(repo_id)
  model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

  # Return a text-generation pipeline configured with your loaded components
  return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


# Load model with a loading spinner
with st.spinner("Loading model from Hugging Face... Please wait."):
  try:
    corrector = load_model()
  except Exception as e:
    st.error(f"Error loading model: {e}")
    corrector = None

# User input text box
user_input = st.text_area(
    "Enter your sentence:", "Me and him went to the store yesterday."
)

if st.button("Correct Grammar"):
  if not user_input.strip():
    st.warning("Please enter some text to correct.")
  elif corrector is None:
    st.error(
        "Model is not loaded properly. Please check your configuration or logs."
    )
  else:
    with st.spinner("Correcting..."):
      # Generate correction
      results = corrector(
          user_input, max_length=128, num_return_sequences=1
      )
      corrected_text = results[0]["generated_text"]

      # Display output
      st.success("Corrected Sentence:")
      st.write(corrected_text)

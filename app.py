import streamlit as st
from transformers import pipeline

# App Header and Instructions
st.title("Apple Leaf Disease Detection & Grammar Checker")
st.write(
    "Enter text or details below to run inference using your fine-tuned Hugging Face model."
)

# Replace this with your actual Hugging Face repository ID
# Example: "Shahab/apple_disease_identification"
REPO_ID = "your-username/your-repo-name"


@st.cache_resource
def load_model():
    """Load the model pipeline from Hugging Face with caching."""
    # Using 'text2text-generation' or 'text-generation' depending on your model task.
    # If your model is a causal language model (like GPT), change task to "text-generation".
    return pipeline("text2text-generation", model=REPO_ID)


# Try loading the model with error handling for the UI
try:
    with st.spinner("Loading model from Hugging Face... Please wait."):
        corrector = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(
        f"Failed to load model from repository `{REPO_ID}`. Error details: {e}"
    )
    st.stop()

# User Input Form / Widget
user_input = st.text_area("Enter your text here:", "")

if st.button("Run Inference"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Processing..."):
            try:
                # Run prediction
                results = corrector(user_input)
                # Display output
                st.subheader("Output Result:")
                st.write(results[0]["generated_text"])
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")

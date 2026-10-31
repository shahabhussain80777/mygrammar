@st.cache_resource
def load_model():
    repo_id = "shahab1/my-grammar-model"
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


import json, os, tempfile
from huggingface_hub import snapshot_download

@st.cache_resource
def load_model():
    repo_id = "shahab1/my-grammar-model"
    path = snapshot_download(repo_id, local_dir=os.path.join(tempfile.gettempdir(), "grammar_model"))

    cfg_file = os.path.join(path, "tokenizer_config.json")
    with open(cfg_file) as f:
        cfg = json.load(f)
    if isinstance(cfg.get("extra_special_tokens"), list):
        cfg.pop("extra_special_tokens")
    with open(cfg_file, "w") as f:
        json.dump(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID = "Xtern0723/bert-imdb-sentiment"

MAX_LENGTH = 128       # Must match training (tokenize_fn max_length=128)
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}
COLORS = {"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444"}
ICONS  = {"POSITIVE": "✅", "NEGATIVE": "❌"}

# ── Model loader (cached across reruns — loads only ONCE per session) ─────────
# @st.cache_resource vs @st.cache_data:
#   cache_data   → serialises/deserialises return value (works for DataFrames, dicts)
#   cache_resource → keeps the Python object itself in memory (required for PyTorch models)
#   Using cache_data on a model would crash: PyTorch tensors can't be pickled via st's protocol
@st.cache_resource(show_spinner=False)
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()   # turns off dropout — critical: keeps predictions deterministic at inference
    return tokenizer, model

# ── Run inference ─────────────────────────────────────────────────────────────
def predict(text: str, tokenizer, model):
    """
    Tokenize → forward pass → softmax → return (label, confidence, all_probs)
    torch.no_grad() disables autograd tape — saves memory and speeds up inference.
    No gradients needed at inference; we're not calling .backward().
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits           # shape: (1, 2)

    probs    = F.softmax(logits, dim=-1).squeeze()  # raw logits → probabilities
    pred_idx = torch.argmax(probs).item()
    return LABELS[pred_idx], probs[pred_idx].item(), probs

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 BERT Sentiment Analyzer")
st.markdown(
    "Fine-tuned **`bert-base-uncased`** on IMDb movie reviews.  \n"
    "Classifies any text as **Positive** or **Negative** with a confidence score."
)
st.divider()

# Load model with status indicator
with st.status("Loading fine-tuned BERT from HuggingFace Hub…", expanded=False) as status:
    try:
        tokenizer, model = load_model(MODEL_ID)
        status.update(label="Model loaded ✅", state="complete", expanded=False)
    except Exception as e:
        status.update(label="Model load failed ❌", state="error")
        st.error(
            f"Could not load model `{MODEL_ID}`.  \n"
            f"Make sure you've pushed your fine-tuned model to HuggingFace Hub first.  \n\n"
            f"**Error:** `{e}`"
        )
        st.stop()

# Text input
text_input = st.text_area(
    "Enter a review or any text:",
    placeholder=(
        "e.g. This product completely changed how our team handles support tickets. "
        "Couldn't imagine going back."
    ),
    height=150,
)

analyze_btn = st.button("Analyze Sentiment →", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if analyze_btn:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running inference…"):
            label, confidence, probs = predict(text_input.strip(), tokenizer, model)

        color = COLORS[label]
        icon  = ICONS[label]

        st.divider()

        # Primary result
        col_label, col_conf = st.columns(2)
        with col_label:
            st.markdown("#### Prediction")
            st.markdown(
                f"<h2 style='color:{color}; margin-top:0'>{icon} {label}</h2>",
                unsafe_allow_html=True,
            )
        with col_conf:
            st.markdown("#### Confidence")
            st.markdown(
                f"<h2 style='color:{color}; margin-top:0'>{confidence:.1%}</h2>",
                unsafe_allow_html=True,
            )

        # Confidence bar
        st.progress(confidence)

        # Both class probabilities
        st.divider()
        st.markdown("**Class probabilities**")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.metric(label="🟢 Positive", value=f"{probs[1].item():.1%}")
        with p_col2:
            st.metric(label="🔴 Negative", value=f"{probs[0].item():.1%}")

        # Token count info
        token_count = len(tokenizer.encode(text_input.strip()))
        if token_count > MAX_LENGTH:
            st.info(
                f"⚠️ Input was **{token_count} tokens** — truncated to {MAX_LENGTH} "
                f"(model's training max). Result reflects the first {MAX_LENGTH} tokens only."
            )
        else:
            st.caption(f"Token count: {token_count} / {MAX_LENGTH}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with 🤗 Transformers · Fine-tuned on 5,000 IMDb reviews · "
    "Model: `bert-base-uncased` · [GitHub](https://github.com/codewith-krishh)"
)
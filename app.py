from pathlib import Path
import html
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

st.set_page_config(
    page_title="NewsLens | Category Classifier",
    page_icon="N",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
CATEGORIES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
CATEGORY_COLORS = {
    "World": "#2f6fed",
    "Sports": "#0c9b72",
    "Business": "#d47a17",
    "Sci/Tech": "#b94b8d",
}

DEMO_ROWS = [
    ("UN summit opens with talks on climate cooperation", "World"),
    ("Leaders agree to strengthen international trade ties", "World"),
    ("Diplomats seek a peaceful solution to the border dispute", "World"),
    ("Global aid agencies respond after the earthquake", "World"),
    ("Championship final ends with a dramatic last-minute goal", "Sports"),
    ("The tennis star advances after a three-set victory", "Sports"),
    ("Coach announces the squad for the national tournament", "Sports"),
    ("Olympic athletes begin training ahead of the summer games", "Sports"),
    ("Markets rise as investors react to the latest earnings report", "Business"),
    ("The central bank keeps interest rates unchanged this month", "Business"),
    ("Retail company opens new stores and increases quarterly revenue", "Business"),
    ("Oil prices fall as analysts revise their economic forecast", "Business"),
    ("Researchers release a faster processor for artificial intelligence", "Sci/Tech"),
    ("The new smartphone includes an improved camera and battery", "Sci/Tech"),
    ("Scientists discover a promising method for clean energy storage", "Sci/Tech"),
    ("Software developers build an open source security update", "Sci/Tech"),
]


def make_demo_data():
    return pd.DataFrame(DEMO_ROWS, columns=["Text", "Class Label"])


def preprocess_text(value):
    """Apply the same lightweight cleaning to every article before training."""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_news(frame):
    """Convert an AG News-style frame into clean, model-ready data."""
    columns = {str(column).strip().lower(): column for column in frame.columns}
    label_column = (
        columns.get("class label")
        or columns.get("class index")
        or columns.get("label")
        or columns.get("category")
    )
    if label_column is None:
        raise ValueError("The CSV needs a 'Class Index', 'Class Label', 'label', or 'category' column.")

    if "text" in columns:
        text = frame[columns["text"]].fillna("").astype(str)
    elif "title" in columns and "description" in columns:
        text = (
            frame[columns["title"]].fillna("").astype(str)
            + " "
            + frame[columns["description"]].fillna("").astype(str)
        )
    else:
        raise ValueError("The CSV needs 'Text' or both 'Title' and 'Description' columns.")

    labels = frame[label_column]
    if pd.api.types.is_numeric_dtype(labels):
        numeric_labels = pd.to_numeric(labels, errors="coerce")
        label_values = set(numeric_labels.dropna().astype(int).unique())
        if label_values.issubset(set(CATEGORIES)):
            labels = numeric_labels.map(CATEGORIES)
        else:
            labels = numeric_labels.map(lambda value: CATEGORIES.get(int(value) - 1))
    else:
        label_names = {
            "world": "World",
            "sports": "Sports",
            "business": "Business",
            "sci/tech": "Sci/Tech",
            "science/technology": "Sci/Tech",
            "technology": "Sci/Tech",
        }
        labels = labels.astype(str).str.strip().str.lower().map(label_names)

    result = pd.DataFrame({"Text": text.map(preprocess_text), "Class Label": labels})
    before = len(result)
    result = result.dropna().drop_duplicates(subset=["Text"])
    result = result[(result["Text"].str.len() >= 20) & result["Class Label"].isin(CATEGORIES.values())]
    result = result.reset_index(drop=True)
    result["Class Index"] = result["Class Label"].map({name: index for index, name in CATEGORIES.items()})
    if result.empty:
        raise ValueError("No usable rows were found for the four AG News categories.")
    return result, {"input_rows": before, "output_rows": len(result), "removed_rows": before - len(result)}


def load_ag_news(data_source=None):
    if data_source is not None:
        frame = pd.read_csv(data_source)
        is_demo = False
    else:
        candidates = [
            PROJECT_DIR / "train.csv",
            PROJECT_DIR / "data/train.csv",
            PROJECT_DIR / "dataset/train.csv",
        ]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            frame = make_demo_data()
            is_demo = True
        else:
            frame = pd.read_csv(source)
            is_demo = False
    result, stats = preprocess_news(frame)
    output_path = PROJECT_DIR / "data/preprocessed_news.csv"
    output_path.parent.mkdir(exist_ok=True)
    result.to_csv(output_path, index=False)
    return result, is_demo, stats


@st.cache_resource(show_spinner="Training the neural network...")
def train_model(texts, labels):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=12000,
        min_df=1,
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(texts)
    classifier = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=min(32, len(texts)),
        learning_rate_init=0.001,
        max_iter=250,
        early_stopping=len(texts) >= 50,
        random_state=42,
    )
    classifier.fit(features, labels)
    return vectorizer, classifier


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #17212b; --muted: #71808b; --paper: #f5f6f3; --line: #dfe4df; --lime: #c7ef6b; --orange: #f08b5b; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: var(--ink); border-right: 1px solid #293744; }
    [data-testid="stSidebar"] * { color: #f5f6f3; }
    [data-testid="stSidebar"] .stFileUploader { background: #202d38; border: 1px solid #3b4a56; border-radius: 8px; padding: .25rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--ink); }
    p, label, input, textarea, button { font-family: 'DM Sans', sans-serif; }
    .topline { display: flex; justify-content: space-between; align-items: center; padding: .8rem 0 1.2rem; border-bottom: 1px solid var(--line); }
    .brand { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; }
    .brand-mark { display: inline-block; background: var(--lime); color: var(--ink); padding: .25rem .45rem; border-radius: 4px; margin-right: .45rem; }
    .status { color: #397154; font-size: .82rem; font-weight: 600; }
    .hero { padding: 2.3rem 0 1.7rem; }
    .eyebrow { color: #d66a37; font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { font-size: clamp(2.3rem, 4vw, 4rem); line-height: 1.02; max-width: 720px; margin: .55rem 0 .7rem; }
    .hero p { color: var(--muted); font-size: 1rem; max-width: 560px; line-height: 1.55; margin: 0; }
    .section-label { color: var(--muted); font-size: .76rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .45rem; }
    .panel { background: white; border: 1px solid var(--line); border-radius: 10px; padding: 1.25rem; box-shadow: 0 8px 24px rgba(23,33,43,.04); }
    .panel h3 { margin-top: 0; }
    .result { background: var(--ink); color: #f5f6f3; padding: 1.35rem; border-radius: 8px; margin-top: 1rem; border-left: 5px solid var(--lime); }
    .result-label { color: #aab7bf; font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; }
    .result-category { font-family: 'Space Grotesk', sans-serif; font-size: 2.45rem; margin: .35rem 0 .15rem; }
    .confidence { color: var(--lime); font-size: .9rem; }
    .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .75rem; margin-top: .7rem; }
    .metric { background: white; border: 1px solid var(--line); border-radius: 8px; padding: .9rem; color: var(--muted); font-size: .82rem; }
    .metric strong { display: block; color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; margin-bottom: .2rem; }
    [data-testid="stTextArea"] textarea { background: #fbfcfa; border: 1px solid #cfd8d1; border-radius: 8px; color: var(--ink); font-size: 1rem; line-height: 1.55; }
    [data-testid="stTextArea"] textarea:focus { border-color: #9abe42; box-shadow: 0 0 0 1px #9abe42; }
    .stButton > button[kind="primary"] { background: var(--ink); border: 0; border-radius: 7px; color: white; font-weight: 700; min-height: 2.8rem; }
    .stButton > button[kind="primary"]:hover { background: #2a3a47; color: var(--lime); }
    @media (max-width: 900px) { .hero { padding-top: 1.5rem; } .hero h1 { font-size: 2.5rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## <span class='brand-mark'>N</span> NewsLens", unsafe_allow_html=True)
    st.caption("Editorial intelligence for fast, consistent news tagging.")
    st.markdown("### Dataset")
    uploaded_file = st.file_uploader("Optional AG News CSV", type=["csv"])
    st.caption("Cleaning, validation, deduplication, and export happen automatically.")
    st.markdown("### Label set")
    for category in CATEGORIES.values():
        st.markdown(f"- {category}")
    st.markdown("---")
    st.caption("The notebook label mapping is preserved: 1 World, 2 Sports, 3 Business, 4 Sci/Tech.")

st.markdown(
    '<div class="topline"><div class="brand"><span class="brand-mark">N</span>NEWS CLASSIFIER</div><div class="status">● MODEL ONLINE</div></div><section class="hero"><div class="eyebrow">Four-category neural classifier</div><h1>Turn a headline into a clear category.</h1><p>Paste a headline or article excerpt. The model reads the signal, ranks the possibilities, and returns its strongest classification.</p></section>',
    unsafe_allow_html=True,
)

try:
    data, is_demo, preprocessing_stats = load_ag_news(uploaded_file)
    vectorizer, classifier = train_model(tuple(data["Text"]), tuple(data["Class Label"]))
except Exception as error:
    st.error(str(error))
    st.stop()

left, right = st.columns([1.4, 1], gap="large")
with left:
    st.markdown('<div class="panel"><div class="section-label">Live classification</div><h3>Classify an article</h3>', unsafe_allow_html=True)
    article = st.text_area(
        "Article text",
        value="",
        height=220,
        placeholder="Example: Researchers unveil a new battery design that charges in minutes...",
        label_visibility="collapsed",
    )
    classify = st.button("Classify story", type="primary", use_container_width=True)

    if classify:
        if not article.strip():
            st.warning("Add a headline or article excerpt first.")
        else:
            probabilities = classifier.predict_proba(vectorizer.transform([article]))[0]
            order = np.argsort(probabilities)[::-1]
            predicted = classifier.classes_[order[0]]
            confidence = probabilities[order[0]]
            st.session_state["last_prediction"] = {
                "category": predicted,
                "confidence": float(confidence),
                "probabilities": [(classifier.classes_[index], float(probabilities[index])) for index in order[1:]],
            }

    prediction = st.session_state.get("last_prediction")
    if prediction:
            st.markdown(
                f'<div class="result"><div class="result-label">Predicted category</div><div class="result-category">{prediction["category"]}</div><div class="confidence">{prediction["confidence"]:.1%} model confidence</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("#### Other signals")
            for category, probability in prediction["probabilities"]:
                st.progress(probability, text=f"{category}  {probability:.1%}")
    else:
        st.markdown('<div class="result" style="border-left-color:#8b98a0"><div class="result-label">Prediction stage</div><div class="result-category">Awaiting a headline</div><div class="confidence" style="color:#aab7bf">Your classification will appear here.</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="section-label">System overview</div><h3>Model snapshot</h3><div class="metric-grid">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric"><strong>{len(data):,}</strong>training articles</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric"><strong>{preprocessing_stats["removed_rows"]:,}</strong>rows removed</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric"><strong>{len(vectorizer.vocabulary_):,}</strong>TF-IDF features</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric"><strong>2 layers</strong>ReLU neural network</div></div>', unsafe_allow_html=True)
    if is_demo:
        st.info("Demo mode is active. Add an AG News train.csv in the project folder or upload one in the sidebar for a dataset-trained model.")
    else:
        st.success("Using your AG News dataset.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Quick test headlines")
examples = [
    "The football club signs a young striker before the new season",
    "Chip makers announce a breakthrough in processor design",
    "Investors welcome stronger quarterly profits from the retailer",
]
for example in examples:
    st.caption(example)

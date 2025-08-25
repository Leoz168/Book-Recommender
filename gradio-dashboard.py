# gradio_dashboard_relaxed.py
import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# LC compatibility shim
try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

import gradio as gr
print("Gradio version:", getattr(gr, "__version__", "unknown"))

# -------------------------
# Setup
# -------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGING_FACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN or HUGGING_FACE_TOKEN in your .env")

emb = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)

BOOKS_CSV = "books_with_emotions.csv"
PERSIST_DIR = "chroma_books"  # persistent index directory

# -------------------------
# Load data
# -------------------------
books = pd.read_csv(BOOKS_CSV)

# Required columns sanity
required_cols = {
    "isbn13", "title", "authors", "description", "thumbnail",
    "simplified_categories", "tagged_description"
}
missing = required_cols - set(books.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# Normalize types/values
books["isbn13"] = pd.to_numeric(books["isbn13"], errors="coerce").astype("Int64")
books["authors"] = books["authors"].fillna("Unknown")
books["description"] = books["description"].fillna("")
books["tagged_description"] = books["tagged_description"].fillna("")
books["simplified_categories"] = books["simplified_categories"].fillna("Other")

for col in ["average_rating", "ratings_count", "num_pages", "published_year",
            "joy", "surprise", "fear", "anger", "sadness", "neutral"]:
    if col in books.columns:
        books[col] = pd.to_numeric(books[col], errors="coerce")

books["large_thumbnail"] = np.where(
    books["thumbnail"].isna() | (books["thumbnail"] == ""),
    "cover-not-found.png",
    books["thumbnail"] + "&fife=w800",
)

# Tone columns
TONE_COLS = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
    "Neutral (calm)": "neutral",
}

# UI choices
categories = ["All"] + sorted([str(c) for c in books["simplified_categories"].dropna().unique()])
tones = ["All"] + list(TONE_COLS.keys())

# Filter bounds
def _safe_min(series, default):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return int(s.min()) if len(s) else default

def _safe_max(series, default):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return int(s.max()) if len(s) else default

YEAR_MIN = _safe_min(books.get("published_year", pd.Series(dtype=float)), 1900)
YEAR_MAX = _safe_max(books.get("published_year", pd.Series(dtype=float)), 2025)
PAGES_MIN = _safe_min(books.get("num_pages", pd.Series(dtype=float)), 1)
PAGES_MAX = _safe_max(books.get("num_pages", pd.Series(dtype=float)), 1500)

# -------------------------
# Build vector docs from CSV
# -------------------------
def clean_tagged_text(s: str) -> str:
    """Strip a leading ISBN and delimiter from tagged_description."""
    s = str(s or "").strip()
    m = re.match(r'^\s*"?(\d{10,13})"?\s*[,|\t;:\- ]\s*(.+)$', s)
    return m.group(2).strip() if m else s

def docs_from_csv(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    for _, row in df.iterrows():
        if pd.isna(row["isbn13"]):
            continue
        isbn = int(row["isbn13"])
        text = clean_tagged_text(row.get("tagged_description", "")) or str(row.get("description", ""))
        text = text.strip()
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={"isbn13": isbn}))
    return docs

documents = docs_from_csv(books)

# -------------------------
# Build / load vector store
# -------------------------
# Using persist_directory automatically persists in langchain_chroma; no .persist() method.
if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    db_books = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
else:
    db_books = Chroma.from_documents(documents, embedding_function=emb, persist_directory=PERSIST_DIR)

# -------------------------
# Retrieval
# -------------------------
def retrieve_semantic_recs(
    query: str,
    category: str | None,
    tone: str | None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
    tone_weight: float = 0.35,  # 0 = ignore tone, 1 = tone only
    min_rating: float = 0.0,
    year_range: tuple[int, int] = (YEAR_MIN, YEAR_MAX),
    pages_range: tuple[int, int] = (PAGES_MIN, PAGES_MAX),
    sort_by: str = "Best match",
):
    # Base set
    if not query.strip():
        df = books.copy()
    else:
        isbn_scores = []
        k = max(initial_top_k, final_top_k)
        # Try multiple scoring APIs for compatibility across versions
        try:
            recs = db_books.similarity_search_with_score(query, k=k)  # (Document, distance) lower=better
            for doc, dist in recs:
                isbn = (doc.metadata or {}).get("isbn13")
                if isbn is not None:
                    isbn_scores.append((int(isbn), float(dist)))
        except Exception:
            try:
                recs = db_books.similarity_search_with_relevance_scores(query, k=k)  # (Document, score) higher=better
                for doc, rel in recs:
                    isbn = (doc.metadata or {}).get("isbn13")
                    if isbn is not None:
                        isbn_scores.append((int(isbn), 1.0 - float(rel)))  # pseudo-distance
            except Exception:
                docs = db_books.similarity_search(query, k=k)
                for i, d in enumerate(docs):
                    isbn = (d.metadata or {}).get("isbn13")
                    if isbn is not None:
                        isbn_scores.append((int(isbn), float(i)))

        if not isbn_scores:
            # Fallback: naive contains on descriptions
            df = books[books["description"].str.contains(re.escape(query), case=False, na=False)].copy()
        else:
            isbns = [i for (i, _) in isbn_scores]
            df = books[books["isbn13"].isin(isbns)].copy()
            # Map distances to similarities in [0,1]
            best_dist = {}
            for isbn, dist in isbn_scores:
                best_dist[isbn] = min(dist, best_dist.get(isbn, dist))
            df["__dist"] = df["isbn13"].map(best_dist).fillna(1e9)
            d = df["__dist"].to_numpy()
            if len(d) > 1 and np.isfinite(d).any():
                d_min, d_max = np.nanmin(d), np.nanmax(d)
                sim = 1.0 - (d - d_min) / (d_max - d_min) if d_max > d_min else np.ones_like(d)
            else:
                sim = np.ones_like(d)
            df["__sim"] = sim

    if df.empty:
        return df

    # Category filter
    if category and category != "All":
        df = df[df["simplified_categories"] == category].copy()

    # Numeric filters
    if "average_rating" in df.columns:
        df = df[(df["average_rating"].fillna(0) >= float(min_rating))].copy()

    if "published_year" in df.columns:
        y0, y1 = int(year_range[0]), int(year_range[1])
        yr = pd.to_numeric(df["published_year"], errors="coerce")
        df = df[(yr.fillna(YEAR_MIN).astype(int) >= y0) & (yr.fillna(YEAR_MIN).astype(int) <= y1)].copy()

    if "num_pages" in df.columns:
        p0, p1 = int(pages_range[0]), int(pages_range[1])
        pg = pd.to_numeric(df["num_pages"], errors="coerce")
        df = df[(pg.fillna(PAGES_MIN).astype(int) >= p0) & (pg.fillna(PAGES_MIN).astype(int) <= p1)].copy()

    if df.empty:
        return df

    # Ensure a base similarity
    if "__sim" not in df.columns:
        df["__sim"] = 0.5  # neutral if no query

    # Tone blending (optional)
    if tone and tone in TONE_COLS:
        emo_col = TONE_COLS[tone]
        if emo_col in df.columns:
            e = df[emo_col].fillna(0).to_numpy(dtype=float)
            if len(e) > 1 and (np.nanmax(e) > np.nanmin(e)):
                e_norm = (e - np.nanmin(e)) / (np.nanmax(e) - np.nanmin(e))
            else:
                e_norm = np.zeros_like(e)
            df["__score"] = (1 - float(tone_weight)) * df["__sim"].to_numpy() + float(tone_weight) * e_norm
        else:
            df["__score"] = df["__sim"]
    else:
        df["__score"] = df["__sim"]

    # Sorting
    sort_by = str(sort_by or "Best match")
    if sort_by == "Best match":
        df.sort_values("__score", ascending=False, inplace=True)
    elif sort_by == "Highest rating" and "average_rating" in df.columns:
        df.sort_values(["average_rating", "__score"], ascending=[False, False], inplace=True)
    elif sort_by == "Most rated" and "ratings_count" in df.columns:
        df.sort_values(["ratings_count", "__score"], ascending=[False, False], inplace=True)
    elif sort_by == "Newest" and "published_year" in df.columns:
        df.sort_values(["published_year", "__score"], ascending=[False, False], inplace=True)
    elif sort_by == "Shortest" and "num_pages" in df.columns:
        df.sort_values(["num_pages", "__score"], ascending=[True, False], inplace=True)
    elif sort_by == "Longest" and "num_pages" in df.columns:
        df.sort_values(["num_pages", "__score"], ascending=[False, False], inplace=True)
    else:
        df.sort_values("__score", ascending=False, inplace=True)

    out = df.head(final_top_k).copy()
    for c in ["__dist", "__sim", "__score"]:
        if c in out.columns:
            out.drop(columns=[c], inplace=True, errors="ignore")
    out.reset_index(drop=True, inplace=True)
    return out

def format_results_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["title", "authors", "simplified_categories", "isbn13",
            "average_rating", "ratings_count", "published_year", "num_pages"]
    cols += [c for c in ["joy", "surprise", "fear", "anger", "sadness", "neutral"] if c in df.columns]
    show = [c for c in cols if c in df.columns]
    return df[show].rename(columns={"simplified_categories": "category"})

def rec_books(query, category, tone, init_k, final_k, tone_weight, min_rating,
              year_min, year_max, pages_min, pages_max, sort_by, show_table):
    df = retrieve_semantic_recs(
        query=query,
        category=category if category != "All" else None,
        tone=tone if tone != "All" else None,
        initial_top_k=int(init_k),
        final_top_k=int(final_k),
        tone_weight=float(tone_weight),
        min_rating=float(min_rating),
        year_range=(int(year_min), int(year_max)),
        pages_range=(int(pages_min), int(pages_max)),
        sort_by=str(sort_by),
    )
    if df is None or df.empty:
        gallery = [("cover-not-found.png", "No results — try a broader description or different filters.")]
        return gallery, gr.update(visible=False, value=pd.DataFrame())

    gallery = []
    for _, row in df.iterrows():
        desc = str(row.get("description", "") or "")
        words = desc.split()
        truncated = " ".join(words[:28]) + ("..." if len(words) > 28 else "")
        authors = str(row.get("authors", "Unknown") or "Unknown")
        if ";" in authors:
            parts = [a.strip() for a in authors.split(";") if a.strip()]
            if len(parts) == 2:
                authors_str = f"{parts[0]} and {parts[1]}"
            elif len(parts) > 3:
                authors_str = f"{', '.join(parts[:-1])}, and {parts[-1]}"
            else:
                authors_str = ", ".join(parts)
        else:
            authors_str = authors

        title = str(row.get("title", "Untitled") or "Untitled")
        rating = row.get("average_rating", np.nan)
        rating_str = f" — ★ {rating:.2f}" if pd.notna(rating) else ""
        caption = f"{title}{rating_str} by {authors_str}: {truncated}"
        thumb = row.get("large_thumbnail", "cover-not-found.png") or "cover-not-found.png"
        gallery.append((thumb, caption))

    table = format_results_table(df)
    return gallery, gr.update(visible=bool(show_table), value=table if show_table else pd.DataFrame())

def clear_all():
    return (
        [], "", "All", "All",
        50, 16, 0.35,  # k0, k1, tone weight
        0.0, YEAR_MIN, YEAR_MAX, PAGES_MIN, PAGES_MAX,
        "Best match", False,
        gr.update(value=pd.DataFrame(), visible=False),
    )

# -------------------------
# UI (relaxed, lighter look)
# -------------------------
light_css = """
body { background: linear-gradient(180deg, #f9fbff 0%, #f3f7ff 100%); }
.gradio-container { font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji"; }
#title h1 { font-weight: 700; }
#subtitle { color: #50607a; }
footer { opacity: 0.7; }
"""

theme = gr.themes.Soft()

with gr.Blocks(theme=theme, css=light_css, fill_height=True) as dashboard:
    gr.Markdown("<div id='title'><h1>🌤️  Calm Reads — Semantic Book Recommender</h1></div>")
    gr.Markdown("<div id='subtitle'>Describe the vibe you want. Filter by category, rating, year, pages, and boost an emotion tone.</div>")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe a book (theme, vibe, or plot)",
            placeholder="e.g., A quiet story about forgiveness and second chances",
            scale=3,
        )

    with gr.Row():
        category_drop = gr.Dropdown(choices=categories, label="Category", value="All", allow_custom_value=False)
        tone_drop = gr.Dropdown(choices=tones, label="Tone boost", value="All")

    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            init_k = gr.Slider(10, 200, value=50, step=5, label="Initial retrieval (k)")
            final_k = gr.Slider(4, 32, value=16, step=1, label="Final results")
            tone_weight = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Tone weight (0 = ignore tone)")
        with gr.Row():
            min_rating = gr.Slider(0.0, 5.0, value=0.0, step=0.1, label="Minimum rating")
            year_min = gr.Number(value=YEAR_MIN, label="Year from")
            year_max = gr.Number(value=YEAR_MAX, label="Year to")
        with gr.Row():
            pages_min = gr.Number(value=PAGES_MIN, label="Min pages")
            pages_max = gr.Number(value=PAGES_MAX, label="Max pages")
            sort_by = gr.Radio(
                choices=["Best match", "Highest rating", "Most rated", "Newest", "Shortest", "Longest"],
                value="Best match",
                label="Sort by"
            )
        show_table = gr.Checkbox(False, label="Show results table")  # <-- create BEFORE wiring submit

    with gr.Row():
        submit_button = gr.Button("✨ Find books", variant="primary")
        clear_button = gr.Button("Reset")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2, height=520, preview=True)
    results_table = gr.Dataframe(value=pd.DataFrame(), interactive=False, visible=False)

    first_real_cat = next((c for c in categories if c != "All"), "All")
    gr.Examples(
        examples=[
            ["hopeful story of friendship after loss", "All", "Happy"],
            ["a twisty plot with shocking reveals", "All", "Surprising"],
            ["a tense cat-and-mouse thriller", "All", "Suspenseful"],
            ["a heartbreaking family drama", "All", "Sad"],
            ["epic political intrigue", first_real_cat, "All"],
        ],
        inputs=[user_query, category_drop, tone_drop],
        label="Try these",
    )

    def _submit(query, cat, tone, k0, k1, tw, minr, y0, y1, p0, p1, sort, table_flag):
        gallery, table_update = rec_books(query, cat, tone, k0, k1, tw, minr, y0, y1, p0, p1, sort, table_flag)
        return gallery, table_update

    submit_button.click(
        _submit,
        inputs=[
            user_query, category_drop, tone_drop,
            init_k, final_k, tone_weight,
            min_rating, year_min, year_max, pages_min, pages_max, sort_by,
            show_table  # pass the actual checkbox as an input
        ],
        outputs=[output, results_table],
        queue=True,
        api_name="recommend",
    )

    clear_button.click(
        clear_all,
        inputs=[],
        outputs=[
            output, user_query, category_drop, tone_drop,
            init_k, final_k, tone_weight,
            min_rating, year_min, year_max, pages_min, pages_max, sort_by, show_table, results_table
        ],
    )

    gr.Markdown("<br><small>Tip: increase <em>Tone weight</em> or switch <em>Sort by</em> to explore different facets.</small>")

if __name__ == "__main__":
    # Works on both older 3.x and newer 4.x Gradio
    try:
        app = dashboard.queue()  # no concurrency_count argument
    except TypeError:
        app = dashboard          # some older builds don’t have .queue()

    app.launch(
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True,
    )

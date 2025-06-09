import os
import json
import requests
import faiss
import numpy as np
import re
import streamlit as st
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

CACHE_DIR = "MTGCacheAllCards"
EMBED_MODEL = "text-embedding-ada-002"
SIMILARITY_THRESHOLD = 0.4

GITHUB_RELEASE = "https://github.com/cfle/mtg_oracle/releases/download/v1.0"
REQUIRED_FILES = {
    "cards.json": f"{GITHUB_RELEASE}/cards.json",
    "embeddings_trimmed.npy": f"{GITHUB_RELEASE}/embeddings_trimmed.npy",
    "faiss_trimmed.index": f"{GITHUB_RELEASE}/faiss_trimmed.index",
}

def download_file(filename, url):
    local_path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(local_path):
        st.info(f"📦 Downloading `{filename}`...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logging.info(f"Downloaded {filename}")
        except Exception as e:
            logging.error(f"Failed to download {filename}: {e}")
            raise

def validate_cache_files():
    missing = []
    for filename, url in REQUIRED_FILES.items():
        path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(path):
            try:
                download_file(filename, url)
            except Exception as e:
                missing.append((filename, str(e)))
    if missing:
        st.error("🛑 Failed to download required files:")
        for fname, err in missing:
            st.code(f"{fname}: {err}")
        logging.error(f"Missing files: {missing}")
        st.stop()

def fetch_cards():
    try:
        with open(os.path.join(CACHE_DIR, "cards.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.exception("Failed to load cards.json")
        st.error("❌ Failed to load card data.")
        st.stop()

def get_card_text(card):
    name = card.get("name", "")
    text = card.get("oracle_text", "")
    if name:
        text = re.sub(rf'\b{re.escape(name)}\b', "this card", text, flags=re.IGNORECASE)
    keywords = " ".join(card.get("keywords", []))
    return f"{text} {keywords}".strip()

@st.cache_data
def load_data():
    try:
        cards = fetch_cards()
        embeddings = np.load(os.path.join(CACHE_DIR, "embeddings_trimmed.npy"))
        index = faiss.read_index(os.path.join(CACHE_DIR, "faiss_trimmed.index"))
        return cards, embeddings, index
    except Exception as e:
        logging.exception("Failed during data loading")
        st.error("❌ Failed to load cache files or embeddings.")
        st.stop()

def try_get_card_text_from_name(name):
    try:
        url = f"https://api.scryfall.com/cards/named?fuzzy={name}"
        res = requests.get(url)
        if res.status_code == 200:
            card = res.json()
            return get_card_text(card), card
    except Exception as e:
        logging.warning(f"Scryfall lookup failed for name '{name}': {e}")
    return None, None

def main():
    resolved_card = None
    st.set_page_config(layout="wide")
    st.title("🧙‍♂️ MTG Semantic Search (Trimmed Embeddings Only)")
    st.markdown("Search for similar cards using a clean MTG dataset.")

    query = st.text_input("🔎 Enter a card name or description:")
    search_button = st.button("🔍 Search")

    validate_cache_files()

    with st.spinner("🔁 Loading prebuilt data..."):
        cards, embeddings, index = load_data()

    color_options = ["W", "U", "B", "R", "G", "C"]
    color_labels = {
        "W": "⚪️ White", "U": "🔵 Blue", "B": "⚫️ Black",
        "R": "🔴 Red", "G": "🟢 Green", "C": "💠 Colorless"
    }

    with st.expander("🎨 Filter by Color Identity", expanded=True):
        selected_colors = []
        cols = st.columns(len(color_options))
        for i, color in enumerate(color_options):
            if cols[i].checkbox(color_labels[color], value=True):
                selected_colors.append(color)

    if query and search_button:
        logging.info(f"User submitted query: '{query}'")
        with st.spinner("🔍 Searching..."):
            try:
                resolved_text, resolved_card = try_get_card_text_from_name(query)
                if not resolved_card:
                    st.error("❌ Could not resolve that card name via Scryfall.")
                    logging.warning(f"Could not resolve card: {query}")
                    return

                query_text = get_card_text(resolved_card)

                try:
                    ref_index = next(i for i, c in enumerate(cards) if c["id"] == resolved_card["id"])
                    query_vec = embeddings[ref_index].reshape(1, -1)
                    query_vec /= np.linalg.norm(query_vec)
                except StopIteration:
                    st.error("❌ Couldn't find prebuilt embedding for this card.")
                    logging.warning(f"No embedding for card: {resolved_card.get('name')}")
                    return

                scores, I = index.search(query_vec, 200)
                results = [
                    (score, cards[idx])
                    for score, idx in zip(scores[0], I[0])
                    if score >= SIMILARITY_THRESHOLD and cards[idx].get("id") != resolved_card.get("id")
                ]

                def matches_color(card):
                    identity = card.get("color_identity", [])
                    if "C" in selected_colors:
                        return (not identity and "C" in selected_colors) or any(c in identity for c in selected_colors if c != "C")
                    return any(c in identity for c in selected_colors)

                if selected_colors:
                    results = [(score, card) for score, card in results if matches_color(card)]

                if not results:
                    st.warning("😕 No sufficiently similar cards found.")
                    logging.info("No similar cards found.")
                else:
                    cols = st.columns(5)
                    if resolved_card:
                        with cols[0]:
                            image_url = resolved_card.get("image_uris", {}).get("normal")
                            if not image_url and "card_faces" in resolved_card:
                                image_url = resolved_card["card_faces"][0].get("image_uris", {}).get("normal")
                            if image_url:
                                st.image(image_url, use_container_width=True)
                            st.markdown(f"[**{resolved_card.get('name', 'Unknown Card')}**]({resolved_card.get('scryfall_uri', '#')})")
                            st.markdown("**Similarity:** `1.000`")

                    for i, (score, card) in enumerate(results):
                        with cols[(i + 1) % 5]:
                            image_url = card.get("image_uris", {}).get("normal")
                            if not image_url and "card_faces" in card:
                                image_url = card["card_faces"][0].get("image_uris", {}).get("normal")
                            if image_url:
                                st.image(image_url, use_container_width=True)
                            st.markdown(f"[**{card.get('name', 'Unknown Card')}**]({card.get('scryfall_uri', '#')})")
                            st.markdown(f"**Similarity:** `{score:.3f}`")

            except Exception as e:
                logging.exception("Unexpected error during search")
                st.error("🚨 An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main()

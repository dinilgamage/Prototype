# app.py  –  Movie-plot similarity + scene-progression metric
# -----------------------------------------------------------
from flask import Flask, request, render_template
import os, re, json, pickle, requests
import numpy as np
import markdown2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from fastdtw import fastdtw
from scipy.spatial.distance import cosine as cosine_distance

# -------------------------
# Configuration & startup
# -------------------------
app = Flask(__name__)

EMB_PATH   = "EmbeddingsBGE/embeddings.npy"
DF_PATH    = "EmbeddingsBGE/df_cleaned.pkl"
ALPHA      = 0.7                      # weight for whole-plot vs. scene metric
MAX_SCENES = 20                       # LLM scene cap

BGE = "BAAI/bge-large-en-v1.5"
MINILM = "all-MiniLM-L6-v2"
MPNET = "all-mpnet-base-v2"

sbert_model = SentenceTransformer(BGE)
sbert_model_sequnce = SentenceTransformer(MINILM)
nlp         = spacy.load("en_core_web_sm")

API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-exp:free"

# -------------------------
# Utility helpers
# -------------------------
def normalize_names(text: str) -> str:
    """Replace PERSON/GPE/ORG… entities with <NAME> so names don’t dominate."""
    doc = nlp(text)
    tokens = [tok.text for tok in doc]
    for ent in reversed(doc.ents):
        if ent.label_ in {"PERSON", "GPE", "LOC", "FAC", "ORG", "NORP"}:
            tokens[ent.start : ent.end] = ["<NAME>"]
    return " ".join(tokens)

def preprocess_text(text: str) -> str:
    text = normalize_names(text)
    text = text.lower().strip()
    text = re.sub(r"\W+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ----------  Load catalog ----------
if os.path.exists(EMB_PATH) and os.path.exists(DF_PATH):
    embeddings = np.load(EMB_PATH)
    with open(DF_PATH, "rb") as f:
        df_cleaned = pickle.load(f)
    print("✅ Loaded embeddings and DataFrame")
else:
    raise RuntimeError("❌ Embedding files not found – run preprocessing first.")

# ----------  Whole-plot similarity ----------
def find_best_match(plot: str):
    emb_in  = sbert_model.encode(preprocess_text(plot), convert_to_numpy=True)
    sims    = cosine_similarity([emb_in], embeddings)[0]
    idx     = int(np.argmax(sims))
    return df_cleaned.iloc[idx]["Title"], sims[idx], idx

# ----------  LLM scene extractor ----------
def extract_scenes(plot: str, cap: int = MAX_SCENES):
    prompt = (
        "DO NOT include MARKDOWN FORMATTING if this is done the reponse will be invalid, provide the response as a plain JSON array. This is vital and mardown will break the code."
        "You are an expert screenplay analyst with 20 years of experience in scene breakdown. "
        "I need you to split the movie plot into distinct scenes based on these EXACT criteria:\n\n"
+       "1. Create between 5 and 15 scenes, depending on plot length (never more than 20)\n"
        "2. Each scene should represent a distinct narrative unit with a clear purpose\n"
        "3. Break scenes when there is:\n"
        "   - A change in location\n"
        "   - A significant time jump\n"
        "   - A shift in character focus\n"
        "   - A transition between major plot events\n"
        "4. Each scene should be 3-5 sentences long when possible\n"
        "5. Do NOT paraphrase or alter any wording from the original text\n"
        "6. Each scene must be extracted VERBATIM from the input text\n\n"
        "Example format: [\"Scene 1 text exactly as written...\", \"Scene 2 text exactly as written...\"]\n\n"
        f"PLOT TO ANALYZE:\n{plot}\n"
    )
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(API_URL, json=data, headers=headers, timeout=60)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    print("RAW LLM OUTPUT:\n", raw)
    try:
        scenes = json.loads(raw)
        scenes = [s.strip() for s in scenes if s.strip()]
        return scenes[:cap]
    except Exception:
        print("⚠️  Scene JSON parse failed – falling back to paragraph split.")
        return [p.strip() for p in plot.split("\n") if p.strip()][:cap]

# ----------  Scene-progression similarity ----------
def scene_progression_similarity(plot_a: str, plot_b: str) -> float:
    sc_a = extract_scenes(plot_a)
    sc_b = extract_scenes(plot_b)
    emb_a = sbert_model_sequnce.encode([preprocess_text(s) for s in sc_a], convert_to_numpy=True)
    emb_b = sbert_model_sequnce.encode([preprocess_text(s) for s in sc_b], convert_to_numpy=True)
    dist, _ = fastdtw(emb_a, emb_b, dist=cosine_distance)
    norm_dist = dist / max(len(emb_a), len(emb_b))  # ~[0,2]
    return 1 / (1 + norm_dist)                      # ∈ (0,1]

def scene_progression_similarity_llm(plot_a: str, plot_b: str) -> float:
    """Use LLM to directly analyze scene progression similarity between plots"""
    
    # Truncate long plots if needed
    max_length = 6000
    if len(plot_a) > max_length:
        plot_a = plot_a[:max_length] + "..."
    if len(plot_b) > max_length:
        plot_b = plot_b[:max_length] + "..."
    
    prompt = (
        "You are an expert screenplay analyst specializing in narrative structure comparison.\n\n"
        "Analyze these two movie plots for scene progression similarity. Follow these steps exactly:\n"
        "1. Break each plot into 5-7 key scenes\n"
        "2. Compare how the scenes progress in both plots (setup, conflict, resolution, etc.)\n"
        "3. Identify structural similarities in how the stories unfold\n"
        "4. Recognize similar narrative beats even when described with completely different vocabulary\n\n"
        "5. Calculate a scene progression similarity score from 0.0 to 1.0 where:\n"
        "   - 1.0: Nearly identical scene progression\n"
        "   - 0.7-0.9: Very similar narrative beats in similar order\n"
        "   - 0.4-0.6: Moderate similarities in structure\n"
        "   - 0.1-0.3: Few structural similarities\n"
        "   - 0.0: Completely different narrative structures\n\n"
        "IMPORTANT: Your score must be a precise value with 2 decimal places, NOT a whole number or range.\n\n"
        "RESPOND WITH ONLY THE NUMERICAL SCORE (WITH 2 DECIMAL PLACES), NOTHING ELSE.\n\n"
        f"PLOT A:\n{plot_a}\n\nPLOT B:\n{plot_b}\n\n"
        "Scene progression similarity score (0.00-1.00):"
    )
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": MODEL,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        r = requests.post(API_URL, json=data, headers=headers, timeout=90)
        r.raise_for_status()
        response = r.json()["choices"][0]["message"]["content"].strip()
        
        # Extract just the number from the response
        import re
        score_match = re.search(r'(\d+\.\d+)', response)
        if score_match:
            score = float(score_match.group(1))
            return min(max(score, 0.0), 1.0)  # Ensure value is between 0 and 1
        else:
            # Fallback to old method if no number found
            print("⚠️ No score found in LLM response, falling back to DTW method")
            return scene_progression_similarity(plot_a, plot_b)
            
    except Exception as e:
        print(f"❌ LLM scene progression analysis failed: {e}")
        # Fallback to original DTW method
        return scene_progression_similarity(plot_a, plot_b)

def combined_score(base: float, scene_sim: float) -> float:
    return ALPHA * base + (1 - ALPHA) * scene_sim

# ----------  LLM explanation ----------
def generate_explanation(plot1: str, plot2: str, score: float) -> str:
    prompt = (
        f"Two movie plots have a combined similarity score of {score:.2f}. "
        "Explain in detail why they are similar, covering story structure, themes, "
        "character arcs, and other key elements. Provide a clear conclusion. "
        "Use well-labeled sections.\n\n"
        f"Plot 1:\n{plot1}\n\nPlot 2:\n{plot2}\n\nExplanation:"
    )
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(API_URL, json=data, headers=headers, timeout=90)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"].strip()
    print("❌ Explanation LLM call failed:", r.text)
    return "Error: Unable to generate explanation."

# -------------------------
# Flask routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/similarity", methods=["POST"])
def similarity():
    input_plot = request.form.get("plot") or (request.json or {}).get("plot")
    if not input_plot:
        return render_template("index.html", error="No plot provided.")

    # Whole-plot match
    title, base_sim, idx = find_best_match(input_plot)
    matched_plot = df_cleaned.iloc[idx]["Plot"]

    # New scene-progression metric & blend
    scene_sim = scene_progression_similarity_llm(input_plot, matched_plot)
    combo_sim = combined_score(base_sim, scene_sim)

    # LLM explanation
    explanation_md = markdown2.markdown(
        # generate_explanation(input_plot, matched_plot, combo_sim)
        "Explanation placeholder: This is where the LLM explanation will go."
    )

    # Package for template
    result = {
        "input_plot": input_plot,
        "matched_plot": matched_plot,
        "matched_title": title,
        "plot_similarity": round(float(base_sim), 3),
        "scene_progression_similarity": round(float(scene_sim), 3),
        "combined_similarity": round(float(combo_sim), 3),
        "explanation_html": explanation_md,
    }
    return render_template("result.html", result=result)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)   # disable debug in prod!

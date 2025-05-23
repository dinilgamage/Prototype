from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pickle
import re
import requests
import markdown2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# -------------------------
# Configuration and Setup
# -------------------------
app = Flask(__name__)

# Define file paths for saved embeddings and DataFrame
embeddings_path = "Embeddings5/embeddings.npy"
df_cleaned_path = "Embeddings5/df_cleaned.pkl"

# Load the SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load spaCy's English model for named entity recognition
nlp = spacy.load('en_core_web_sm')

# Updated Named Entity Normalization
def normalize_names(text):
    doc = nlp(text)
    # Create a copy of the text with all tokens
    tokens = [token.text for token in doc]
    
    # Process entities in reverse order to avoid index issues
    for ent in reversed(doc.ents):
        start = ent.start
        end = ent.end
        
        # Replace any named entity with <NAME>
        if ent.label_ in ["PERSON", "GPE", "LOC", "FAC", "ORG", "NORP"]:
            tokens[start:end] = ["<NAME>"]
    
    return " ".join(tokens)


# Preprocessing function remains the same, now using improved normalization
def preprocess_text(text):
    text = normalize_names(text)
    text = text.lower().strip()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------
# Load cleaned DataFrame and embeddings
# -------------------------
if os.path.exists(embeddings_path) and os.path.exists(df_cleaned_path):
    embeddings = np.load(embeddings_path)
    with open(df_cleaned_path, "rb") as f:
        df_cleaned = pickle.load(f)
    print("Loaded saved embeddings and DataFrame.")
else:
    raise Exception("Embeddings not found. Please run the embedding computation pipeline first.")

# Similarity function
def find_most_similar_movie(input_plot, df, embeddings, model):
    processed_input = preprocess_text(input_plot)
    input_embedding = model.encode(processed_input, convert_to_numpy=True)
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    max_idx = np.argmax(similarities)
    best_match_movie = df.iloc[max_idx]['Title']
    best_match_score = similarities[max_idx]
    return best_match_movie, best_match_score, max_idx

# Explanation function via OpenRouter
API_KEY = os.environ.get('OPENROUTER_API_KEY')
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

def generate_explanation(plot1, plot2, similarity_score):
    prompt = (
        f"Two movie plots have been identified with a similarity score of {similarity_score:.2f}. "
        "Based on the following two plots, accurately explain in full detail why these movies are similar. "
        "Highlight the narrative structure, themes, character development, and any other key elements that contribute to this similarity and also a conclusion at the end. Ensure these sections are seperated into detailed, structured and focused sections\n\n"
        f"Plot 1:\n{plot1}\n\n"
        f"Plot 2:\n{plot2}\n\n"
        "Explanation:"
    )

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "meta-llama/llama-3-8b-instruct:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        response_json = response.json()
        explanation = response_json['choices'][0]['message']['content'].strip()
        return explanation
    else:
        print(f"Failed to fetch data from API. Status Code: {response.status_code}")
        print("Response:", response.text)
        return "Error: Unable to generate explanation."

# -------------------------
# Flask Routes
# -------------------------

@app.route("/")
def index():
    # Render a simple HTML form.
    return render_template("index.html")

@app.route("/similarity", methods=["POST"])
def similarity():
    # Retrieve the movie plot from the form or JSON payload
    input_plot = request.form.get("plot") or request.json.get("plot")
    if not input_plot:
        return render_template("index.html", error="No plot provided.")
    
    # Show the processed version of the input plot (with NER changes)
    processed_input = normalize_names(input_plot)
    print("Processed Input Plot:", processed_input)
    
    # Find the most similar movie
    movie, score, best_idx = find_most_similar_movie(input_plot, df_cleaned, embeddings, sbert_model)
    matched_movie_plot = df_cleaned.iloc[best_idx]['Plot']
    
    # Process the matched movie plot for NER changes as well and print
    processed_matched = normalize_names(matched_movie_plot)
    print("Processed Matched Movie Plot:", processed_matched)
    
    # Generate explanation using DeepSeek
    explanation = generate_explanation(input_plot, matched_movie_plot, score)
    explanation_html = markdown2.markdown(explanation)
    
    # Build result dictionary
    result = {
        "input_plot": input_plot,
        "matched": matched_movie_plot,
        "best_match_movie": movie,
        "similarity_score": float(score),
        "explanation": explanation_html
    }
    
    # Render result template with the results
    return render_template("result.html", result=result)

# -------------------------
# Run the Flask Application
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


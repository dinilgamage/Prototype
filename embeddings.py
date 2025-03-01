import pandas as pd
import numpy as np
import pickle
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

tqdm.pandas()

# Load spaCy's English model for named entity recognition
nlp = spacy.load('en_core_web_sm')

# Function to normalize person names by replacing them with "NAME"
def normalize_names(text):
    doc = nlp(text)
    normalized_tokens = []
    for token in doc:
        if token.ent_type_ in ["PERSON", "GPE", "LOC"]:
            normalized_tokens.append("NAME")
        else:
            normalized_tokens.append(token.text)
    return " ".join(normalized_tokens)

# Updated preprocessing function that includes name normalization
def preprocess_text(text):
    text = normalize_names(text)
    text = text.lower().strip()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the dataset
file_path = r"C:\Users\Dinil\Desktop\Dinil\FYP\Datasets\wiki_movie_plots_deduped.csv"
df = pd.read_csv(file_path)
df_cleaned = df.dropna(subset=['Plot'])

# Apply updated preprocessing
df_cleaned['Processed Plot'] = df_cleaned['Plot'].progress_apply(preprocess_text)

# Load the SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
embeddings_list = []
for text in tqdm(df_cleaned['Processed Plot'], desc="Computing embeddings"):
    embedding = sbert_model.encode(text, convert_to_numpy=True)
    embeddings_list.append(embedding)
embeddings = np.vstack(embeddings_list)

# Define file paths to save the embeddings and DataFrame locally
embeddings_path = "Embeddings3/embeddings.npy" 
df_cleaned_path = "Embeddings3/df_cleaned.pkl"

# Save the computed embeddings and DataFrame
np.save(embeddings_path, embeddings)
with open(df_cleaned_path, "wb") as f:
    pickle.dump(df_cleaned, f)
    
print("Saved new embeddings and DataFrame with updated preprocessing.")

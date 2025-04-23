import pandas as pd
import numpy as np
import pickle
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger

tqdm.pandas()

# Load Flair's NER model (this loads a more accurate model than spaCy's default)
ner_tagger = SequenceTagger.load('ner')

# Enhanced named entity normalization using Flair
def normalize_names(text):
    # Create a Flair sentence
    sentence = Sentence(text)
    
    # Run NER on the sentence
    ner_tagger.predict(sentence)
    
    # Extract entities and their positions
    entities = []
    for entity in sentence.get_spans('ner'):
        start_pos = entity.start_position
        end_pos = entity.end_position
        entity_type = entity.tag
        entities.append((start_pos, end_pos, entity_type))
    
    # Sort entities in reverse order by start position to safely replace text
    entities.sort(reverse=True)
    
    # Replace entities with appropriate tokens
    for start_pos, end_pos, entity_type in entities:
        if entity_type == 'PER':
            text = text[:start_pos] + "<PERSON>" + text[end_pos:]
        elif entity_type in ['LOC', 'GPE']:
            text = text[:start_pos] + "<LOCATION>" + text[end_pos:]
        elif entity_type == 'ORG':
            text = text[:start_pos] + "<ORGANIZATION>" + text[end_pos:]
        elif entity_type == 'NORP':
            text = text[:start_pos] + "<NATIONALITY>" + text[end_pos:]
        elif entity_type == 'MISC':
            text = text[:start_pos] + "<MISCELLANEOUS>" + text[end_pos:]
    
    return text

# The rest of your preprocessing function remains the same
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

# REDUCED TO 5 ENTRIES FOR TESTING
df_cleaned = df_cleaned.head(5)
print(f"Testing with reduced dataset of {len(df_cleaned)} entries")

# Apply updated preprocessing
print("Preprocessing movie plots...")
df_cleaned['Processed Plot'] = df_cleaned['Plot'].progress_apply(preprocess_text)

# Load the SBERT model
print("Loading Sentence-BERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
print("Computing embeddings...")
embeddings_list = []
for text in tqdm(df_cleaned['Processed Plot'], desc="Computing embeddings"):
    embedding = sbert_model.encode(text, convert_to_numpy=True)
    embeddings_list.append(embedding)
embeddings = np.vstack(embeddings_list)

# Define file paths to save the embeddings and DataFrame locally
os.makedirs("Embeddings_test", exist_ok=True)
embeddings_path = "Embeddings_test/embeddings.npy"
df_cleaned_path = "Embeddings_test/df_cleaned.pkl"

# Save the embeddings and DataFrame
print("Saving embeddings and processed data...")
np.save(embeddings_path, embeddings)
with open(df_cleaned_path, "wb") as f:
    pickle.dump(df_cleaned, f)

print(f"Processing complete. Saved embeddings to {embeddings_path} and DataFrame to {df_cleaned_path}")
print(f"Processed {len(df_cleaned)} movie plots with embedding dimension {embeddings.shape[1]}")

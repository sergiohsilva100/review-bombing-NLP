import pandas as pd
import re
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import os

# --- Global Configurations ---
# List of all products analyzed in the dissertation. This list makes the script modular.
PRODUCTS = [
    'captain_marvel', 'days_gone', 'inception', 'last_of_us_part_2', 
    'logan', 'red_dead_redemption_2', 'resident_evil_7', 'tlj'
]
BASE_INPUT_DIR = os.path.join("data", "raw")
# Dedicated folder for BERTopic output (models, info, etc.)
BASE_OUTPUT_DIR = os.path.join("data", "bertopic_models") 

# Check for GPU availability
DEVICE = 0 if torch.cuda.is_available() else -1
if DEVICE == 0:
    print("A GPU está disponível e será usada.")
else:
    print("A GPU não está disponível. O código usará o CPU.")

def analyze_topics_from_csv(
    input_file_path: str,
    output_dir: str,
    model_name: str
):
    """
    Loads reviews from a CSV file, performs topic modeling using BERTopic,
    and saves the model and the results.

    Args:
        input_file_path (str): Path to the CSV file containing the reviews.
        output_dir (str): Directory where results will be saved.
        model_name (str): Base name for the model and results files (usually the product slug).
    """
    
    # Define column names (based on the raw data structure)
    column_names = [
        'r_id',         # unique review identifier
        'of_title',     # title/abbreviation (same for all entries)
        'r_date',       # review date (YYYY-MM-DD)
        'score',        # 0-10 (games) or 1-5 (films)
        'review',       # review text
        'is_english'    # boolean (True if English)
    ]

    # Try to load the CSV file
    try:
        # Assuming no header is present in the raw data
        df = pd.read_csv(input_file_path, names=column_names, encoding='utf-8')
    except Exception as e:
        print(f"ERRO ao carregar {input_file_path}: {e}")
        return

    # Filter for English reviews and drop any NaN review text
    reviews_df = df[df['is_english'] == True].copy()
    reviews = reviews_df['review'].dropna().tolist()
    
    if not reviews:
        print("AVISO: Nenhuma review em inglês encontrada ou dados vazios. Pulando a análise.")
        return

    # Basic preprocessing: remove non-word characters and convert to lowercase
    reviews = [re.sub(r'[^\\w\\s]', '', str(review).lower()) for review in reviews]

    # --- BERTopic Configuration ---
    # Load embedding model and set device
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    # Dimensionality Reduction (UMAP) and Clustering (HDBSCAN) models
    # UMAP configurations
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    # HDBSCAN configurations
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', 
                            cluster_selection_method='eom', prediction_data=True)

    # Vectorizer model (for topic representation)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    # Create and fit the BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=True,
        verbose=True
    )
    
    print("Iniciando a análise de tópicos. Isso pode levar alguns minutos...")
    topics, probs = topic_model.fit_transform(reviews)
    
    # Get topic information and print the most frequent ones
    topic_info = topic_model.get_topic_info()
    print("\\n--- Tópicos Mais Frequentes ---")
    print(topic_info.head(10))

    # --- Save Results ---
    print("\\nSalvando resultados...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the full BERTopic model
    model_path = os.path.join(output_dir, f"{model_name}")
    topic_model.save(model_path, serialization="safetensors")

    # Save the topic information table
    info_path = os.path.join(output_dir, f"{model_name}_topics_info.csv")
    topic_info[["Topic", "Count", "Name", "Representation"]].to_csv(info_path, index=False)
    print(f"Modelo e resultados salvos com sucesso em: {output_dir}")

def main():
    """
    Main function to orchestrate the BERTopic analysis for all defined products.
    """
    print("="*70)
    print("Iniciando a análise de Modelagem de Tópicos (BERTopic) para todos os produtos.")
    print("="*70)
    
    # Ensure the output directory exists
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        print(f"Diretório de saída criado: {BASE_OUTPUT_DIR}")

    # Loop through each product defined in the global list
    for product_slug in PRODUCTS:
        # Construct the input file path: data/raw/{product}_reviews.csv
        input_file = os.path.join(BASE_INPUT_DIR, f"{product_slug}_reviews.csv")
        
        # Define the model name for saving (based on the product slug)
        model_name = f"{product_slug}_bertopic_model"

        print("\n" + "#"*50)
        print(f"PRODUTO ATUAL: {product_slug.upper()}")
        print("#"*50)

        if not os.path.exists(input_file):
            print(f"AVISO: Arquivo de entrada não encontrado: {input_file}. Pulando este produto.")
            continue

        analyze_topics_from_csv(
            input_file_path=input_file, 
            output_dir=BASE_OUTPUT_DIR, 
            model_name=model_name
        )

if __name__ == '__main__':
    main()
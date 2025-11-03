import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import os
import sys
import io

# --- Configuration for Brazilian Portuguese output ---
# Setting stdout to UTF-8 to correctly display accented characters in prints
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# --- Global Configurations ---
# List of all products analyzed in the dissertation.
PRODUCTS = [
    'captain_marvel', 'days_gone', 'inception', 'last_of_us_part_2', 
    'logan', 'red_dead_redemption_2', 'resident_evil_7', 'tlj'
]
BASE_INPUT_DIR = os.path.join("data", "raw")

# Model-specific slug for naming files and output folders
MODEL_SLUG = "covid-twitter-bert-v2" 
BASE_OUTPUT_DIR = os.path.join("data", MODEL_SLUG) 
MODEL_NAME = "digitalepidemiologylab/covid-twitter-bert-v2-mnli"

# Set device configuration (GPU/CPU)
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Usando {'GPU' if DEVICE == 0 else 'CPU'} - Modelo: {MODEL_NAME}")

# Define column names based on the raw data structure
COLUNAS = [
    'r_id',           # Unique review identifier
    'of_title',       # Product title
    'r_date',         # Review date (YYYY-MM-DD)
    'score',          # Score (1-5 or 0-10)
    'review',         # Review text
    'is_english'      # English language indicator (boolean)
]

# Labels for Zero-Shot classification
LABELS = [
    "Ideological Bias",
    "LGBTQ+ Criticism",
    "Conspiracy Theory",
    "Review Ecosystem Critique",
    "Technical Criticism",
    "Tactical Praise",
    "Coordinated Hate",
    "Studio Criticism",
    "Frustration/Expectation"
]

# Initialize the ZERO-SHOT classifier pipeline
try:
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=DEVICE
    )
except Exception as e:
    print(f"ERRO ao carregar o modelo {MODEL_NAME}: {e}")
    classifier = None

def load_and_filter_data(input_file_path: str) -> pd.DataFrame:
    """Loads the raw CSV file and filters for English reviews."""
    if not os.path.exists(input_file_path):
        print(f"AVISO: Arquivo de entrada não encontrado: {input_file_path}. Pulando.")
        return pd.DataFrame()

    try:
        # Load data assuming no header is present
        df = pd.read_csv(input_file_path, header=None, names=COLUNAS, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to Latin-1 encoding if UTF-8 fails
        df = pd.read_csv(input_file_path, header=None, names=COLUNAS, encoding='latin-1')
        
    # Convert 'is_english' column to boolean
    df['is_english'] = df['is_english'].astype(bool)

    # Filter only English reviews
    english_reviews = df[df['is_english']].copy()
    print(f"Total de reviews no arquivo: {len(df)} | Reviews em inglês a classificar: {len(english_reviews)}")
    
    return english_reviews

def classify_text(text: str) -> dict:
    """
    Performs Zero-Shot classification on a single text with robust error handling.
    """
    if not classifier:
        return {label: None for label in LABELS}
        
    if not text or str(text).strip() == "":
        return {label: None for label in LABELS}
    
    try:
        result = classifier(
            str(text),
            candidate_labels=LABELS,
            truncation=True,
            max_length=512,
            multi_label=True  # Enables multi-label mode
        )
        # Map labels and scores to a dictionary
        return dict(zip(result['labels'], result['scores']))
    except Exception as e:
        # Log error but return None for scores
        print(f"Erro na classificação: {str(e)} para o texto '{text[:50]}...'")
        return {label: None for label in LABELS}

def process_product_reviews(product_slug: str):
    """
    Main logic to classify reviews of a single product and save the result.
    """
    input_file = os.path.join(BASE_INPUT_DIR, f"{product_slug}_reviews.csv")
    output_file = os.path.join(BASE_OUTPUT_DIR, f"{product_slug}_classified.csv")

    print("\n" + "#"*50)
    print(f"PRODUTO ATUAL: {product_slug.upper()}")
    print("#"*50)
    
    english_reviews = load_and_filter_data(input_file)
    
    if english_reviews.empty:
        print("AVISO: Pulando análise. DataFrame de reviews em inglês está vazio.")
        return
        
    if not classifier:
        print("AVISO: O classificador não pôde ser inicializado. Pulando classificação.")
        return

    # Add new columns for scores, ensuring column names are safe
    for label in LABELS:
        # Replace problematic characters for column names
        col_name = label.replace(" ", "_").replace("+", "plus").replace("/", "_")
        english_reviews[col_name] = None

    print(f"Iniciando classificação Zero-Shot para {len(english_reviews)} reviews...")

    # Iterate over reviews and classify
    for idx, row in tqdm(english_reviews.iterrows(), total=len(english_reviews), desc="Classificando"):
        scores = classify_text(row['review'])
        
        # Update the DataFrame with the scores
        for label, score in scores.items():
            col_name = label.replace(" ", "_").replace("+", "plus").replace("/", "_")
            english_reviews.at[idx, col_name] = score

    # Save results
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        print(f"Diretório de saída criado: {BASE_OUTPUT_DIR}")
        
    english_reviews.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nProcesso concluído! Resultados salvos em: {output_file}")


def main():
    """
    Orchestrates the Zero-Shot classification for all products defined in the PRODUCTS list.
    """
    print("="*70)
    print(f"Iniciando a Classificação Zero-Shot com o modelo: {MODEL_NAME}")
    print(f"Total de Produtos a serem processados: {len(PRODUCTS)}")
    print("="*70)
    
    # Loop through each product and process its reviews
    for product_slug in PRODUCTS:
        process_product_reviews(product_slug)

if __name__ == '__main__':
    main()
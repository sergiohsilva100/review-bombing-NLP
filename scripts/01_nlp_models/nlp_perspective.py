import pandas as pd
import time
import random
import requests
from tqdm import tqdm
import urllib3
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import os
import sys
import io

# --- Configuration for Brazilian Portuguese output ---
# Setting stdout to UTF-8 to correctly display accented characters in prints
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# --- Global Configurations ---
# List of all products analyzed in the dissertation. This list makes the script modular.
PRODUCTS = [
    'captain_marvel', 'days_gone', 'inception', 'last_of_us_part_2', 
    'logan', 'red_dead_redemption_2', 'resident_evil_7', 'tlj'
]
BASE_INPUT_DIR = os.path.join("data", "raw")
# Output will be saved in a specific folder for toxicity data
BASE_OUTPUT_DIR = os.path.join("data", "toxicity_analysis") 

# --- API Configuration (CRITICAL FOR PUBLIC REPOSITORY) ---
# NOTE: The user must set the PERSPECTIVE_API_KEY environment variable.
# It is replaced here with os.getenv() for security and public release.
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY") 
if not PERSPECTIVE_API_KEY:
    print("\n[ERRO] A variável de ambiente 'PERSPECTIVE_API_KEY' não foi definida.")
    print("Por favor, defina sua chave da API Perspective antes de executar o script.")
    # Exit gracefully if the key is not set
    # sys.exit(1)
    
PERSPECTIVE_API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

# Attributes requested from the Perspective API
ATTRIBUTES = {
    "TOXICITY": {},
    "SEVERE_TOXICITY": {},
    "IDENTITY_ATTACK": {},
    "INSULT": {},
    "PROFANITY": {},
    "THREAT": {}
}

# Column names based on the raw data structure
COLUNAS = [
    'r_id', 'of_title', 'r_date', 'score', 'review', 'is_english'
]

# Disable SSL warnings (Use with caution, only if necessary)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Robust Connection Strategy ---
# Configure retry strategy for handling temporary API errors (429, 5xx)
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

def load_data(input_file_path: str) -> pd.DataFrame:
    """Loads the raw CSV file, filters for English reviews."""
    try:
        # Assuming no header is present in the raw data
        df = pd.read_csv(input_file_path, header=None, names=COLUNAS, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to Latin-1 encoding
        df = pd.read_csv(input_file_path, header=None, names=COLUNAS, encoding='latin-1')
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {input_file_path}")
        return pd.DataFrame()
    
    # Filter only English reviews and make a copy to avoid SettingWithCopyWarning
    english_reviews = df[df['is_english'] == True].copy()
    
    return english_reviews

def analyze_toxicity(text: str) -> dict:
    """
    Calls the Perspective API for a given text, with robust error handling.
    Returns a dictionary of toxicity scores.
    """
    if not PERSPECTIVE_API_KEY:
        return {key: None for key in ATTRIBUTES}

    # API Payload structure
    payload = {
        'comment': {'text': str(text)},
        'languages': ['en'],
        'requestedAttributes': ATTRIBUTES,
        'doNotStore': True  # Recommended for privacy
    }
    params = {'key': PERSPECTIVE_API_KEY}
    
    try:
        response = session.post(
            PERSPECTIVE_API_URL, 
            json=payload, 
            params=params,
            timeout=30,
            verify=False  # Ignore SSL verification - USE WITH CAUTION
        )
        response.raise_for_status() # Raise exception for 4xx or 5xx responses (except those handled by Retry)
        data = response.json()

        scores = {}
        for attr in ATTRIBUTES:
            # Safely extract the 'summaryScore' value
            scores[attr] = data['attributeScores'].get(attr, {}).get('summaryScore', {}).get('value', None)
        return scores
    
    except Exception as e:
        print(f"Erro persistente na API para o texto '{text[:50]}...': {str(e)}")
        # Return None for all scores on persistent failure
        return {key: None for key in ATTRIBUTES}

def process_product_reviews(product_slug: str):
    """
    Main logic to load data, process reviews with Perspective API, and save results for a single product.
    """
    input_file = os.path.join(BASE_INPUT_DIR, f"{product_slug}_reviews.csv")
    output_file = os.path.join(BASE_OUTPUT_DIR, f"{product_slug}_toxicity_analysis.csv")

    print("\n" + "#"*50)
    print(f"PRODUTO ATUAL: {product_slug.upper()} | Arquivo: {input_file}")
    print("#"*50)
    
    english_reviews = load_data(input_file)
    
    if english_reviews.empty:
        print("AVISO: Pulando análise. DataFrame de reviews em inglês está vazio.")
        return

    # Add new columns for toxicity scores
    for attr in ATTRIBUTES:
        english_reviews[attr.lower()] = None

    print(f"Iniciando análise de toxicidade para {len(english_reviews)} reviews...")
    
    # Iterate over reviews and call the API
    for idx, row in tqdm(english_reviews.iterrows(), total=len(english_reviews), desc="Analisando Toxicidade"):
        # The 'review' column must contain the text to be analyzed
        scores = analyze_toxicity(row['review'])
        
        # Update the DataFrame with the scores
        for attr, score in scores.items():
            english_reviews.at[idx, attr.lower()] = score
        
        # Introduce a random delay to prevent hitting rate limits
        time.sleep(random.uniform(0.3, 1.2))

    # Save results
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        
    english_reviews.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nProcesso concluído! Resultados salvos em: {output_file}")


def main():
    """
    Orchestrates the toxicity analysis for all products defined in the PRODUCTS list.
    """
    print("="*70)
    print("Iniciando a Análise de Toxicidade com Google Perspective API.")
    print(f"Total de Produtos a serem processados: {len(PRODUCTS)}")
    print("="*70)
    
    # Check for API key again before starting the loop
    if not PERSPECTIVE_API_KEY:
        print("\nO script foi interrompido. A chave da API é necessária para continuar.")
        return

    # Loop through each product and process its reviews
    for product_slug in PRODUCTS:
        process_product_reviews(product_slug)

if __name__ == '__main__':
    main()
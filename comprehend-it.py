import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import os

def classify_movie_reviews(input_file_path: str, output_file_path: str):
    """
    Carrega reviews de um arquivo CSV, realiza a classificação zero-shot
    e salva os resultados em um novo arquivo CSV.

    Args:
        input_file_path (str): O caminho para o arquivo CSV de entrada.
        output_file_path (str): O caminho para o arquivo CSV de saída.
    """
    # Configuração do dispositivo
    device = 0 if torch.cuda.is_available() else -1
    print(f"Usando {'GPU' if device == 0 else 'CPU'} - Modelo: knowledgator/comprehend_it-base")

    # Nomes das colunas esperadas no arquivo de entrada
    COLUNAS = ['r_id', 'of_title', 'r_date', 'score', 'review', 'is_english']

    # Tenta carregar dados com diferentes encodings
    try:
        df = pd.read_csv(input_file_path, header=None, names=COLUNAS, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file_path, header=None, names=COLUNAS, encoding='latin-1')
        except FileNotFoundError:
            print(f"Erro: O arquivo '{input_file_path}' não foi encontrado.")
            return

    # Converter coluna is_english para booleano
    df['is_english'] = df['is_english'].astype(bool)

    # Filtrar apenas reviews em inglês
    english_reviews = df[df['is_english']].copy()
    if english_reviews.empty:
        print("Nenhuma review em inglês encontrada. Processo cancelado.")
        return

    print(f"Total de reviews: {len(df)} | Reviews em inglês a serem classificadas: {len(english_reviews)}")

    # Labels para classificação
    LABELS = [
        "Ideological Bias", "LGBTQ+ Criticism", "Conspiracy Theory",
        "Review Ecosystem Critique", "Technical Criticism", "Tactical Praise",
        "Coordinated Hate", "Studio Criticism", "Frustration/Expectation"
    ]

    # Inicializar o classificador ZERO-SHOT
    classifier = pipeline(
        "zero-shot-classification",
        model="knowledgator/comprehend_it-base",
        device=device
    )

    # Função para classificação com tratamento robusto
    def classify_text(text):
        if not text or str(text).strip() == "":
            return {label: None for label in LABELS}
        
        try:
            result = classifier(
                str(text),
                candidate_labels=LABELS,
                truncation=True,
                max_length=512,
                multi_label=True
            )
            return dict(zip(result['labels'], result['scores']))
        except Exception as e:
            print(f"Erro na classificação de um texto: {str(e)}")
            return {label: None for label in LABELS}

    # Adicionar colunas para scores
    for label in LABELS:
        english_reviews[label] = None

    # Processar as reviews e adicionar os scores
    tqdm.pandas(desc="Classificando reviews")
    english_reviews[LABELS] = english_reviews['review'].progress_apply(
        lambda x: pd.Series(classify_text(x))
    )

    # Criar diretório de saída se não existir
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salvar resultados
    english_reviews.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"Processo concluído! Arquivo salvo em: '{output_file_path}'")


if __name__ == "__main__":
    # Define o nome do arquivo de entrada e saída
    INPUT_FILE = "captain_marvel_reviews.csv"
    OUTPUT_FILE = "output/captain_marvel_comprehend_it_classified.csv"
    
    classify_movie_reviews(INPUT_FILE, OUTPUT_FILE)
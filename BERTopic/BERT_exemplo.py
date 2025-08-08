import pandas as pd
import re
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import os

def analyze_topics_from_csv(
    input_file_path: str,
    output_dir: str = "output",
    model_name: str = "captain_marvel_reviews_bertopic"
):
    """
    Carrega reviews de um arquivo CSV, realiza a análise de tópicos usando BERTopic
    e salva o modelo e os resultados.

    Args:
        input_file_path (str): O caminho para o arquivo CSV contendo as revisões.
        output_dir (str): O diretório onde os resultados serão salvos.
        model_name (str): O nome base para os arquivos de modelo e resultados.
    """
    # Verifica e configura o uso de GPU
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("A GPU está disponível e será usada.")
    else:
        print("A GPU não está disponível. O código usará o CPU.")

    # Tenta carregar o arquivo
    try:
        df = pd.read_csv(input_file_path, names=[
            'r_id', 'of_title', 'r_date', 'score', 'review', 'is_english'
        ])
    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_file_path}' não foi encontrado.")
        return

    # Extrai e pré-processa as reviews em inglês
    print("Processando dados...")
    reviews = df[df['is_english']]['review'].dropna().tolist()
    reviews = [re.sub(r'[^\w\s]', '', str(review).lower()) for review in reviews]

    if not reviews:
        print("Nenhuma review em inglês foi encontrada no arquivo. Análise cancelada.")
        return

    # Configura os modelos para BERTopic
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    # Cria e ajusta o modelo BERTopic
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
    
    # Exibe os tópicos mais relevantes
    topic_info = topic_model.get_topic_info()
    print("\n--- Tópicos Mais Frequentes ---")
    print(topic_info.head(10))

    # Salva os resultados
    print("\nSalvando resultados...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, f"{model_name}")
    topic_model.save(model_path, serialization="safetensors")

    info_path = os.path.join(output_dir, f"{model_name}_topics_info.csv")
    topic_info[["Topic", "Count", "Name", "Representation"]].to_csv(info_path, index=False)
    print(f"Modelo e resultados salvos com sucesso em '{output_dir}'.")


if __name__ == "__main__":
    # --- Configuração da Análise ---
    # Coloque o nome do seu arquivo aqui, ou de qualquer outro que você queira analisar.
    file_to_analyze = "captain_marvel_reviews.csv"
    
    # Chama a função principal com o arquivo desejado
    analyze_topics_from_csv(file_to_analyze)
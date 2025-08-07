import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import torch

device = 0 if torch.cuda.is_available() else -1

# Imprimir qual dispositivo está sendo usado
if device == 0:
    print("A GPU está disponível e será usada.")
else:
    print("A GPU não está disponível. O código usará o CPU.")

column_names = [
    'r_id',         # unique review identifier
    'of_title',     # title/abbreviation (same for all entries)
    'r_date',       # review date (YYYY-MM-DD)
    'score',        # 0-10 (games) or 1-5 (films)
    'review',       # review text
    'is_english'    # boolean (True if English)
]

# Carregar o CSV com os nomes das colunas
df = pd.read_csv("last_of_us_part_2_reviews.csv", names=column_names)

df['is_english'] = df['is_english'].astype(bool)
df = df.dropna(subset=['is_english'])

reviews = df.loc[df['is_english'], 'review'].dropna().tolist()

# pré-processamento básico
import re
reviews = [re.sub(r'[^\w\s]', '', str(review).lower()) for review in reviews]

# Configurar BERTopic para usar GPU
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Redução de dimensionalidade e clustering
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Vetorização
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

# Criar e ajustar o modelo BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    language="english",
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(reviews)

# Visualizar tópicos
topic_info = topic_model.get_topic_info()
print(topic_info.head(10))

# Salvar resultados (apenas com as colunas selecionadas)
topic_model.save("last_of_us_part_2_reviews_bertopic")
topic_info[["Topic", "Count", "Name", "Representation"]].to_csv("last_of_us_part_2_topics_info.csv", index=False)
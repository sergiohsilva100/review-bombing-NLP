import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys
from matplotlib.colors import to_hex

# --- Configurações Iniciais ---
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
sns.set_palette("pastel")
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.titlepad': 20,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'
})

# Lista de produtos e categorização
products = [
    'captain_marvel', 'days_gone', 'inception', 'last_of_us_part_2',
    'logan', 'red_dead_redemption_2', 'resident_evil_7', 'tlj'
]

review_bombed = ['captain_marvel', 'last_of_us_part_2', 'tlj']
normal = [p for p in products if p not in review_bombed]
games = ['days_gone', 'last_of_us_part_2', 'red_dead_redemption_2', 'resident_evil_7']
movies = [p for p in products if p not in games]

# Mapeamento de labels para português
label_translation = {
    'Ideological Bias': 'Viés Ideológico',
    'LGBTQ+ Criticism': 'Crítica LGBTQ+',
    'Conspiracy Theory': 'Teoria da Conspiração',
    'Review Ecosystem Critique': 'Meta-Crítica',
    'Technical Criticism': 'Crítica Técnica',
    'Tactical Praise': 'Elogio Tático',
    'Coordinated Hate': 'Ódio Coordenado',
    'Studio Criticism': 'Crítica ao Estúdio',
    'Frustration/Expectation': 'Frustração/Expectativa',
    'score': 'Nota',
    'Content Warning': 'Aviso de Conteúdo',
    'Creative/Artistic Criticism': 'Crítica Artística',
    'Praise': 'Elogio',
    'Product/Character Criticism': 'Crítica ao Produto',
    'Sexual Content': 'Conteúdo Sexual',
    'toxicity': 'Toxicidade'
}

# --- Classes e Funções de Utilitário ---

class Tee:
    """Classe para duplicar output (terminal + arquivo)"""
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def setup_logging():
    """Configura o sistema de log"""
    log_filename = f"analise_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Tee(log_filename)
    print(f"Log salvo em: {log_filename}\n")
    return log_filename

def normalize_scores(df, product):
    """Normaliza scores para 0-1 conforme tipo de produto"""
    df = df.copy()
    if product in games:
        df['score'] = df['score'] / 10  # Jogos (0-10)
    else:
        df['score'] = df['score'] / 5  # Filmes (0-5)
    return df

def format_label(label):
    """Formata nomes para exibição"""
    return label.replace('_', ' ').title()

def get_label_columns(df):
    """Retorna a lista de colunas de labels disponíveis"""
    label_list = [
        'Content Warning', 'Coordinated Hate', 'Creative/Artistic Criticism', 
        'Frustration/Expectation', 'Ideological Bias', 'LGBTQ+ Criticism', 
        'Praise', 'Product/Character Criticism', 'Sexual Content', 
        'Studio Criticism', 'Tactical Praise', 'Technical Criticism', 'Conspiracy Theory'
    ]
    return [col for col in label_list if col in df.columns]

def load_data():
    """Carrega e combina os dados"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = {'comprehend': {}, 'perspective': {}}
    
    for product in tqdm(products, desc="Carregando Comprehend-It"):
        filename = f"{product}_comprehend_it_classified.csv"
        filepath = os.path.join(script_dir, filename)
        try:
            df = pd.read_csv(filepath)
            df = normalize_scores(df, product)
            df['product'] = product
            df['is_review_bombed'] = df['product'].isin(review_bombed)
            df['r_date'] = pd.to_datetime(df['r_date'])
            data['comprehend'][product] = df
        except FileNotFoundError:
            print(f"\nArquivo não encontrado: {filename}")
    
    for product in tqdm(products, desc="Carregando Perspective"):
        filename = f"{product}_toxicity_analysis.csv"
        filepath = os.path.join(script_dir, filename)
        try:
            df = pd.read_csv(filepath)
            df['product'] = product
            data['perspective'][product] = df
        except FileNotFoundError:
            print(f"\nArquivo não encontrado: {filename}")
    
    if not data['comprehend'] or not data['perspective']:
        print("Erro: Não foi possível carregar os dados.")
        return pd.DataFrame()

    combined_comprehend = pd.concat(data['comprehend'].values(), ignore_index=True)
    combined_perspective = pd.concat(data['perspective'].values(), ignore_index=True)
    
    combined_df = pd.merge(
        combined_comprehend,
        combined_perspective[['review', 'toxicity']],
        on='review',
        how='inner'
    )
    
    return combined_df

# --- Funções de Análise ---

def analyze_toxic_groups_with_scores(df):
    """
    Analisa grupos de reviews tóxicos separando por notas altas e baixas.
    Retorna os dados prontos para o plot de linha.
    """
    high_score_threshold = df['score'].quantile(0.75)
    low_score_threshold = df['score'].quantile(0.25)
    
    # Criando os 4 grupos de interesse
    toxic_groups = {
        'bombed_high_score': df[df['is_review_bombed'] & (df['score'] >= high_score_threshold)].sort_values('toxicity', ascending=False),
        'bombed_low_score': df[df['is_review_bombed'] & (df['score'] <= low_score_threshold)].sort_values('toxicity', ascending=False),
        'normal_high_score': df[~df['is_review_bombed'] & (df['score'] >= high_score_threshold)].sort_values('toxicity', ascending=False),
        'normal_low_score': df[~df['is_review_bombed'] & (df['score'] <= low_score_threshold)].sort_values('toxicity', ascending=False)
    }
    
    results = {}
    top_n_list = np.arange(50, 501, 50)
    labels = get_label_columns(df)
    
    for group_name, group_df in toxic_groups.items():
        if group_df.empty:
            print(f"Grupo {group_name} está vazio, pulando a análise.")
            results[group_name] = pd.DataFrame(columns=labels, index=top_n_list)
            continue
            
        group_results = {}
        for n in tqdm(top_n_list, desc=f"Analisando {group_name}"):
            if n > len(group_df):
                break
            top_toxic = group_df.head(n)
            label_means = top_toxic[labels].mean()
            group_results[n] = label_means
        
        results[group_name] = pd.DataFrame(group_results).T
    
    return results

def plot_toxic_trends(results):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharey=True)

    plot_data = {
        'bombed_high_score': {'ax': axes[0, 0], 'title': 'Grupo Atacados - Reviews Positivas Tóxicas'},
        'normal_high_score': {'ax': axes[0, 1], 'title': 'Grupo Regulares - Reviews Positivas Tóxicas'},
        'bombed_low_score': {'ax': axes[1, 0], 'title': 'Grupo Atacados - Reviews Negativas Tóxicas'},
        'normal_low_score': {'ax': axes[1, 1], 'title': 'Grupo Regulares - Reviews Negativas Tóxicas'}
    }

    # 🔹 3 rótulos especiais com cores fortes fixas
    special_color_map = {
        'Coordinated Hate': '#E41A1C',   # vermelho forte
        'Ideological Bias': '#377EB8',   # azul forte
        'LGBTQ+ Criticism': '#984EA3'    # roxo forte
    }

    # 🔹 Coleta todos os rótulos existentes
    all_original_labels = set()
    for df in results.values():
        if df is not None and not df.empty:
            all_original_labels.update(df.columns.tolist())

    # 🔹 Rótulos não especiais
    other_labels = sorted([lab for lab in all_original_labels if lab not in special_color_map])

    # 🔹 Gera cores claras contrastantes para os outros
    if other_labels:
        light_palette = sns.color_palette("Set3", n_colors=len(other_labels))
        other_label_map = {lab: to_hex(light_palette[i]) for i, lab in enumerate(other_labels)}
    else:
        other_label_map = {}

    # --- Loop de plots
    for group_name, data in plot_data.items():
        ax = data['ax']
        df_to_plot = results[group_name]

        if df_to_plot is not None and not df_to_plot.empty:
            df_translated = df_to_plot.rename(columns=label_translation)
            orig_cols = df_to_plot.columns.tolist()

            color_list = []
            for orig in orig_cols:
                if orig in special_color_map:
                    color = special_color_map[orig]   # forte sempre
                else:
                    color = other_label_map.get(orig, '#cccccc')  # cor clara
                color_list.append(color)

            df_translated.plot(
                ax=ax, kind='line', marker='o',
                color=color_list, linewidth=2, markersize=6
            )
            ax.set_title(data['title'], fontsize=16)
            ax.set_xlabel('Tamanho da Amostra (Top N mais Tóxicos)')
            ax.set_ylabel('Média do Score do Rótulo')
            ax.grid(True, linestyle='--', alpha=0.35)
            ax.legend(title='Rótulo', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xticks(df_translated.index)
        else:
            ax.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center', fontsize=16)
            ax.set_title(data['title'])
            ax.set_xlabel('Tamanho da Amostra')
            ax.set_ylabel('Média do Score do Rótulo')

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig('evolucao_labels_toxicas.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráficos de evolução das labels salvos como 'evolucao_labels_toxicas.png'.")


def main():
    log_file = setup_logging()
    print("=== ANÁLISE DE TENDÊNCIAS DE LABELS EM REVIEWS TÓXICAS ===")
    
    # 1. Carrega os dados
    print("\nCarregando dados...")
    df = load_data()
    
    if df.empty:
        return
        
    # 2. Analisa a evolução das médias das labels em amostras crescentes
    print("\nCalculando médias das labels em amostras de 50 a 500 reviews mais tóxicas...")
    toxic_results = analyze_toxic_groups_with_scores(df)
    
    # 3. Plota os gráficos de linha
    print("\nGerando gráficos de linha...")
    plot_toxic_trends(toxic_results)
    
    print("\nAnálise concluída. Resultados salvos nos gráficos.")

if __name__ == "__main__":
    main()

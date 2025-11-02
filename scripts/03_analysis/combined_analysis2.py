# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys
import statsmodels.api as sm
from scipy.stats import pearsonr
import matplotlib.ticker as ticker
import logging

# =========================================================================
# --- CONSTANTES DE CONFIGURAÇÃO ---
# =========================================================================

# Diretórios base dos resultados
BASE_DIR_PERSPECTIVE = os.path.join("data", "perspective")
BASE_DIR_COMPREHEND = os.path.join("data", "comprehend-it")

# Sufixos de arquivo
PERSPECTIVE_SUFFIX = "_toxicity_analysis.csv"
COMPREHEND_SUFFIX = "_comprehend_it_classified.csv"

# Lista de produtos
PRODUCTS = [
    'captain_marvel', 'days_gone', 'inception', 'last_of_us_part_2',
    'logan', 'red_dead_redemption_2', 'resident_evil_7', 'tlj'
]

# Categorização de produtos
REVIEW_BOMBED = ['captain_marvel', 'last_of_us_part_2', 'tlj']
GAMES = ['days_gone', 'last_of_us_part_2', 'red_dead_redemption_2', 'resident_evil_7']
NORMAL = [p for p in PRODUCTS if p not in REVIEW_BOMBED]
MOVIES = [p for p in PRODUCTS if p not in GAMES]

# Nomes completos para plotagem
PRODUCT_FULL_NAMES = {
    'captain_marvel': 'Captain Marvel', 
    'days_gone': 'Days Gone', 
    'inception': 'Inception', 
    'last_of_us_part_2': 'The Last of Us Part II',
    'logan': 'Logan', 
    'red_dead_redemption_2': 'Red Dead Redemption 2',
    'resident_evil_7': 'Resident Evil 7',
    'tlj': 'Star Wars: The Last Jedi'
}

# Colunas de Score da Perspective API (devem ser mapeadas para minúsculas)
PERSPECTIVE_ATTRIBUTES = [
    'TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 
    'INSULT', 'PROFANITY', 'THREAT'
]

# Labels usadas na classificação do Comprehend-It
COMPREHEND_LABELS = [
    'Ideological Bias', 'LGBTQ+ Criticism', 'Conspiracy Theory',
    'Review Ecosystem Critique', 'Technical Criticism', 'Tactical Praise',
    'Coordinated Hate', 'Studio Criticism', 'Frustration/Expectation'
]

# --- PARÂMETROS PARA O ÍNDICE DE SUSPEITA ---
TOXICITY_THRESHOLD = 0.3  # Score mínimo de toxicidade para considerar "tóxico"
LOW_SCORE_THRESHOLD = 3.0 # Score máximo de avaliação para considerar "baixo" (após normalização)
# --------------------------------------------

# --- Configurações de Plot e Exibição ---
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
sns.set_palette("pastel")
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12,
    'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.titlepad': 20,
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black'
})

# =========================================================================
# --- FUNÇÕES AUXILIARES ---
# =========================================================================

def setup_logging():
    """Configura o logging para salvar a saída do console em um arquivo."""
    log_file = f"analysis_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Log iniciado. Saída salva em: {log_file}")
    return log_file

def print_header(text):
    """Imprime um cabeçalho formatado no log e no console."""
    logging.info("\n" + "=" * 80)
    logging.info(f"--- {text.upper()} ---")
    logging.info("=" * 80)

def load_csv(filepath):
    """Carrega arquivo CSV com tratamento de encoding."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            return df
        except Exception:
            continue
    raise Exception(f"Falha ao carregar o arquivo {filepath} com múltiplos encodings.")

def normalize_score(score, product_slug):
    """Normaliza o score de 0-10 (jogos) para 1-5."""
    # Apenas jogos tem scores 0-10, e a normalização se aplica se o score > 5
    if product_slug in GAMES and score > 5:
        return score / 2.0
    return score

# =========================================================================
# --- FUNÇÃO PRINCIPAL DE CARREGAMENTO E PROCESSAMENTO ---
# =========================================================================

def load_data():
    """
    Carrega, processa e combina os resultados classificados de todos os produtos.
    Retorna o DataFrame combinado.
    """
    print_header("CARREGAMENTO E COMBINAÇÃO DE DADOS (PERSPECTIVE + COMPREHEND-IT)")
    all_combined_data = []
    
    for product_slug in PRODUCTS:
        
        path_perspective = os.path.join(
            BASE_DIR_PERSPECTIVE, 
            f"{product_slug}{PERSPECTIVE_SUFFIX}"
        )
        path_comprehend = os.path.join(
            BASE_DIR_COMPREHEND, 
            f"{product_slug}{COMPREHEND_SUFFIX}"
        )
        
        logging.info(f"Processando: {product_slug}")
        
        # --- Carregar Perspective ---
        df_perspective = pd.DataFrame()
        if os.path.exists(path_perspective):
            try:
                df_perspective = load_csv(path_perspective)
                cols_to_keep = ['r_id', 'r_date', 'score', 'review'] + [attr.lower() for attr in PERSPECTIVE_ATTRIBUTES]
                df_perspective = df_perspective[[col for col in cols_to_keep if col in df_perspective.columns]]
            except Exception as e:
                logging.error(f"   [ERRO] Falha ao carregar Perspective para {product_slug}: {e}")
        else:
            logging.warning(f"   [AVISO] Arquivo Perspective não encontrado: {path_perspective}")

        # --- Carregar Comprehend-It ---
        df_comprehend = pd.DataFrame()
        if os.path.exists(path_comprehend):
            try:
                df_comprehend = load_csv(path_comprehend)
                cols_to_keep = ['r_id'] + COMPREHEND_LABELS
                df_comprehend = df_comprehend[[col for col in cols_to_keep if col in df_comprehend.columns]]
                df_comprehend = df_comprehend.drop_duplicates(subset=['r_id'])
            except Exception as e:
                logging.error(f"   [ERRO] Falha ao carregar Comprehend-It para {product_slug}: {e}")
        else:
            logging.warning(f"   [AVISO] Arquivo Comprehend-It não encontrado: {path_comprehend}")

        
        # --- Combinar os DataFrames ---
        if df_perspective.empty:
            continue
            
        if not df_comprehend.empty:
            df_combined = pd.merge(df_perspective, df_comprehend, on='r_id', how='left')
        else:
            df_combined = df_perspective.copy()
            
        # --- Processamento Comum ---
        if not df_combined.empty:
            df_combined['product'] = product_slug
            df_combined['formatted_product'] = PRODUCT_FULL_NAMES.get(product_slug, product_slug.replace('_', ' ').title())
            
            # Normalização de scores
            if 'score' in df_combined.columns:
                 df_combined['score'] = df_combined['score'].apply(lambda s: normalize_score(s, product_slug))
            
            # Conversão de data
            if 'r_date' in df_combined.columns:
                df_combined['r_date'] = pd.to_datetime(df_combined['r_date'], errors='coerce')
                df_combined = df_combined.dropna(subset=['r_date'])
                
            all_combined_data.append(df_combined)

    if not all_combined_data:
        logging.error("\nFALHA: Não foi possível carregar dados para combinação de nenhum produto.")
        return pd.DataFrame() 

    # Combina todos os DataFrames
    combined_df = pd.concat(all_combined_data, ignore_index=True)
    
    logging.info(f"\nCarregamento e Combinação concluídos. Total de reviews: {len(combined_df)}")
    
    # Adiciona coluna de grupo (Review Bombed vs. Normal)
    combined_df['group'] = combined_df['product'].apply(
        lambda p: 'Atacado' if p in REVIEW_BOMBED else 'Regular'
    )
    
    return combined_df

# =========================================================================
# --- FUNÇÕES DE ANÁLISE DO REVIEW BOMBING ---
# =========================================================================

def calculate_suspicion_index(df):
    """
    Calcula o Índice de Suspeita com base no Score de Toxicidade (Perspective) 
    e no Score de Avaliação (Baixo).
    
    O índice é: (Toxicidade > Threshold) AND (Score <= Low Score Threshold)
    """
    print_header(f"CÁLCULO DO ÍNDICE DE SUSPEITA (Toxicidade > {TOXICITY_THRESHOLD} & Score <= {LOW_SCORE_THRESHOLD})")
    
    # Cria a coluna booleana para reviews suspeitas
    df['is_suspicious'] = (
        (df['toxicity'] > TOXICITY_THRESHOLD) & 
        (df['score'] <= LOW_SCORE_THRESHOLD)
    )
    
    # Agrupa por produto e calcula métricas
    summary = df.groupby('product').agg(
        total_reviews=('r_id', 'count'),
        low_score_reviews=('score', lambda x: (x <= LOW_SCORE_THRESHOLD).sum()),
        toxic_reviews=('toxicity', lambda x: (x > TOXICITY_THRESHOLD).sum()),
        suspicious_reviews=('is_suspicious', 'sum'),
        mean_score=('score', 'mean'),
        mean_toxicity=('toxicity', 'mean')
    ).reset_index()
    
    # Calcula a Porcentagem de Suspeita
    summary['suspicion_percentage'] = (summary['suspicious_reviews'] / summary['total_reviews']) * 100
    
    # Calcula o Índice de Suspeita (Log-Ratio do índice vs. média de reviews)
    mean_suspicion_rate = summary['suspicious_reviews'].sum() / summary['total_reviews'].sum()
    summary['suspicion_index'] = summary.apply(
        lambda row: (row['suspicious_reviews'] / row['total_reviews']) / mean_suspicion_rate,
        axis=1
    )
    
    # Aplica log para melhor visualização
    summary['suspicion_index_log'] = np.log10(summary['suspicion_index'])
    
    # Adiciona a classificação
    summary['classification'] = summary['product'].apply(
        lambda p: 'Atacado' if p in REVIEW_BOMBED else 'Regular'
    )
    
    logging.info("\n--- Resumo do Índice de Suspeita por Produto ---\n")
    logging.info(summary[['product', 'classification', 'mean_score', 'mean_toxicity', 'suspicion_percentage', 'suspicion_index_log']].to_string(float_format="%.4f"))

    return summary

def plot_suspicion_index(summary):
    """Gera um gráfico de barras do Índice de Suspeita."""
    print_header("PLOTAGEM DO ÍNDICE DE SUSPEITA")
    
    # Adiciona o nome completo do produto para o plot
    summary['product_full_name'] = summary['product'].map(PRODUCT_FULL_NAMES)

    summary = summary.sort_values('suspicion_index_log', ascending=False)
    plt.figure(figsize=(12, 8))
    
    sns.barplot(
        x='suspicion_index_log',
        y='product_full_name',
        hue='classification',
        data=summary,
        palette={'Atacado': 'red', 'Regular': 'blue'}
    )
    
    # Garante que o formato dos números no eixo X seja simples, sem notação científica
    # e que a label seja clara (Log10)
    plt.ticklabel_format(style='plain', axis='x')
    
    plt.xlabel('Índice de Suspeita (Log10)')
    plt.ylabel("")
    plt.legend(title='Grupo')
    plt.title('Índice de Suspeita de Review Bombing', pad=20)
    plt.tight_layout()
    plt.savefig('suspicion_index_log_bar.png', dpi=300)
    plt.close()
    logging.info("Gráfico do índice de suspeita salvo como 'suspicion_index_log_bar.png'.")

# =========================================================================
# --- PRINCIPAL ---
# =========================================================================

def main():
    log_file = setup_logging()
    logging.info("=== ANÁLISE COMPLETA DO REVIEW BOMBING E ÍNDICE DE SUSPEITA ===")
    
    # 1. Carrega os dados
    df = load_data()
    
    if df.empty:
        logging.error("Processo de análise encerrado, pois nenhum dado foi carregado.")
        return

    # 2. Calcula o Índice de Suspeita
    summary_df = calculate_suspicion_index(df)

    # 3. Plota o Índice de Suspeita
    plot_suspicion_index(summary_df)
    
    logging.info("\n" + "#"*80)
    logging.info("FIM DAS ANÁLISES. Verifique a saída no console e os gráficos PNG gerados.")
    logging.info("#"*80)

if __name__ == '__main__':
    # Usar try/except para capturar erros e garantir que o programa termine
    try:
        main()
    except Exception as e:
        logging.error(f"Erro fatal no script: {e}", exc_info=True)
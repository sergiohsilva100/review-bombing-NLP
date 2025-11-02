# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr
import sys
import io

# Configuração para tentar usar UTF-8 no console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configuração inicial para plots e exibição
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)
pd.set_option('display.max_colwidth', 500) 

# Configurações de texto e cores
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

# --- Configurações de Diretório e Arquivos ---

# Diretórios base dos resultados, conforme solicitado
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

# Categorização dos produtos (para análises de review bombing)
REVIEW_BOMBED = ['captain_marvel', 'last_of_us_part_2', 'tlj']
NORMAL = [p for p in PRODUCTS if p not in REVIEW_BOMBED]

# Definição de quais produtos são jogos (para normalização de scores)
GAMES = ['days_gone', 'last_of_us_part_2', 'red_dead_redemption_2', 'resident_evil_7']

# Colunas de Score da Perspective API (devem ser mapeadas para minúsculas)
PERSPECTIVE_ATTRIBUTES = [
    'toxicity', 'severe_toxicity', 'identity_attack', 
    'insult', 'profanity', 'threat'
]

# Colunas de Labels do Comprehend-It
COMPREHEND_LABELS = [
    'Ideological Bias', 'LGBTQ+ Criticism', 'Conspiracy Theory',
    'Review Ecosystem Critique', 'Technical Criticism', 'Tactical Praise',
    'Coordinated Hate', 'Studio Criticism', 'Frustration/Expectation'
]

# Mapeamento de labels para português (opcional, para visualização)
LABEL_TRANSLATION = {
    'Ideological Bias': 'Viés Ideológico',
    'LGBTQ+ Criticism': 'Crítica LGBTQ+',
    'Conspiracy Theory': 'Teoria da Conspiração',
    'Review Ecosystem Critique': 'Crítica ao Ecossistema de Reviews',
    'Technical Criticism': 'Crítica Técnica',
    'Tactical Praise': 'Elogio Tático',
    'Coordinated Hate': 'Ódio Coordenado',
    'Studio Criticism': 'Crítica ao Estúdio',
    'Frustration/Expectation': 'Frustração/Expectativa'
}

# --- Funções Auxiliares ---

def print_header(text):
    """Imprime um cabeçalho formatado."""
    print("\n" + "=" * 80)
    print(f"--- {text.upper()} ---")
    print("=" * 80)

def load_csv(filepath):
    """Carrega arquivo CSV com tratamento de encoding."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            # Tenta carregar com o cabeçalho existente
            df = pd.read_csv(filepath, encoding=encoding)
            return df
        except Exception:
            continue
    raise Exception(f"Falha ao carregar o arquivo {filepath} com múltiplos encodings.")

def normalize_score(score, product_slug):
    """Normaliza o score de 0-10 (jogos) para 1-5."""
    if product_slug in GAMES and score > 5:
        return score / 2.0
    return score

def format_product_name(product_slug):
    """Formata o slug do produto para um nome mais legível."""
    names = {
        'captain_marvel': 'Captain Marvel', 'days_gone': 'Days Gone', 
        'inception': 'Inception', 'last_of_us_part_2': 'The Last of Us Part II',
        'logan': 'Logan', 'red_dead_redemption_2': 'Red Dead Redemption 2', 
        'resident_evil_7': 'Resident Evil 7', 'tlj': 'Star Wars: The Last Jedi'
    }
    return names.get(product_slug, product_slug.replace('_', ' ').title())

# --- Função Principal de Carregamento ---

def load_data():
    """
    Carrega, processa e combina os resultados classificados de todos os produtos.
    Retorna o DataFrame combinado.
    """
    print_header("CARREGAMENTO E COMBINAÇÃO DE DADOS (PERSPECTIVE + COMPREHEND-IT)")
    all_combined_data = []
    
    for product_slug in PRODUCTS:
        
        # 1. Caminhos dos arquivos
        path_perspective = os.path.join(
            BASE_DIR_PERSPECTIVE, 
            f"{product_slug}{PERSPECTIVE_SUFFIX}"
        )
        path_comprehend = os.path.join(
            BASE_DIR_COMPREHEND, 
            f"{product_slug}{COMPREHEND_SUFFIX}"
        )
        
        print(f"Processando: {product_slug}")
        
        # --- Carregar Perspective ---
        df_perspective = pd.DataFrame()
        if os.path.exists(path_perspective):
            try:
                df_perspective = load_csv(path_perspective)
                # Seleciona as colunas essenciais do Perspective
                cols_to_keep = ['r_id', 'r_date', 'score', 'review'] + [attr.lower() for attr in PERSPECTIVE_ATTRIBUTES]
                df_perspective = df_perspective[[col for col in cols_to_keep if col in df_perspective.columns]]
            except Exception as e:
                print(f"   [ERRO] Falha ao carregar Perspective para {product_slug}: {e}")
        else:
            print(f"   [AVISO] Arquivo Perspective não encontrado: {path_perspective}")


        # --- Carregar Comprehend-It ---
        df_comprehend = pd.DataFrame()
        if os.path.exists(path_comprehend):
            try:
                df_comprehend = load_csv(path_comprehend)
                # Seleciona as colunas essenciais do Comprehend
                cols_to_keep = ['r_id'] + COMPREHEND_LABELS
                df_comprehend = df_comprehend[[col for col in cols_to_keep if col in df_comprehend.columns]]
                # Renomeia colunas para o merge (se houver) e garante o r_id como chave
                df_comprehend = df_comprehend.rename(columns={c: c.replace(' ', '_') for c in df_comprehend.columns if ' ' in c})
                df_comprehend = df_comprehend.drop_duplicates(subset=['r_id'])
            except Exception as e:
                print(f"   [ERRO] Falha ao carregar Comprehend-It para {product_slug}: {e}")
        else:
            print(f"   [AVISO] Arquivo Comprehend-It não encontrado: {path_comprehend}")

        
        # --- Combinar os DataFrames ---
        if df_perspective.empty and df_comprehend.empty:
            continue
            
        if not df_perspective.empty and not df_comprehend.empty:
            # Merge baseado no identificador único da review ('r_id')
            df_combined = pd.merge(df_perspective, df_comprehend, on='r_id', how='inner')
        elif not df_perspective.empty:
            df_combined = df_perspective.copy()
        elif not df_comprehend.empty:
            # Não deve acontecer, pois Comprehend não tem info de score/review se Perspective falhar
            continue 
            
        # --- Processamento Comum ---
        if not df_combined.empty:
            df_combined['product'] = product_slug
            df_combined['formatted_product'] = format_product_name(product_slug)
            
            # Normalização de scores
            if 'score' in df_combined.columns:
                 df_combined['score'] = df_combined['score'].apply(lambda s: normalize_score(s, product_slug))
            
            # Conversão de data (assumindo a coluna 'r_date')
            if 'r_date' in df_combined.columns:
                df_combined['r_date'] = pd.to_datetime(df_combined['r_date'], errors='coerce')
                df_combined = df_combined.dropna(subset=['r_date'])
                
            all_combined_data.append(df_combined)

    if not all_combined_data:
        print("\nFALHA: Não foi possível carregar dados para combinação de nenhum produto.")
        return pd.DataFrame() 

    # Combina todos os DataFrames
    combined_df = pd.concat(all_combined_data, ignore_index=True)
    
    print(f"\nCarregamento e Combinação concluídos. Total de reviews: {len(combined_df)}")
    
    # Adiciona coluna de grupo (Review Bombed vs. Normal)
    combined_df['group'] = combined_df['product'].apply(
        lambda p: 'Atacado' if p in REVIEW_BOMBED else 'Regular'
    )
    
    return combined_df

# =========================================================================
# FUNÇÕES DE ANÁLISE COMBINADA
# =========================================================================

def correlation_analysis(df):
    """Calcula e exibe a correlação entre as métricas de Perspective e Comprehend-It."""
    print_header("ANÁLISE DE CORRELAÇÃO (PERSPECTIVE VS. COMPREHEND-IT)")
    
    # Colunas de interesse (Perspective vs. Comprehend-It)
    perspective_cols = [attr.lower() for attr in PERSPECTIVE_ATTRIBUTES]
    comprehend_cols = [label.replace(' ', '_') for label in COMPREHEND_LABELS]
    
    # Cria uma matriz de correlação cruzada
    correlation_matrix = df[perspective_cols + comprehend_cols].corr()
    
    # Filtra apenas a correlação cruzada (Perspective vs. Comprehend-It)
    cross_corr = correlation_matrix.loc[perspective_cols, comprehend_cols]
    
    print("\n--- Correlação de Pearson entre Scores de Toxicidade e Scores de Labels ---\n")
    print(cross_corr.to_string(float_format="%.4f"))

    # Plot da correlação entre TOXICITY e Viés Ideológico
    if 'toxicity' in df.columns and 'Ideological_Bias' in df.columns:
        corr_val, p_val = pearsonr(df['toxicity'].dropna(), df['Ideological_Bias'].dropna())
        
        plt.figure(figsize=(8, 6))
        sns.regplot(x='toxicity', y='Ideological_Bias', data=df, scatter_kws={'alpha':0.2})
        plt.title(f'Toxicidade vs. Viés Ideológico (Corr: {corr_val:.4f}, p: {p_val:.4e})', pad=20)
        plt.xlabel('Score de Toxicidade (Perspective)')
        plt.ylabel('Score de Viés Ideológico (Comprehend-It)')
        plt.tight_layout()
        plt.savefig('correlation_toxicity_ideological_bias.png', dpi=300)
        plt.close()
        print("\nGráfico de Correlação salvo como 'correlation_toxicity_ideological_bias.png'.")

def top_toxic_labels_analysis(df, top_n=500):
    """
    Analisa as labels de Comprehend-It nas reviews classificadas como mais tóxicas
    pela Perspective API.
    """
    print_header(f"ANÁLISE DAS LABELS DO COMPREHEND-IT NAS {top_n} REVIEWS MAIS TÓXICAS")
    
    # 1. Filtra as N reviews com maior score de 'toxicity'
    top_toxic_df = df.nlargest(top_n, 'toxicity', keep='first')
    
    if top_toxic_df.empty:
        print(f"AVISO: Menos de {top_n} reviews com score de 'toxicity' válido.")
        return
        
    # Colunas de labels do Comprehend-It (assumindo a renomeação)
    comprehend_cols = [label.replace(' ', '_') for label in COMPREHEND_LABELS]
    
    # 2. Calcula a média dos scores das labels do Comprehend-It
    label_means = top_toxic_df[comprehend_cols].mean().sort_values(ascending=False)
    
    # Mapeamento para português
    label_means.index = label_means.index.map({k.replace(' ', '_'): v for k, v in LABEL_TRANSLATION.items()})

    print("\n--- Média de Scores de Labels nas Reviews Mais Tóxicas ---\n")
    print(label_means.to_string(float_format="%.4f"))
    
    # 3. Plota os resultados
    plt.figure(figsize=(10, 7))
    sns.barplot(x=label_means.values, y=label_means.index, palette='magma', orient='h')
    plt.title(f'Média dos Scores de Labels do Comprehend-It (Top {top_n} Reviews Mais Tóxicas)', pad=20)
    plt.xlabel('Média do Score da Label')
    plt.ylabel('Rótulo (Comprehend-It)')
    plt.tight_layout()
    plt.savefig(f'top_toxic_labels_mean.png', dpi=300)
    plt.close()
    print("\nGráfico salvo como 'top_toxic_labels_mean.png'.")


def main():
    
    # 1. Carregar e combinar os dados
    combined_df = load_data() 
    
    if combined_df.empty:
        print("\nProcesso de análise encerrado, pois nenhum dado foi carregado.")
        return

    print("\n" + "#"*80)
    print("INICIANDO AS ANÁLISES COMBINADAS")
    print("#"*80)
    
    # 2. Executar as análises
    correlation_analysis(combined_df)
    top_toxic_labels_analysis(combined_df, top_n=500)
    
    print("\n" + "#"*80)
    print("FIM DAS ANÁLISES. Verifique a saída no console e os gráficos PNG gerados.")
    print("#"*80)

if __name__ == '__main__':
    main()
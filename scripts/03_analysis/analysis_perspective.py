# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy.stats import mannwhitneyu
import sys
import io

# Configuração para tentar usar UTF-8 no console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configuração inicial para plots e exibição
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500) # Exibe o conteúdo completo da review nos prints

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

# Slug do modelo específico para este script de análise (nome da pasta)
MODEL_SLUG = "perspective"
# Diretório base dos resultados, conforme solicitado: data/perspective
BASE_CLASSIFIED_DIR = os.path.join("data", MODEL_SLUG)
# Sufixo do arquivo, conforme solicitado: <produto>_toxicity_analysis.csv
FILE_SUFFIX = "_toxicity_analysis.csv"

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

# Nomes das colunas de score da Perspective (em minúsculo)
ATTRIBUTES = [
    'toxicity', 'severe_toxicity', 'identity_attack', 
    'insult', 'profanity', 'threat'
]

# --- Funções Auxiliares ---

def print_header(text):
    """Imprime um cabeçalho formatado."""
    print("\n" + "=" * 70)
    print(f"--- {text.upper()} ---")
    print("=" * 70)

def load_csv(filepath):
    """Carrega arquivo CSV com tratamento de encoding."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            # Assumindo que os arquivos salvos têm cabeçalho
            df = pd.read_csv(filepath, encoding=encoding)
            return df
        except Exception:
            continue
    raise Exception(f"Falha ao carregar o arquivo {filepath} com múltiplos encodings.")

def normalize_score(score, product_slug):
    """Normaliza o score de 0-10 (jogos) para 1-5."""
    # Aplicável apenas se a coluna 'score' existir e for de 0-10
    if product_slug in GAMES and score > 5:
        return score / 2.0
    return score

def format_product_name(product_slug):
    """Formata o slug do produto para um nome mais legível."""
    names = {
        'captain_marvel': 'Captain Marvel',
        'days_gone': 'Days Gone',
        'inception': 'Inception',
        'last_of_us_part_2': 'The Last of Us Part II',
        'logan': 'Logan',
        'red_dead_redemption_2': 'Red Dead Redemption 2',
        'resident_evil_7': 'Resident Evil 7',
        'tlj': 'Star Wars: The Last Jedi'
    }
    return names.get(product_slug, product_slug.replace('_', ' ').title())


# --- Função Principal de Carregamento ---

def load_data():
    """
    Carrega e combina os resultados classificados de toxicidade de todos os produtos.
    """
    print_header("CARREGAMENTO DE DADOS DE TOXICIDADE")
    all_data = []
    
    for product_slug in PRODUCTS:
        # Caminho conforme solicitado: data/perspective/X_toxicity_analysis.csv
        input_file = os.path.join(
            BASE_CLASSIFIED_DIR, 
            f"{product_slug}{FILE_SUFFIX}"
        )
        
        print(f"Tentando carregar: {input_file}")

        if not os.path.exists(input_file):
            print(f"AVISO: Arquivo não encontrado: {input_file}. Pulando.")
            continue
            
        try:
            df = load_csv(input_file)
            
            # Limpeza e formatação
            df['product'] = product_slug
            df['formatted_product'] = format_product_name(product_slug)
            
            # Normalização de scores (se a coluna 'score' existir)
            if 'score' in df.columns:
                df['score'] = df.apply(lambda row: normalize_score(row['score']), axis=1)

            # Conversão de data (assumindo a coluna 'r_date')
            if 'r_date' in df.columns:
                df['r_date'] = pd.to_datetime(df['r_date'], errors='coerce')
                df = df.dropna(subset=['r_date']) # Remove linhas sem data válida
            
            # Adiciona ao pool de dados
            all_data.append(df)
            
        except Exception as e:
            print(f"ERRO ao processar {input_file}: {e}. Pulando.")
            continue

    if not all_data:
        print("FALHA: Não foi possível carregar dados de nenhum produto.")
        return pd.DataFrame() # Retorna DataFrame vazio

    # Combina todos os DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nCarregamento concluído. Total de reviews: {len(combined_df)}")
    
    # Adiciona coluna de grupo (Review Bombed vs. Normal)
    combined_df['group'] = combined_df['product'].apply(
        lambda p: 'Atacado' if p in REVIEW_BOMBED else 'Regular'
    )
    
    return combined_df


# =========================================================================
# FUNÇÕES DE ANÁLISE ESPECÍFICAS PARA TOXICIDADE
# =========================================================================

def basic_analysis(df, model_name):
    """Análise estatística básica dos scores de toxicidade."""
    print_header(f"ANÁLISE ESTATÍSTICA BÁSICA ({model_name.upper()})")
    
    # Agrupa por produto e calcula a média dos scores de toxicidade
    mean_scores_product = df.groupby('formatted_product')[ATTRIBUTES].mean().T
    print("\n--- Média dos Scores de Toxicidade por Produto ---\n")
    print(mean_scores_product.to_string(float_format="%.4f"))
    
    # Média geral por grupo
    print("\n--- Média de Scores de Toxicidade por Grupo ---\n")
    mean_by_group = df.groupby('group')[ATTRIBUTES].mean().T
    print(mean_by_group.to_string(float_format="%.4f"))

    # Teste de Mann-Whitney U para TOXICITY (Atacado vs Regular)
    print("\n--- Teste de Mann-Whitney U (Atacado vs Regular) para TOXICITY ---")
    score_bombed = df[df['group'] == 'Atacado']['toxicity'].dropna()
    score_normal = df[df['group'] == 'Regular']['toxicity'].dropna()
    
    if len(score_bombed) > 20 and len(score_normal) > 20:
        stat, p = mannwhitneyu(score_bombed, score_normal, alternative='two-sided')
        significance = "Estatísticamente Significativa" if p < 0.05 else "Não Significativa"
        
        print(f"Atributo: TOXICITY")
        print(f"  P-value: {p:.4e} ({significance})")
        print(f"  Média Atacado: {score_bombed.mean():.4f}")
        print(f"  Média Regular: {score_normal.mean():.4f}")
    else:
        print(f"Atributo: TOXICITY - Dados insuficientes para teste.")
        
    print("-" * 70)


def score_bin_toxicity_analysis(df, model_name):
    """Analisa a toxicidade média agrupada pela pontuação (score) da review."""
    print_header(f"TOXICIDADE MÉDIA POR PONTUAÇÃO DA REVIEW ({model_name.upper()})")

    # Garante que 'score' é inteiro e está no intervalo 1-5
    df_clean = df.dropna(subset=['score'])
    df_clean['score_bin'] = df_clean['score'].astype(int)
    df_clean = df_clean[df_clean['score_bin'].between(1, 5)]

    if df_clean.empty:
        print("AVISO: Dados de 'score' inválidos ou insuficientes.")
        return

    # Calcula a média dos atributos de toxicidade por score
    toxicity_by_score = df_clean.groupby('score_bin')[ATTRIBUTES].mean().T
    print("\n--- Média dos Scores de Toxicidade agrupada pelo Score da Review ---\n")
    print(toxicity_by_score.to_string(float_format="%.4f"))
    
    # Plota a toxicidade média por score
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x='score_bin', 
        y='toxicity', 
        data=df_clean, 
        estimator='mean', 
        errorbar=('ci', 95), # Intervalo de confiança
        marker='o'
    )
    plt.title(f'Toxicidade Média vs. Score da Review ({model_name.upper()})', pad=20)
    plt.xlabel('Score da Review (1-5)')
    plt.ylabel('Média do Score de Toxicidade')
    plt.xticks(sorted(df_clean['score_bin'].unique()))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'toxicity_by_score_mean_{model_name}.png', dpi=300)
    plt.close()
    print(f"\nGráfico de toxicidade média por score salvo como 'toxicity_by_score_mean_{model_name}.png'.")


def score_toxicity_scatter_plot(df, model_name):
    """Gera um scatter plot de Score vs. Toxicidade."""
    
    # Filtra scores baixos (suspeitos de review bombing) e altos
    df_low_score = df[df['score'] <= 2.5]
    df_high_score = df[df['score'] >= 4.5]

    plt.figure(figsize=(12, 7))
    
    # Plota reviews com score baixo
    plt.scatter(
        df_low_score['score'], 
        df_low_score['toxicity'], 
        alpha=0.2, 
        s=50, 
        label='Score Baixo (< 3)', 
        color='red'
    )
    # Plota reviews com score alto
    plt.scatter(
        df_high_score['score'], 
        df_high_score['toxicity'], 
        alpha=0.2, 
        s=50, 
        label='Score Alto (> 4)', 
        color='blue'
    )
    
    plt.title(f'Relação entre Score da Review e Toxicidade ({model_name.upper()})', pad=20)
    plt.xlabel('Score da Review')
    plt.ylabel('Score de Toxicidade')
    plt.xlim(0.5, 5.5)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'score_toxicity_scatter_{model_name}.png', dpi=300)
    plt.close()
    print(f"Gráfico de dispersão salvo como 'score_toxicity_scatter_{model_name}.png'.")


def top_toxic_reviews(df):
    """Exibe as reviews mais tóxicas, separadas por grupo (Atacado vs. Regular)."""
    print_header("REVIEWS COM MAIOR SCORE DE TOXICIDADE")
    
    bombed_df = df[df['group'] == 'Atacado'].copy()
    normal_df = df[df['group'] == 'Regular'].copy()

    # Exibe os 15 reviews mais tóxicos para o grupo "Atacado"
    print("\n--- 15 REVIEWS MAIS TÓXICOS DO GRUPO 'ATACADO' ---")
    if not bombed_df.empty:
        # Seleciona as colunas a exibir e ordena
        top_toxic_bombed = bombed_df.nlargest(15, 'toxicity')[['formatted_product', 'score', 'toxicity', 'review']]
        print(top_toxic_bombed.to_string())
    else:
        print("Nenhum review encontrado no grupo 'Atacado'.")
    print("-" * 50)

    # Exibe os 15 reviews mais tóxicos para o grupo "Regular"
    print("\n--- 15 REVIEWS MAIS TÓXICOS DO GRUPO 'REGULAR' ---")
    if not normal_df.empty:
        top_toxic_normal = normal_df.nlargest(15, 'toxicity')[['formatted_product', 'score', 'toxicity', 'review']]
        print(top_toxic_normal.to_string())
    else:
        print("Nenhum review encontrado no grupo 'Regular'.")
    print("-" * 50)


# --- Função Principal de Execução ---

def main():
    
    model_name = "perspective_api"
    
    # 1. Carregar os dados
    combined_df = load_data() 
    
    if combined_df.empty:
        print("\nProcesso de análise encerrado.")
        return

    print("\n" + "#"*70)
    print("INICIANDO AS ANÁLISES DE DADOS DE TOXICIDADE (PERSPECTIVE API)")
    print("#"*70)
    
    # 2. Executar as análises
    basic_analysis(combined_df, model_name)
    score_bin_toxicity_analysis(combined_df, model_name)
    score_toxicity_scatter_plot(combined_df, model_name)
    top_toxic_reviews(combined_df)
    
    # 3. Análise de correlação (Exemplo)
    correlation = combined_df[['score', 'toxicity']].corr().iloc[0, 1]
    print_header(f"CORRELAÇÃO ENTRE SCORE E TOXICIDADE")
    print(f"Coeficiente de Correlação de Pearson (Score vs. Toxicity): {correlation:.4f}")
    print("-" * 70)


    print("\n" + "#"*70)
    print("FIM DAS ANÁLISES. Verifique os gráficos PNG gerados.")
    print("#"*70)

if __name__ == '__main__':
    main()
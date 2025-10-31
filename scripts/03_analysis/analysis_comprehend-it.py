import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy.stats import mannwhitneyu
import sys
import io

# Configuração para tentar usar UTF-8 no console do Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configuração inicial
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
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

# Lista de produtos
products = [
    'captain_marvel', 'days_gone', 'inception', 'last_of_us_part_2',
    'logan', 'red_dead_redemption_2', 'resident_evil_7', 'tlj'
]

# Categorização dos produtos
review_bombed = ['captain_marvel', 'last_of_us_part_2', 'tlj']
normal = [p for p in products if p not in review_bombed]

# Definição de quais produtos são jogos (para normalização)
games = ['days_gone', 'last_of_us_part_2', 'red_dead_redemption_2', 'resident_evil_7']
movies = [p for p in products if p not in games]

# Mapeamento de labels para português (apenas para visualização)
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
    'score': 'Nota'
}

# Novo mapeamento para nomes de produtos na exibição
product_display_names = {
    'tlj': 'Star Wars The Last Jedi',
    'captain_marvel': 'Captain Marvel',
    'days_gone': 'Days Gone',
    'inception': 'Inception',
    'last_of_us_part_2': 'Last of Us Part 2',
    'logan': 'Logan',
    'red_dead_redemption_2': 'Red Dead Redemption 2',
    'resident_evil_7': 'Resident Evil 7'
}

def normalize_scores(df, product):
    """Normaliza todas as notas para escala 0-1"""
    df = df.copy()
    if product in games:
        df['score'] = df['score'] / 10
    else:
        df['score'] = df['score'] / 5
    return df

def format_label(label):
    """Remove underscores e formata labels para exibição, incluindo tradução de produtos"""
    if label in product_display_names:
        return product_display_names[label]
    return label.replace('_', ' ').title()

def load_data():
    """Carrega todos os arquivos CSV e retorna um dicionário de DataFrames"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = {}

    for product in products:
        filename = f"{product}_comprehend_it_classified.csv"
        filepath = os.path.join(script_dir, filename)

        try:
            df = pd.read_csv(filepath)
            df['product'] = product
            df['formatted_product'] = product_display_names.get(product, format_label(product))
            df['r_date'] = pd.to_datetime(df['r_date'])
            df['is_review_bombed'] = df['product'].isin(review_bombed)
            df['grupo'] = df['is_review_bombed'].map({True: 'Atacados', False: 'Regulares'})

            df = normalize_scores(df, product)

            data[product] = df
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {filename}")

    combined_df = pd.concat(data.values(), ignore_index=True)

    return data, combined_df

def basic_analysis(combined_df):
    """Análises básicas dos dados combinados"""
    print("\n=== ANÁLISE BÁSICA ===")

    print("\n1. Contagem de reviews por produto:")
    print(combined_df['product'].value_counts())

    print("\n1.1 Contagem de usuários únicos:")
    print("Atacados:", combined_df[combined_df['is_review_bombed']]['r_id'].nunique())
    print("Regulares:", combined_df[~combined_df['is_review_bombed']]['r_id'].nunique())

    plt.figure(figsize=(14, 8))
    sns.boxplot(x='formatted_product', y='score', data=combined_df,
                palette={'Atacados': 'red', 'Regulares': 'blue'},
                hue='grupo',
                showfliers=False)
    plt.xlabel('')
    plt.ylabel('Nota (0-1)')
    plt.xticks(rotation=45)
    plt.legend(title='Grupo')
    plt.tight_layout()
    plt.savefig('scores_distribution.png', dpi=300)
    plt.close()

    print("\n2. Estatísticas descritivas das notas (0-1):")
    print(combined_df['score'].describe())

    print("\n3. Comparação entre grupos (Atacados vs Regulares):")
    print("\nAtacados:")
    print(combined_df[combined_df['is_review_bombed']]['score'].describe())
    print("\nRegulares:")
    print(combined_df[~combined_df['is_review_bombed']]['score'].describe())

    from scipy.stats import ttest_ind
    bombed_scores = combined_df[combined_df['is_review_bombed']]['score']
    normal_scores = combined_df[~combined_df['is_review_bombed']]['score']
    t_stat, p_value = ttest_ind(bombed_scores, normal_scores)
    print(f"\nTeste t: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    original_labels = [
        'Ideological Bias', 'LGBTQ+ Criticism', 'Conspiracy Theory',
        'Review Ecosystem Critique', 'Technical Criticism', 'Tactical Praise',
        'Coordinated Hate', 'Studio Criticism', 'Frustration/Expectation'
    ]

    available_labels = [col for col in original_labels if col in combined_df.columns]

    print("\n4. Correlação entre labels e score do usuário:")

    corr_matrix_original_names = combined_df[available_labels + ['score']].corr()
    corr_matrix_translated = corr_matrix_original_names.rename(columns=label_translation, index=label_translation)

    print(corr_matrix_translated['Nota'].sort_values(ascending=False))

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix_translated, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                annot_kws={"size": 10})
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.close()

    # --- HEATMAPS DE CORRELAÇÃO LADO A LADO ---
    print("\n5. Heatmaps de Correlação entre Rótulos por Grupo (Lado a Lado):")

    bombed_df = combined_df[combined_df['is_review_bombed']].copy()
    normal_df = combined_df[~combined_df['is_review_bombed']].copy()

    # Cria a figura e os eixos para 2 subplots (1 linha, 2 colunas)
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharex=True, sharey=True)
    fig.suptitle('Correlação entre Rótulos por Grupo', fontsize=16, y=1.02)

    if not bombed_df.empty and available_labels:
        bombed_corr = bombed_df[available_labels].corr()
        bombed_corr_translated = bombed_corr.rename(columns=label_translation, index=label_translation)
        sns.heatmap(bombed_corr_translated, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                    annot_kws={"size": 10}, ax=axes[0])
        axes[0].set_title('Grupo Atacados')

    if not normal_df.empty and available_labels:
        normal_corr = normal_df[available_labels].corr()
        normal_corr_translated = normal_corr.rename(columns=label_translation, index=label_translation)
        sns.heatmap(normal_corr_translated, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                    annot_kws={"size": 10}, ax=axes[1])
        axes[1].set_title('Grupo Regulares')

    plt.tight_layout()
    plt.savefig('correlation_heatmaps_grouped.png', dpi=300)
    plt.close()
    print("\nHeatmaps para os grupos 'Atacados' e 'Regulares' salvos como 'correlation_heatmaps_grouped.png'.")
    # --- FIM DO NOVO TRECHO ---

def calculate_days_since_release(combined_df):
    """Calcula dias desde o lançamento para cada produto"""
    release_dates = combined_df.groupby('product')['r_date'].min().to_dict()

    combined_df['days_since_release'] = combined_df.apply(
        lambda row: (row['r_date'] - release_dates[row['product']]).days,
        axis=1
    )

    return combined_df

def temporal_analysis(combined_df):
    """Análise temporal dos dados com eixo X em dias reais e suavização reforçada"""
    print("\n=== ANÁLISE TEMPORAL ===")

    combined_df = calculate_days_since_release(combined_df)

    colors = {
        'captain_marvel': '#FF6B6B',
        'tlj': '#FFD166',
        'last_of_us_part_2': '#4ECDC4',
        'days_gone': '#06D6A0',
        'inception': '#7F7F7F',
        'logan': '#FF9A76',
        'red_dead_redemption_2': '#A26769',
        'resident_evil_7': '#B8B8B8'
    }

    plt.figure(figsize=(16, 8))

    for product in products:
        product_df = combined_df[combined_df['product'] == product].copy()
        product_df = product_df.sort_values('days_since_release')

        daily_avg = product_df.groupby('days_since_release')['score'].mean().reset_index()

        daily_avg['smoothed_score'] = daily_avg['score'].rolling(window=30, min_periods=1, center=True).mean()

        line_alpha = 0.9 if product in review_bombed else 0.7
        linewidth = 2.5 if product in review_bombed else 1.5

        plt.plot(daily_avg['days_since_release'], daily_avg['smoothed_score'],
                 color=colors[product],
                 label=format_label(product),
                 linewidth=linewidth,
                 alpha=line_alpha)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Dias desde o lançamento')
    plt.ylabel('Nota (0-1)')
    plt.xlim(0, combined_df['days_since_release'].max())
    plt.tight_layout()
    plt.savefig('temporal_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

    relevant_labels = ['Coordinated Hate', 'Ideological Bias', 'LGBTQ+ Criticism']
    available_relevant_labels = [col for col in relevant_labels if col in combined_df.columns]

    for product in review_bombed:
        product_df = combined_df[combined_df['product'] == product].copy()
        product_df = product_df.sort_values('days_since_release')

        plt.figure(figsize=(16, 8))

        for label in available_relevant_labels:
            daily_avg = product_df.groupby('days_since_release')[label].mean().reset_index()

            daily_avg[f'smoothed_{label}'] = daily_avg[label].rolling(window=30, min_periods=1, center=True).mean()

            plt.plot(daily_avg['days_since_release'], daily_avg[f'smoothed_{label}'],
                     label=label_translation.get(label, label), linewidth=2)

        plt.xlabel('Dias desde o lançamento')
        plt.ylabel('Score da Label')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim(0, product_df['days_since_release'].max())
        plt.tight_layout()
        plt.savefig(f'temporal_labels_{product}.png', dpi=300, bbox_inches='tight')
        plt.close()

def label_comparison(combined_df):
    """Comparação das labels entre produtos com e sem review bombing"""
    print("\n=== COMPARAÇÃO DE LABELS ===")

    original_labels = [
        'Ideological Bias', 'LGBTQ+ Criticism', 'Conspiracy Theory',
        'Review Ecosystem Critique', 'Technical Criticism', 'Tactical Praise',
        'Coordinated Hate', 'Studio Criticism', 'Frustration/Expectation'
    ]
    available_labels = [col for col in original_labels if col in combined_df.columns]

    melted_df = combined_df.melt(id_vars=['is_review_bombed'], value_vars=available_labels,
                                 var_name='Label', value_name='Score')
    melted_df['Grupo'] = melted_df['is_review_bombed'].map({True: 'Atacados', False: 'Regulares'})

    melted_df['Label'] = melted_df['Label'].map(lambda x: label_translation.get(x, x))

    plt.figure(figsize=(16, 8))
    sns.boxplot(x='Label', y='Score', hue='Grupo', data=melted_df,
                palette={'Atacados': 'red', 'Regulares': 'blue'},
                showfliers=False)
    plt.xlabel('')
    plt.ylabel('Probabilidade de pertencimento')
    plt.xticks(rotation=45)
    plt.legend(title='Grupo', loc='upper center')
    plt.tight_layout()
    plt.savefig('label_comparison.png', dpi=300)
    plt.close()

def calculate_bias_percentages(combined_df, threshold=0.5):
    negative_reviews = combined_df[combined_df['score'] < combined_df['score'].mean()]
    print("\nContagem de reviews negativas por grupo:")
    print(negative_reviews['is_review_bombed'].value_counts())

    mean_score_atacados = combined_df[combined_df['is_review_bombed']]['score'].mean()
    mean_score_regulares = combined_df[~combined_df['is_review_bombed']]['score'].mean()
    negative_reviews_atacados = combined_df[(combined_df['is_review_bombed']) & (combined_df['score'] < mean_score_atacados)]
    negative_reviews_regulares = combined_df[(~combined_df['is_review_bombed']) & (combined_df['score'] < mean_score_regulares)]

    print("\nViés Ideológico (>0.5) usando médias por grupo:")
    print(f"Atacados: {(negative_reviews_atacados['Ideological Bias'] > 0.5).mean() * 100:.1f}%")
    print(f"Regulares: {(negative_reviews_regulares['Ideological Bias'] > 0.5).mean() * 100:.1f}%")

def review_bombing_characteristics(combined_df, threshold=0.5):
    """
    Análise específica das características do review bombing
    """
    relevant_labels = ['Coordinated Hate', 'Ideological Bias', 'LGBTQ+ Criticism']
    available_relevant_labels = [col for col in relevant_labels if col in combined_df.columns]

    plt.figure(figsize=(14, 7))
    sns.scatterplot(x='Ideological Bias', y='score', hue='grupo',
                    alpha=0.3, data=combined_df,
                    palette={'Atacados': 'red', 'Regulares': 'blue'})
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Limite ({threshold})')
    plt.xlabel('Viés Ideológico')
    plt.ylabel('Nota (0-1)')
    plt.legend(title='Grupo')
    plt.tight_layout()
    plt.savefig('bias_vs_score.png', dpi=300)
    plt.close()

    plt.figure(figsize=(14, 7))
    for label in available_relevant_labels:
        for group, group_df in combined_df.groupby('is_review_bombed'):
            group_label = 'Atacados' if group else 'Regulares'
            color = 'red' if group else 'blue'
            sns.ecdfplot(data=group_df, x=label,
                         label=f'{group_label} - {label_translation.get(label, label)}',
                         color=color, linewidth=2)

    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Limite ({threshold})')
    plt.xlabel('Probabilidade do Rótulo')
    plt.ylabel('Proporção Acumulada')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cdf_labels.png', dpi=300)
    plt.close()

def product_specific_analysis(data):
    """Análises específicas para cada produto"""
    print("\n=== ANÁLISES ESPECÍFICAS POR PRODUTO ===")

    for product, df in data.items():
        print(f"\nProduto: {format_label(product)}")

        if 'Ideological Bias' in df.columns:
            print("\nTop 10 reviews com maior Viés Ideológico:")
            print(df.nlargest(10, 'Ideological Bias')[['score', 'review', 'Ideological Bias']])
        else:
            print("\nColuna 'Ideological Bias' não encontrada no DataFrame")
        
        # --- NOVO TRECHO PARA INCEPTION ---
        if product == 'inception' and 'Conspiracy Theory' in df.columns:
            print("\nTop 5 reviews de Inception com maior Teoria da Conspiração:")
            top_inception_reviews = df.nlargest(5, 'Conspiracy Theory')
            for index, row in top_inception_reviews.iterrows():
                print(f"Teoria da Conspiração: {row['Conspiracy Theory']:.4f}")
                print(f"Nota: {row['score']:.2f}")
                print(f"Review: {row['review']}\n")
        # --- FIM DO NOVO TRECHO ---

        if 'Technical Criticism' in df.columns:
            corr = df['Technical Criticism'].corr(df['score'])
            print(f"\nCorrelação entre Crítica Técnica e Nota: {corr:.2f}")
        else:
            print("\nColuna 'Technical Criticism' não encontrada no DataFrame")

def analyze_individual_product_labels(combined_df):
    """
    Calcula e mostra a média das probabilidades de cada rótulo
    para cada obra individualmente, incluindo um gráfico e as top reviews.
    """
    print("\n=== MÉDIAS DE PROBABILIDADES POR OBRA ===")

    relevant_labels = [
        'Ideological Bias', 'LGBTQ+ Criticism', 'Conspiracy Theory',
        'Review Ecosystem Critique', 'Technical Criticism', 'Tactical Praise',
        'Coordinated Hate', 'Studio Criticism', 'Frustration/Expectation'
    ]

    for product in products:
        product_df = combined_df[combined_df['product'] == product].copy()
        
        # Nome formatado do produto para os prints e gráficos
        product_name = product_display_names.get(product, product)
        print(f"\n--- Produto: {product_name} ---")

        # Verifica quais rótulos estão realmente no DataFrame
        available_labels = [label for label in relevant_labels if label in product_df.columns]

        if not available_labels:
            print("Nenhum rótulo relevante encontrado para este produto.")
            continue

        # Calcula a média dos rótulos
        mean_labels = product_df[available_labels].mean().sort_values(ascending=False)
        mean_labels_translated = mean_labels.rename(index=label_translation)

        # Imprime as médias
        print("\n** Médias de Probabilidades: **")
        print(mean_labels_translated.to_string())

        # Criação do gráfico de barras para as médias
        plt.figure(figsize=(12, 7))
        sns.barplot(x=mean_labels_translated.values, y=mean_labels_translated.index,
                    palette='viridis', orient='h')
        plt.title(f'Média de Probabilidades dos Rótulos para {product_name}', pad=20)
        plt.xlabel('Média da Probabilidade')
        plt.ylabel('Rótulo')
        plt.tight_layout()
        plt.savefig(f'label_means_{product}.png', dpi=300)
        plt.close()

        # Imprime as 3 reviews com maior 'Viés Ideológico'
        print("\n** Top 3 Reviews com Maior 'Viés Ideológico': **")
        if 'Ideological Bias' in product_df.columns:
            top_reviews = product_df.nlargest(3, 'Ideological Bias')
            for index, row in top_reviews.iterrows():
                print(f"Viés Ideológico: {row['Ideological Bias']:.4f}")
                print(f"Review: {row['review']}\n")
        else:
            print("Coluna 'Ideological Bias' não encontrada.")


def main():
    # Carrega os dados
    data, combined_df = load_data()

    # Executa as análises
    basic_analysis(combined_df)
    temporal_analysis(combined_df)
    label_comparison(combined_df)
    calculate_bias_percentages(combined_df, threshold=0.5)
    review_bombing_characteristics(combined_df, threshold=0.5)
    product_specific_analysis(data)

    # Nova função adicionada
    analyze_individual_product_labels(combined_df)

    print("\nAnálises concluídas. Gráficos salvos no diretório atual.")


if __name__ == "__main__":
    main()
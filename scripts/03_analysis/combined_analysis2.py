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

# Cores para visualização
product_colors = {
    'captain_marvel': '#FF6B6B',
    'tlj': '#FFD166',
    'last_of_us_part_2': '#4ECDC4',
    'days_gone': '#06D6A0',
    'inception': '#7F7F7F',
    'logan': '#FF9A76',
    'red_dead_redemption_2': '#A26769',
    'resident_evil_7': '#B8B8B8'
}

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
    
    def flush(self, *args, **kwargs):
        self.file.flush()
        self.stdout.flush()

def setup_logging():
    log_filename = f"analise_completa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Tee(log_filename)
    print(f"Log salvo em: {log_filename}\n")
    return log_filename

def normalize_scores(df, product):
    df = df.copy()
    if product in games:
        df['score'] = df['score'] / 10  # Jogos (0-10)
    else:
        df['score'] = df['score'] / 5  # Filmes (0-5)
    return df

def format_label(label):
    return label.replace('_', ' ').title()

def get_label_columns(df):
    label_list = [
        'Content Warning', 'Coordinated Hate', 'Creative/Artistic Criticism', 
        'Frustration/Expectation', 'Ideological Bias', 'LGBTQ+ Criticism', 
        'Praise', 'Product/Character Criticism', 'Sexual Content', 
        'Studio Criticism', 'Tactical Praise', 'Technical Criticism', 'Conspiracy Theory'
    ]
    return [col for col in label_list if col in df.columns]

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = {'comprehend': {}, 'perspective': {}}
    
    for product in tqdm(products, desc="Carregando Comprehend-It"):
        filename = f"{product}_comprehend_it_classified.csv"
        filepath = os.path.join(script_dir, filename)
        try:
            df = pd.read_csv(filepath)
            df = normalize_scores(df, product)
            df['product'] = product
            df['formatted_product'] = format_label(product)
            df['r_date'] = pd.to_datetime(df['r_date'])
            df['is_review_bombed'] = df['product'].isin(review_bombed)
            data['comprehend'][product] = df
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {filename}")
    
    for product in tqdm(products, desc="Carregando Perspective"):
        filename = f"{product}_toxicity_analysis.csv"
        filepath = os.path.join(script_dir, filename)
        try:
            df = pd.read_csv(filepath)
            df['product'] = product
            df['formatted_product'] = format_label(product)
            data['perspective'][product] = df
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {filename}")
    
    if not data['comprehend'] or not data['perspective']:
        print("Erro: Não foi possível carregar os dados.")
        return pd.DataFrame()

    combined_comprehend = pd.concat(data['comprehend'].values(), ignore_index=True)
    combined_perspective = pd.concat(data['perspective'].values(), ignore_index=True)
    
    combined_df = pd.merge(
        combined_comprehend,
        combined_perspective[['review', 'toxicity', 'severe_toxicity', 'identity_attack']],
        on='review',
        how='inner'
    )
    return combined_df

# --- Funções de Análise ---

## 1. Contra-ataques (Elogio Tático)
def analyze_tactical_praise(df):
    positive_reviews = df[df['score'] >= 0.7]
    negative_reviews = df[df['score'] <= 0.2]

    print("\n--- 1.1 Caracterização do 'Tactical Praise' ---")
    bombed_positive = positive_reviews[positive_reviews['is_review_bombed'] == True]
    normal_positive = positive_reviews[positive_reviews['is_review_bombed'] == False]

    if not bombed_positive.empty:
        tp_metrics = bombed_positive[['Tactical Praise', 'Ideological Bias', 'toxicity']].mean()
        print("Médias - Positivas Atacadas:\n", tp_metrics.round(3))

        corr_tp = bombed_positive[['toxicity', 'Tactical Praise']].corr().iloc[0, 1]
        print(f"Correlação Tactical Praise x Toxicidade (Positivas Atacadas): {corr_tp:.3f}")
    else:
        print("Nenhuma avaliação positiva em produtos atacados.")

    if not normal_positive.empty:
        normal_metrics = normal_positive[['Tactical Praise', 'Ideological Bias', 'toxicity']].mean()
        print("Médias - Positivas Regulares:\n", normal_metrics.round(3))

        # Cálculo da correlação para produtos regulares
        corr_tp_normal = normal_positive[['toxicity', 'Tactical Praise']].corr().iloc[0, 1]
        print(f"Correlação Tactical Praise x Toxicidade (Positivas Regulares): {corr_tp_normal:.3f}")

    else:
        print("Nenhuma avaliação positiva em produtos regulares.")

    if not normal_positive.empty:
        normal_metrics = normal_positive[['Tactical Praise', 'Ideological Bias', 'toxicity']].mean()
        print("Médias - Positivas Regulares:\n", normal_metrics.round(3))

    print("\n--- 1.2 Polarização (Ataque vs. Defesa Ideológica) ---")
    df['attack_bias'] = negative_reviews['Ideological Bias'] + negative_reviews['Coordinated Hate']
    df['defense_bias'] = positive_reviews['Tactical Praise'] + positive_reviews['Ideological Bias']

    polar = pd.DataFrame({
        'Ataque': df.groupby('is_review_bombed')['attack_bias'].mean(),
        'Defesa': df.groupby('is_review_bombed')['defense_bias'].mean()
    })
    print(polar.round(3))
    polar.plot(kind='bar', figsize=(8, 6))
    plt.title('Polarização (Ataque vs Defesa)')
    plt.ylabel('Média dos Scores')
    plt.savefig('polarizacao.png', dpi=150)
    plt.close()

    # --- NOVO: EXEMPLOS DE REVIEWS TÓXICOS ---
    print("\n--- 1.3 Exemplos de Reviews Tóxicos ---")

    # Amostras para Ataque
    attack_samples = negative_reviews[negative_reviews['is_review_bombed'] & (negative_reviews['toxicity'] > 0.5)].nlargest(5, 'toxicity', keep='first')
    print("\n5 Reviews de 'Ataque': (Tóxicos, Negativos, Atacados)")
    if not attack_samples.empty:
        for _, row in attack_samples.iterrows():
            print(f"Review: {row['review']}\n  - Toxicidade: {row['toxicity']:.2f}, Viés Ideológico: {row['Ideological Bias']:.2f}, Ódio Coordenado: {row['Coordinated Hate']:.2f}\n")
    else:
        print("Nenhum review de 'Ataque' encontrado com toxicidade > 0.5.")

    # Amostras para Defesa
    defense_samples = positive_reviews[positive_reviews['is_review_bombed'] & (positive_reviews['toxicity'] > 0.5)].nlargest(5, 'toxicity', keep='first')
    print("\n5 Reviews de 'Defesa': (Tóxicos, Positivos, Atacados)")
    if not defense_samples.empty:
        for _, row in defense_samples.iterrows():
            print(f"Review: {row['review']}\n  - Toxicidade: {row['toxicity']:.2f}, Elogio Tático: {row['Tactical Praise']:.2f}, Viés Ideológico: {row['Ideological Bias']:.2f}\n")
    else:
        print("Nenhum review de 'Defesa' encontrado com toxicidade > 0.5.")

    # Amostras para Regulares
    regular_samples = df[~df['is_review_bombed'] & (df['toxicity'] > 0.5)].nlargest(5, 'toxicity', keep='first')
    print("\n5 Reviews do Grupo 'Regulares': (Tóxicos)")
    if not regular_samples.empty:
        for _, row in regular_samples.iterrows():
            print(f"Review: {row['review']}\n  - Toxicidade: {row['toxicity']:.2f}, Elogio Tático: {row['Tactical Praise']:.2f}, Viés Ideológico: {row['Ideological Bias']:.2f}\n")
    else:
        print("Nenhum review 'Regular' encontrado com toxicidade > 0.5.")
    # --- FIM DO NOVO TRECHO ---

## 2. Correlação e Mediação
def analyze_conditional_correlation(df):
    bombed = df[df['is_review_bombed'] == True]
    normal = df[df['is_review_bombed'] == False]

    labels = ['LGBTQ+ Criticism', 'Frustration/Expectation', 'Technical Criticism', 'Conspiracy Theory']

    print("\n--- 2. Correlações Condicionais ---")
    for label in labels:
        if label in bombed.columns:
            corr = bombed[['toxicity', 'Ideological Bias', label]].corr()
            print(f"Atacados - {label}:\n{corr.round(3)}")
        if label in normal.columns:
            corr = normal[['toxicity', 'Ideological Bias', label]].corr()
            print(f"Regulares - {label}:\n{corr.round(3)}")

    print("\n--- 2.2 Regressão Múltipla (Medição) ---")
    low_score = bombed[bombed['score'] <= 0.2].copy()
    if not low_score.empty:
        X = low_score[['Ideological Bias', 'toxicity']].copy()
        y = low_score['score'].copy()

        # Limpeza de NaN/Inf
        before_len = len(X)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        after_len = len(X)
        removed = before_len - after_len

        print(f"Linhas removidas por NaN/Inf: {removed}")

        if len(X) > 0:
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            print(model.summary())
        else:
            print("Não há dados suficientes após limpeza para regressão múltipla.")

# --- NOVO: CORRELAÇÕES DE TOXICIDADE E HEATMAPS (REFEITO) ---
def analyze_toxicity_correlations(df):
    """
    Gera dois heatmaps (Atacados x Regulares) mostrando a matriz de correlação
    (inclui 'toxicity' e os rótulos) e garante rótulos visíveis no eixo Y.
    """
    print("\n--- CORRELAÇÕES DE TOXICIDADE (heatmaps lado a lado) ---")

    bombed_df = df[df['is_review_bombed'] == True].copy()
    normal_df = df[df['is_review_bombed'] == False].copy()

    # pega os rótulos disponíveis (sem 'toxicity')
    labels_to_correlate = get_label_columns(df)
    correlation_labels = ['toxicity'] + labels_to_correlate

    # checa se há colunas suficientes
    present_overall = [c for c in correlation_labels if c in df.columns]
    if len(present_overall) < 2:
        print("Dados insuficientes: precisa de 'toxicity' e pelo menos um rótulo presente.")
        return

    # cópia do dicionário de tradução e adição de 'toxicity'
    label_translation_with_tox = label_translation.copy()
    label_translation_with_tox.setdefault('toxicity', 'Toxicidade')

    # cria figura com 2 subplots lado a lado (não compartilhar y para garantir rótulos)
    fig, axes = plt.subplots(1, 2, figsize=(26, 12), sharey=True)

    def plot_corr_for_group(group_df, ax, title):
        cols = [c for c in correlation_labels if c in group_df.columns]
        if len(cols) < 2:
            ax.set_title(f'{title} (Dados insuficientes)')
            ax.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        corr_matrix = group_df[cols].corr(method='pearson')
        # traduz rótulos do eixo para exibição (tanto colunas quanto índices)
        translated_index = [label_translation_with_tox.get(i, i) for i in corr_matrix.index]
        translated_columns = [label_translation_with_tox.get(c, c) for c in corr_matrix.columns]
        corr_matrix_translated = corr_matrix.copy()
        corr_matrix_translated.index = translated_index
        corr_matrix_translated.columns = translated_columns

        sns.heatmap(
            corr_matrix_translated,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            center=0,
            annot_kws={"size": 10},
            ax=ax,
            cbar_kws={'label': 'Correlação'},
            xticklabels=corr_matrix_translated.columns,
            yticklabels=corr_matrix_translated.index
        )
        ax.set_title(title)

        # Rotaciona e alinha corretamente os rótulos
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

        # Ajustes finais de aparência
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    # Plota para Atacados
    plot_corr_for_group(bombed_df, axes[0], 'Grupo Atacados')

    # Plota para Regulares
    plot_corr_for_group(normal_df, axes[1], 'Grupo Regulares')

    plt.tight_layout()
    plt.savefig('toxicity_correlations_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nHeatmaps de correlação com 'toxicity' salvos como 'toxicity_correlations_heatmaps.png'.")


## 3. Perfil Discursivo por Produto (com exemplos)
def analyze_product_profiles(df):
    print("\n--- 3. Perfis Discursivos ---")

    # TLoU2
    tloa2 = df[df['product'] == 'last_of_us_part_2']
    if 'LGBTQ+ Criticism' in tloa2.columns:
        corr = tloa2[['LGBTQ+ Criticism', 'toxicity', 'Ideological Bias']].corr()
        print("TLoU2 - Correlações:\n", corr.round(3))
        top = tloa2.nlargest(1, 'LGBTQ+ Criticism')
        if not top.empty:
            print("Exemplo TLoU2:\n", top[['review', 'toxicity', 'LGBTQ+ Criticism']].to_string())

    # Inception
    inc = df[df['product'] == 'inception']
    if 'Conspiracy Theory' in inc.columns:
        corr = inc[['Conspiracy Theory', 'toxicity']].corr()
        print("Inception - Correlações:\n", corr.round(3))
        top = inc.nlargest(1, 'Conspiracy Theory')
        if not top.empty:
            print("Exemplo Inception:\n", top[['review', 'toxicity', 'Conspiracy Theory']].to_string())

    # TLJ
    tlj_df = df[df['product'] == 'tlj']
    if 'Studio Criticism' in tlj_df.columns:
        weekly = tlj_df.groupby(pd.Grouper(key='r_date', freq='W')).agg({
            'Ideological Bias': 'mean', 'Studio Criticism': 'mean', 'toxicity': 'mean'
        })
        weekly.plot(figsize=(10, 6))
        plt.title('TLJ - Flutuação Semanal')
        plt.savefig('tlj_temporal.png', dpi=150)
        plt.close()
        top = tlj_df.nlargest(1, 'Studio Criticism')
        if not top.empty:
            print("Exemplo TLJ:\n", top[['review', 'toxicity', 'Studio Criticism']].to_string())

## 4. Índice de Suspeita
def calculate_review_bombing_index(df):
    attack_reviews = df[df['score'] <= 0.5].copy()
    
    summary = attack_reviews.groupby('product').agg(
        volume=('review', 'count'),
        avg_score=('score', 'mean'),
        avg_tox=('toxicity', 'mean'),
        avg_bias=('Ideological Bias', 'mean')
    ).reset_index()

    # Novo cálculo do índice de suspeita com volume padronizado por log10
    summary['suspicion_index_log'] = (np.log10(summary['volume']) * summary['avg_bias'] * summary['avg_tox']) / summary['avg_score']

    summary['classification'] = summary['product'].apply(lambda p: 'Atacado' if p in review_bombed else 'Regular')

    print("\n--- 4. Índice de Suspeita ---")
    print("\nValores Médios para o Cálculo do Índice:")
    # A linha abaixo irá imprimir a tabela com os valores médios
    print(summary[['product', 'volume', 'avg_score', 'avg_tox', 'avg_bias']].round(4))
    print("\nÍndice de Suspeita calculado:")
    print(summary[['product', 'classification', 'suspicion_index_log']].round(4))
    
    # --- GRÁFICO HORIZONTAL COM ESCALA LINEAR ---
    print("\n--- Gráfico do Índice de Suspeita ---")

    # Mapeamento para nomes completos dos produtos
    full_names = {
        'captain_marvel': 'Captain Marvel',
        'days_gone': 'Days Gone',
        'inception': 'Inception',
        'last_of_us_part_2': 'The Last of Us Part 2',
        'logan': 'Logan',
        'red_dead_redemption_2': 'Red Dead Redemption 2',
        'resident_evil_7': 'Resident Evil 7',
        'tlj': 'Star Wars The Last Jedi'
    }
    
    # Adiciona a nova coluna com os nomes completos
    summary['product_full_name'] = summary['product'].map(full_names)

    summary = summary.sort_values('suspicion_index_log', ascending=False)
    plt.figure(figsize=(12, 8))
    
    sns.barplot(
        x='suspicion_index_log',
        y='product_full_name',
        hue='classification',
        data=summary,
        palette={'Atacado': 'red', 'Regular': 'blue'}
    )
    
    # === ESTA É A MUDANÇA PRINCIPAL! ===
    # Removendo a escala logarítmica.
    # Apenas garantimos que o formato dos números seja simples.
    plt.ticklabel_format(style='plain', axis='x')
    
    plt.xlabel('Índice de Suspeita')
    plt.ylabel("")
    plt.legend(title='Grupo')
    plt.tight_layout()
    plt.savefig('suspicion_index_log_bar.png', dpi=300)
    plt.close()
    print("\nGráfico do índice de suspeita salvo como 'suspicion_index_log_bar.png'.")

# --- Principal ---
def main():
    log_file = setup_logging()
    print("=== ANÁLISE COMPLETA DO REVIEW BOMBING ===")
    df = load_data()
    if df.empty:
        return

    analyze_tactical_praise(df.copy())
    analyze_conditional_correlation(df.copy())
    analyze_toxicity_correlations(df.copy())
    analyze_product_profiles(df.copy())
    calculate_review_bombing_index(df.copy())

    print("\n--- CONCLUÍDO ---")
    print(f"Log: {log_file}")

if __name__ == "__main__":
    main()
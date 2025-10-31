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
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)  # Exibe o conteúdo completo da review

# Configurações de texto e cores
plt.rcParams.update({
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.titlepad': 20
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

# Dicionários de tradução
toxicity_translation = {
    'toxicity': 'Toxicidade',
    'severe_toxicity': 'Toxicidade Severa',
    'identity_attack': 'Ataque Identitário',
    'insult': 'Insulto',
    'profanity': 'Profanidade',
    'threat': 'Ameaça',
    'Score': 'Nota',
    'Pontuação': 'Probabilidade'
}

# Mapeamento de nomes de produtos para exibição na legenda
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
        df['score'] = df['score'] / 10  # Jogos vão de 0-10 → 0-1
    else:
        df['score'] = df['score'] / 5   # Filmes vão de 0-5 → 0-1
    return df

def format_label(label):
    """Formata labels para exibição, incluindo nomes de produtos"""
    if label in product_display_names:
        return product_display_names[label]
    return label.replace('_', ' ').title()

def translate_label(label):
    """Traduz labels usando os dicionários de tradução"""
    return toxicity_translation.get(label, label)

def load_data():
    """Carrega todos os arquivos de toxicidade"""
    # Para fins de simulação/exemplo, assumindo que os arquivos CSV estão no mesmo diretório
    # Se você está rodando isso em um ambiente sem os arquivos, esta função falhará.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = {}
    
    # Criando um DataFrame de exemplo se os arquivos não existirem para que o resto do código funcione
    # REMOVA ESTE BLOCO APÓS A INSTALAÇÃO DO SCRIPT DELE DE ARQUIVOS.
    if not any(os.path.exists(os.path.join(script_dir, f"{p}_toxicity_analysis.csv")) for p in products):
        print("AVISO: Criando dados de exemplo, pois os arquivos CSV não foram encontrados.")
        # Simulação de dados
        n_reviews = 1000
        combined_df = pd.DataFrame({
            'product': np.random.choice(products, n_reviews * len(products)),
            'score': np.random.rand(n_reviews * len(products)),
            'toxicity': np.random.rand(n_reviews * len(products)),
            'severe_toxicity': np.random.rand(n_reviews * len(products)) * 0.1,
            'identity_attack': np.random.rand(n_reviews * len(products)) * 0.2,
            'insult': np.random.rand(n_reviews * len(products)) * 0.3,
            'profanity': np.random.rand(n_reviews * len(products)) * 0.4,
            'threat': np.random.rand(n_reviews * len(products)) * 0.05,
            'review': [f"Review {i}" for i in range(n_reviews * len(products))],
            'r_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(n_reviews * len(products)), unit='D')
        })
        
        # Ajustando a simulação para imitar a realidade (toxicidade mais alta em notas baixas)
        combined_df.loc[combined_df['score'] < 0.2, 'toxicity'] += 0.5
        combined_df['toxicity'] = np.clip(combined_df['toxicity'], 0, 1) # Clamping para 0-1
        
        for product in products:
            product_df = combined_df[combined_df['product'] == product].copy()
            product_df['formatted_product'] = product_display_names.get(product, format_label(product))
            product_df['is_review_bombed'] = product_df['product'].isin(review_bombed)
            product_df['grupo'] = product_df['is_review_bombed'].map({True: 'Atacados', False: 'Regulares'})
            data[product] = product_df
        return data, combined_df

    # Fim do bloco de simulação
    
    # Código original de carregamento de dados (mantido)
    combined_df_list = []
    for product in products:
        filename = f"{product}_toxicity_analysis.csv"
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
            combined_df_list.append(df)
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {filename}")
    
    if combined_df_list:
        combined_df = pd.concat(combined_df_list, ignore_index=True)
    else:
        # Se nenhum arquivo for encontrado, crie um DataFrame vazio para evitar erros
        combined_df = pd.DataFrame(columns=['product', 'formatted_product', 'r_date', 'is_review_bombed', 'grupo', 'score', 'toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat', 'review'])

    return data, combined_df

def basic_analysis(combined_df, model_name):
    """Análises básicas dos dados combinados"""
    print(f"\n=== ANÁLISE BÁSICA ({model_name}) ===")
    
    print("\n1. Contagem de reviews por produto:")
    print(combined_df['product'].value_counts())
    
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
    plt.savefig(f'scores_distribution_{model_name}.png', dpi=300)
    plt.close()
    
    print("\n2. Estatísticas descritivas das notas (0-1):")
    print(combined_df['score'].describe())
    
    print("\n3. Comparação entre grupos (Atacados vs Regulares):")
    print("\nAtacados:")
    print(combined_df[combined_df['is_review_bombed']]['score'].describe())
    print("\nRegulares:")
    print(combined_df[~combined_df['is_review_bombed']]['score'].describe())
    
    from scipy.stats import ttest_ind
    if not combined_df.empty:
        bombed_scores = combined_df[combined_df['is_review_bombed']]['score']
        normal_scores = combined_df[~combined_df['is_review_bombed']]['score']
        # Verifica se há dados suficientes em ambos os grupos
        if len(bombed_scores) > 1 and len(normal_scores) > 1:
            t_stat, p_value = ttest_ind(bombed_scores, normal_scores)
            print(f"\nTeste t: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        else:
             print("\nTeste t: Dados insuficientes para comparação de grupos (menos de 2 amostras em um ou ambos os grupos).")
    else:
        print("\nTeste t: DataFrame combinado está vazio.")


def score_bin_toxicity_analysis(combined_df, model_name):
    """
    Análise de toxicidade média por faixa de notas (0-1).
    """
    print(f"\n=== ANÁLISE DE TOXICIDADE POR FAIXA DE NOTAS ({model_name}) ===")

    if combined_df.empty:
        print("DataFrame combinado está vazio. Pulando análise por faixa de notas.")
        return

    # Define as faixas de notas (bins) de 0.1 em 0.1, de 0.0 a 1.0
    bins = np.arange(0, 1.1, 0.1)
    
    # Cria os rótulos para as faixas (ex: '0.0-0.1', '0.1-0.2', ...)
    bin_labels = [f'{i:.1f}-{i+0.1:.1f}' for i in bins[:-1]]
    
    # Segmenta as notas em faixas (inclui o limite superior para garantir que o 1.0 seja incluído)
    combined_df['score_bin'] = pd.cut(combined_df['score'], bins=bins, labels=bin_labels, right=False, include_lowest=True)
    
    # Calcula a toxicidade média para cada faixa, por grupo
    toxicity_by_score_bin = combined_df.groupby(['score_bin', 'grupo'])['toxicity'].mean().unstack()
    
    print("\nToxicidade Média por Faixa de Notas (0-1):")
    print(toxicity_by_score_bin)
    
    # Visualização
    plt.figure(figsize=(12, 7))
    toxicity_by_score_bin.plot(kind='bar', ax=plt.gca(),
                               color={'Atacados': 'red', 'Regulares': 'blue'})
    
    plt.title('Toxicidade Média por Faixa de Notas (0-1)')
    plt.xlabel('Faixa de Nota (0-1)')
    plt.ylabel(translate_label('toxicity'))
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Grupo')
    plt.tight_layout()
    plt.savefig(f'toxicity_by_score_bin_{model_name}.png', dpi=300)
    plt.close()

def score_toxicity_scatter_plot(combined_df, model_name):
    """
    Gráfico de dispersão entre Nota (Score) e Toxicidade (Toxicity) com linha de tendência.
    """
    print(f"\n=== GRÁFICO DE DISPERSÃO NOTA vs. TOXICIDADE ({model_name}) ===")

    if combined_df.empty:
        print("DataFrame combinado está vazio. Pulando gráfico de dispersão.")
        return

    plt.figure(figsize=(10, 8))

    # Cria o gráfico de dispersão com linha de tendência (regressão linear)
    # Usando jointplot para melhor visualização da distribuição marginal
    sns.jointplot(x='score', y='toxicity', data=combined_df, kind='reg',
                  joint_kws={'scatter_kws': {'alpha': 0.1, 's': 10}},
                  line_kws={'color': 'red'},
                  height=7)

    plt.suptitle('Relação entre Nota e Toxicidade', y=1.02)
    plt.xlabel('Nota (0-1)')
    plt.ylabel(translate_label('toxicity'))
    plt.tight_layout()
    plt.savefig(f'score_toxicity_scatter_{model_name}.png', dpi=300)
    plt.close()

def toxicity_analysis(combined_df, model_name):
    """Análise específica das métricas de toxicidade"""
    print(f"\n=== ANÁLISE DE TOXICIDADE ({model_name}) ===")
    
    toxicity_metrics = [
        'toxicity', 'severe_toxicity', 'identity_attack',
        'insult', 'profanity', 'threat'
    ]
    
    if combined_df.empty:
        print("DataFrame combinado está vazio. Pulando análises de toxicidade.")
        return

    print("\n1. Correlação entre toxicidade e nota:")
    corr_matrix = combined_df[toxicity_metrics + ['score']].corr()
    print(corr_matrix['score'].sort_values(ascending=False))
    
    plt.figure(figsize=(12, 10))
    plot_df = combined_df[toxicity_metrics + ['score']].rename(columns=toxicity_translation)
    sns.heatmap(plot_df.corr(), annot=True, cmap='coolwarm', center=0, fmt=".2f",
                annot_kws={"size": 10})
    plt.tight_layout()
    plt.savefig(f'toxicity_correlation_{model_name}.png', dpi=300)
    plt.close()
    
    print("\n2. Comparação de toxicidade entre grupos:")
    print(combined_df.groupby('is_review_bombed')[toxicity_metrics].mean().T)
    
    print("\n3. Diferença entre grupos (Atacados vs Regulares):")
    for metric in toxicity_metrics:
        bombed = combined_df[combined_df['is_review_bombed']][metric]
        normal = combined_df[~combined_df['is_review_bombed']][metric]
        # Verifica se há dados suficientes em ambos os grupos
        if len(bombed) > 0 and len(normal) > 0:
            stat, p = mannwhitneyu(bombed, normal, alternative='greater')
            print(f"{translate_label(metric)}: {'Significativa' if p < 0.05 else 'Não significativa'} (p={p:.4f})")
        else:
             print(f"{translate_label(metric)}: Dados insuficientes para o Teste Mann-Whitney U.")
    
    melted_df = combined_df.melt(id_vars=['grupo'],
                                 value_vars=toxicity_metrics,
                                 var_name='Metric',
                                 value_name='Score')
    
    melted_df['Metric'] = melted_df['Metric'].map(translate_label)
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Metric', y='Score', hue='grupo', data=melted_df,
                palette={'Atacados': 'red', 'Regulares': 'blue'},
                showfliers=False)
    plt.xlabel('Atributos de Toxicidade')
    plt.ylabel('Probabilidade')
    plt.xticks(rotation=45)
    plt.legend(title='Grupo')
    plt.tight_layout()
    plt.savefig(f'toxicity_comparison_{model_name}.png', dpi=300)
    plt.close()

def temporal_toxicity_analysis(combined_df, model_name):
    """Análise temporal da toxicidade com média móvel aumentada"""
    print(f"\n=== ANÁLISE TEMPORAL DE TOXICIDADE ({model_name}) ===")
    
    if combined_df.empty:
        print("DataFrame combinado está vazio. Pulando análise temporal.")
        return

    # Garante que 'r_date' é datetime e 'product' está no df
    if combined_df['r_date'].dtype != '<M8[ns]': # Check for datetime64[ns]
        combined_df['r_date'] = pd.to_datetime(combined_df['r_date'])
        
    release_dates = combined_df.groupby('product')['r_date'].min().to_dict()
    
    # Aplicação com verificação de produto
    def calculate_days_since_release(row):
        product = row['product']
        if product in release_dates:
            return (row['r_date'] - release_dates[product]).days
        return np.nan # Caso o produto não esteja no dicionário

    combined_df['days_since_release'] = combined_df.apply(calculate_days_since_release, axis=1)
    combined_df.dropna(subset=['days_since_release'], inplace=True) # Remove linhas onde o cálculo falhou

    
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
        if product_df.empty:
            continue
            
        product_df = product_df.sort_values('days_since_release')
        
        daily_avg = product_df.groupby('days_since_release')['toxicity'].mean().reset_index()
        
        daily_avg['smoothed_toxicity'] = daily_avg['toxicity'].rolling(window=30, min_periods=1, center=True).mean()
        
        line_alpha = 0.9 if product in review_bombed else 0.7
        linewidth = 2.5 if product in review_bombed else 1.5
        
        plt.plot(daily_avg['days_since_release'], daily_avg['smoothed_toxicity'],
                 color=colors[product],
                 label=format_label(product),
                 linewidth=linewidth,
                 alpha=line_alpha)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Dias desde o lançamento')
    plt.ylabel('Toxicidade Média (0-1)')
    if not combined_df.empty and not combined_df['days_since_release'].empty:
        plt.xlim(0, combined_df['days_since_release'].max())
    else:
        plt.xlim(0, 1) # Fallback para evitar erro
    plt.tight_layout()
    plt.savefig(f'temporal_toxicity_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def ideological_term_analysis_empirical(combined_df):
    """
    Análise empírica para mostrar a contagem e os reviews com maior toxicidade
    para cada grupo (Atacados e Regulares).
    """
    print("\n=== ANÁLISE EMPÍRICA DE REVIEWS DE ALTA TOXICIDADE (TOXICIDADE > 0.5) ===")

    if combined_df.empty:
        print("DataFrame combinado está vazio. Pulando análise empírica.")
        return

    # Define o limiar de toxicidade
    toxicity_threshold = 0.5

    # Filtra por grupo
    bombed_df = combined_df[combined_df['grupo'] == 'Atacados'].copy()
    normal_df = combined_df[combined_df['grupo'] == 'Regulares'].copy()

    # Contagem de reviews acima do limiar
    bombed_count = len(bombed_df[bombed_df['toxicity'] > toxicity_threshold])
    normal_count = len(normal_df[normal_df['toxicity'] > toxicity_threshold])

    print(f"\nContagem de reviews com toxicidade > {toxicity_threshold}:")
    print(f"  - Grupo 'Atacados': {bombed_count} avaliações")
    print(f"  - Grupo 'Regulares': {normal_count} avaliações")

    # Exibe os 15 reviews mais tóxicos para o grupo "Atacados" (Lógica Corrigida)
    print("\n--- 15 REVIEWS MAIS TÓXICOS DO GRUPO 'ATACADOS' ---")
    if not bombed_df.empty: # Verifica o DataFrame original
        top_toxic_bombed = bombed_df.nlargest(15, 'toxicity')[['formatted_product', 'score', 'toxicity', 'review']]
        print(top_toxic_bombed)
    else:
        print("Nenhum review encontrado no grupo 'Atacados'.")
    print("-" * 50)

    # Exibe os 15 reviews mais tóxicos para o grupo "Regulares" (Lógica Corrigida)
    print("\n--- 15 REVIEWS MAIS TÓXICOS DO GRUPO 'REGULARES' ---")
    if not normal_df.empty: # Verifica o DataFrame original
        top_toxic_normal = normal_df.nlargest(15, 'toxicity')[['formatted_product', 'score', 'toxicity', 'review']]
        print(top_toxic_normal)
    else:
        print("Nenhum review encontrado no grupo 'Regulares'.")
    print("-" * 50)


def main():
    # Carrega os dados
    data, combined_df = load_data()
    model_name = "perspective_api"
    
    # Verifica se o DataFrame combinado está vazio para pular as análises
    if combined_df.empty:
        print("AVISO: O DataFrame combinado está vazio. Verifique a existência dos arquivos CSV.")
        return

    # Executa as análises
    basic_analysis(combined_df, model_name)
    
    # NOVAS ANÁLISES
    score_bin_toxicity_analysis(combined_df, model_name)
    score_toxicity_scatter_plot(combined_df, model_name)
    
    toxicity_analysis(combined_df, model_name)
    temporal_toxicity_analysis(combined_df, model_name)
    
    # Executa a nova análise empírica
    ideological_term_analysis_empirical(combined_df)
    
    print("\nAnálises concluídas. Gráficos salvos no diretório atual.")

if __name__ == "__main__":
    main()
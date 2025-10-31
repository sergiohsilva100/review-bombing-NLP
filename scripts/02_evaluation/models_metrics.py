# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob
# Mantido, embora possa ser usado para visualização futura
import matplotlib.pyplot as plt 
import seaborn as sns 
import sys
import io
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from warnings import filterwarnings

# Configuração para ignorar warnings de concatenação/tipo
filterwarnings('ignore')

# Configuração para tentar usar UTF-8 no console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# --- Configurações Globais ---
# Range de thresholds para testar
THRESHOLDS = np.arange(0.3, 0.91, 0.05) 
TOTAL_REVIEWS = 400 # Número de linhas esperado no Gold Standard

# --- Definição das Colunas ---

# Nomes das colunas de rótulos no arquivo Gold Standard (reviews_labeled.csv)
CLASS_COLUMNS = [
    'Ideological_Bias', 'LGBTQ+_Criticism', 'Conspiracy_Theory',
    'Meta-Criticism', # Nome do rótulo GOLD para 'Review Ecosystem Critique'
    'Technical_Criticism', 'Tactical_Praise',
    'Coordinated_Hate', 'Studio_Criticism', 'Frustration/Expectation'
]

# Nomes das colunas de rótulos geradas pelos scripts Zero-Shot (nos arquivos reviews_X.csv)
MODEL_CLASS_COLUMNS_RAW = [
    'Ideological_Bias', 'LGBTQ+_Criticism', 'Conspiracy_Theory',
    'Review_Ecosystem_Critique', # Nome usado pelos modelos para 'Meta-Criticism'
    'Technical_Criticism', 'Tactical_Praise',
    'Coordinated_Hate', 'Studio_Criticism', 'Frustration_Expectation' # Nome usado pelo modelo
]

# Mapeamento do nome da coluna GOLD para o nome da coluna do arquivo do modelo (para merge/renomeação)
# Isso garante que a coluna 'Meta-Criticism' (GOLD) seja corretamente mapeada para o score
# gerado pela coluna 'Review_Ecosystem_Critique' do modelo.
GOLD_TO_MODEL_COLUMN_MAP = dict(zip(CLASS_COLUMNS, MODEL_CLASS_COLUMNS_RAW))

# --- Configurações de Caminho ---
BASE_LABELED_DIR = os.path.join("data", "labeled")
GOLD_FILE = os.path.join(BASE_LABELED_DIR, 'reviews_labeled.csv')

# Mapeamento dos slugs dos arquivos para os nomes de exibição
MODEL_SLUGS = {
    "comprehend_it": "Comprehend-It",
    "covid_twitter_bert": "Covid-Twitter-BERT",
    "deberta_v3": "DeBERTa-v3",
    "facebook": "Facebook-BART",
    "xlm_roberta": "XLM-RoBERTa"
}
MODEL_SLUGS_LIST = list(MODEL_SLUGS.keys())
MODEL_NAME_MAP = MODEL_SLUGS


# Lista global das colunas de score de todos os modelos (no formato slug_GOLD_COLUMN)
MODEL_SCORE_COLUMNS = []
for slug in MODEL_SLUGS_LIST:
    for gold_col in CLASS_COLUMNS:
        MODEL_SCORE_COLUMNS.append(f'{slug}_{gold_col}')


# --- Funções Auxiliares ---

def print_header(text):
    """Imprime um cabeçalho formatado."""
    print("\n" + "=" * 70)
    print(f"--- {text.upper()} ---")
    print("=" * 70)


def load_csv(filepath):
    """Carrega arquivo CSV com tratamento de encoding."""
    encodings = ['utf-8', 'latin-1', 'utf-16', 'iso-8859-1']
    for encoding in encodings:
        try:
            # Tentar ler com o encoding
            df = pd.read_csv(filepath, encoding=encoding)
            # Tentar inferir o delimitador se houver problemas de coluna
            if df.shape[1] == 1 and ',' in df.iloc[0, 0]:
                 df = pd.read_csv(filepath, encoding=encoding, sep=',')
            if not df.empty:
                return df
        except Exception:
            continue
    raise Exception(f"Falha ao carregar o arquivo {filepath} com múltiplos encodings.")


def load_data():
    """
    Carrega os dados Gold Standard e os resultados dos 5 modelos da pasta 'labeled'.
    Combina todos os scores em um único DataFrame para comparação.
    """
    print_header("CARREGAMENTO DE DADOS")
    print(f"Carregando Gold Standard de: {GOLD_FILE}")
    
    try:
        # Carrega o Gold Standard
        gold_df = load_csv(GOLD_FILE)
        if 'r_id' not in gold_df.columns:
             raise ValueError("O arquivo Gold Standard não contém a coluna 'r_id'.")
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o arquivo Gold Standard. {e}")
        return pd.DataFrame(), pd.DataFrame() # Retorna DataFrames vazios
        
    if gold_df.empty or len(gold_df) != TOTAL_REVIEWS:
        print(f"AVISO: DataFrame Gold Standard com tamanho inesperado ({len(gold_df)} vs {TOTAL_REVIEWS}).")
        
    # DataFrame para combinar todos os resultados. Começa com o Gold Standard e as colunas de rótulo (CLASS_COLUMNS)
    combined_df = gold_df[['r_id'] + CLASS_COLUMNS].copy()
    
    for slug, model_name in MODEL_SLUGS.items():
        model_file = os.path.join(BASE_LABELED_DIR, f'reviews_{slug}.csv')
        print(f"Carregando resultados do modelo {model_name} de: {model_file}")
        
        try:
            df_model = load_csv(model_file)
            
            RENAME_MAP = {}
            # Renomeia as colunas de score do modelo para o formato: slug_GOLD_COLUMN
            for gold_col, model_raw_col in GOLD_TO_MODEL_COLUMN_MAP.items():
                if model_raw_col in df_model.columns:
                    new_col_name = f'{slug}_{gold_col}'
                    RENAME_MAP[model_raw_col] = new_col_name
            
            if 'r_id' not in df_model.columns:
                 raise ValueError(f"O arquivo do modelo {model_name} não contém a coluna 'r_id'.")

            # Colunas que queremos manter do modelo (r_id + scores renomeados)
            cols_to_keep_model = ['r_id'] + list(RENAME_MAP.keys())
            
            # Aplica o mapeamento de renomeação
            df_model_filtered = df_model.rename(columns=RENAME_MAP)
            
            # Filtra apenas as colunas renomeadas e 'r_id'
            df_model_filtered = df_model_filtered[[col for col in RENAME_MAP.values() if col in df_model_filtered.columns] + ['r_id']]

            # Combina com o DataFrame principal usando inner join para manter apenas as reviews comuns
            combined_df = pd.merge(combined_df, df_model_filtered, on='r_id', how='inner') 

        except Exception as e:
            print(f"AVISO: Falha ao carregar ou processar o arquivo do modelo {model_name}: {e}. Pulando.")
            # Se um modelo falhar, ele não estará presente nas métricas, mas o script continua.
            continue
            
    print(f"\nCarregamento concluído. Reviews combinadas (inner join): {len(combined_df)}")
    return gold_df, combined_df # Retorna o df original (se necessário) e o df combinado para métricas

def calculate_metrics_for_threshold(df, model_slug, gold_column, score_column, threshold):
    """
    Calcula as métricas para um modelo, uma classe e um threshold específicos.
    """
    # Converter scores para classes binárias com base no threshold
    y_pred = (df[score_column] >= threshold).astype(int)
    y_true = df[gold_column]

    # Caso em que a classe não ocorre no Gold Standard (todos zeros)
    if y_true.sum() == 0:
        # Se as previsões também são todas zero, a precisão, recall e F1 seriam 1.0.
        # No entanto, se houver previsões positivas (FP), o recall é indefinido (divisão por zero).
        # É mais informativo ignorar essas classes na média geral, mas calcular a acurácia.
        
        # Concordância de Rótulo (Label Agreement) - Acurácia
        # Se a classe não existe no Gold (y_true=0), a Concordância é 1 - (proporção de 1s em y_pred)
        label_agreement = accuracy_score(y_true, y_pred)
        
        # Retorna NaN para as métricas baseadas em TP, FN, FP para evitar inflação na média.
        return {
            'Model': MODEL_NAME_MAP[model_slug],
            'Threshold': threshold,
            'Class': gold_column,
            'Precision': np.nan,
            'Recall': np.nan,
            'F1_Score': np.nan,
            'Label_Agreement': label_agreement, # Mantém a acurácia
            'True_Positives': 0,
            'True_Negatives': len(y_true),
            'False_Positives': y_pred.sum(),
            'False_Negatives': 0,
            'Gold_Count': 0
        }
        
    try:
        precision = precision_score(y_true, y_pred)
    except:
        precision = 0.0 # Se o modelo não previu nenhum 1 (denom. 0), e y_true.sum() > 0, algo está errado, mas forçamos 0.0

    # Recall e F1-Score
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Concordância de Rótulo (Label Agreement) - Acurácia
    label_agreement = accuracy_score(y_true, y_pred)
    
    # Métricas para detalhe
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # Importar confusion_matrix (ou calcular manualmente)

    return {
        'Model': MODEL_NAME_MAP[model_slug],
        'Threshold': threshold,
        'Class': gold_column,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Label_Agreement': label_agreement,
        'True_Positives': tp,
        'True_Negatives': tn,
        'False_Positives': fp,
        'False_Negatives': fn,
        'Gold_Count': y_true.sum()
    }


def get_gold_class_names(class_name):
    """Traduz os nomes das classes para português para exibição."""
    translation_map = {
        'Ideological_Bias': 'Viés Ideológico',
        'LGBTQ+_Criticism': 'Crítica LGBTQ+',
        'Conspiracy_Theory': 'Teoria da Conspiração',
        'Meta-Criticism': 'Crítica ao Ecossistema (GOLD)',
        'Technical_Criticism': 'Crítica Técnica',
        'Tactical_Praise': 'Elogio Tático',
        'Coordinated_Hate': 'Ódio Coordenado',
        'Studio_Criticism': 'Crítica ao Estúdio',
        'Frustration/Expectation': 'Frustração/Expectativa'
    }
    return translation_map.get(class_name, class_name)
    
# Função auxiliar que a sklearn.metrics não expõe diretamente
from sklearn.metrics import confusion_matrix


# --- Função Principal de Execução ---

def main():
    
    # 1. Carregar e combinar os dados
    _, combined_df = load_data() 
    
    if combined_df.empty:
        print("\n[!] Não foi possível prosseguir com o cálculo das métricas devido à falha no carregamento dos dados.")
        return

    all_results = []
    
    # 2. Iterar sobre todos os modelos, classes e thresholds
    for model_slug in MODEL_SLUGS_LIST:
        print_header(f"CALCULANDO MÉTRICAS PARA O MODELO: {MODEL_NAME_MAP[model_slug].upper()}")
        
        # Colunas de score e gold para o modelo atual
        model_results = []
        
        for gold_column in CLASS_COLUMNS:
            score_column = f'{model_slug}_{gold_column}'
            
            # Verifica se o score do modelo existe no DataFrame combinado (pode ter falhado o carregamento)
            if score_column not in combined_df.columns:
                print(f"AVISO: Coluna de score '{score_column}' não encontrada no DataFrame combinado. Pulando classe.")
                continue
            
            for threshold in THRESHOLDS:
                metrics = calculate_metrics_for_threshold(
                    combined_df, 
                    model_slug, 
                    gold_column, 
                    score_column, 
                    threshold
                )
                model_results.append(metrics)
        
        # Converte os resultados do modelo atual para DataFrame
        df_model_results = pd.DataFrame(model_results)
        
        if not df_model_results.empty:
            all_results.append(df_model_results)
            
            # 3. Análise dos Resultados por Modelo (Melhor Threshold por Classe)
            print("--- MELHORES THRESHOLDS POR CLASSE ---")
            
            # Filtra resultados válidos (sem NaN em F1-Score - classes com Gold Count=0)
            df_valid_results = df_model_results.dropna(subset=['F1_Score']).copy()
            
            if not df_valid_results.empty:
                # Encontra o threshold com o melhor F1-Score para cada classe
                best_f1_by_class = df_valid_results.loc[df_valid_results.groupby('Class')['F1_Score'].idxmax()]
                
                # Prepara para impressão
                best_f1_by_class['Class_PT'] = best_f1_by_class['Class'].apply(get_gold_class_names)
                
                # Seleciona e ordena as colunas para exibição
                display_cols = ['Class_PT', 'Threshold', 'Precision', 'Recall', 'F1_Score', 'Gold_Count']
                print(best_f1_by_class[display_cols].sort_values('F1_Score', ascending=False).to_string(index=False, float_format="%.4f"))
                
                print("-" * 70)
                
                # Média das Métricas (excluindo classes com Gold_Count=0)
                mean_precision = best_f1_by_class['Precision'].mean()
                mean_recall = best_f1_by_class['Recall'].mean()
                mean_f1 = best_f1_by_class['F1_Score'].mean()

                print(f"Média Geral (exceto classes com apenas 0s no gold):")
                print(f"  Precisão Média: {mean_precision:.4f}")
                print(f"  Recall Médio:   {mean_recall:.4f}")
                print(f"  F1-Score Médio: {mean_f1:.4f}")
                print("=" * 70)
            
            else:
                print("\n[!] Nenhum resultado por classe para exibir.")
                print("=" * 70)

    # 4. THRESHOLDS ÓTIMOS GLOBAIS
    if all_results: # Garante que há resultados para concatenar
        print_header("THRESHOLDS ÓTIMOS GLOBAIS (Média entre todos os modelos)")
        
        df_all_models_consolidated = pd.concat(all_results)
        
        # Agrupa pelo Threshold, calculando a média das métricas (F1 e Label Agreement)
        df_metrics_by_threshold = df_all_models_consolidated.groupby('Threshold')[['F1_Score', 'Label_Agreement']].mean().reset_index()
        
        # Remove linhas com NaN (ocorre se apenas classes Gold_Count=0 existirem para aquele threshold)
        df_metrics_by_threshold = df_metrics_by_threshold.dropna(subset=['F1_Score'])
        
        if not df_metrics_by_threshold.empty:
            # Melhor Threshold Global (F1-Score)
            best_threshold_f1 = df_metrics_by_threshold.loc[df_metrics_by_threshold['F1_Score'].idxmax()]
            
            # Melhor Threshold Global (Label Agreement - Acurácia)
            best_threshold_agreement = df_metrics_by_threshold.loc[df_metrics_by_threshold['Label_Agreement'].idxmax()]

            print(f"- Melhor Threshold Global para Concordância por Rótulo (Acurácia): {best_threshold_agreement['Threshold']:.2f} (Média: {best_threshold_agreement['Label_Agreement']:.4f})")
            print(f"- Melhor Threshold Global para F1-Score Médio:                  {best_threshold_f1['Threshold']:.2f} (Média: {best_threshold_f1['F1_Score']:.4f})")
            print("=" * 70)
        else:
            print("\n[!] Não foi possível calcular métricas globais.")
            print("=" * 70)
    else:
        print("\n[!] Não há resultados de modelos válidos para análise consolidada.")

if __name__ == '__main__':
    main()
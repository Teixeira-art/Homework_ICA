import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO DE DIRETÓRIOS
# ============================================================
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

for folder in [FIG_DIR, TAB_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. INGESTÃO DE DADOS E CORREÇÃO METODOLÓGICA
# ============================================================
def load_and_merge():
    # Carregamento explícito com injeção de metadados para evitar Paradoxo de Simpson
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    
    red['wine_type'] = 'red'
    white['wine_type'] = 'white'
    
    df = pd.concat([red, white], ignore_index=True)
    return df, red, white

df_all, df_red, df_white = load_and_merge()
predictors = [c for c in df_all.columns if c not in ['quality', 'wine_type']]

# ============================================================
# 2. CÁLCULO MANUAL DE PEARSON (Requisito da disciplina)
# ============================================================
def pearson_manual(x, y):
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    denom = np.sqrt(np.sum(x_c**2) * np.sum(y_c**2))
    return float(np.sum(x_c * y_c) / denom) if denom != 0 else np.nan

def compute_corr_matrix(data):
    corr = pd.DataFrame(index=predictors, columns=predictors, dtype=float)
    for fx in predictors:
        for fy in predictors:
            corr.loc[fx, fy] = pearson_manual(data[fx].to_numpy(), data[fy].to_numpy())
    return corr

# Geração das matrizes segmentadas e global
corr_all = compute_corr_matrix(df_all)
corr_red = compute_corr_matrix(df_red)
corr_white = compute_corr_matrix(df_white)

# Salvar tabelas numéricas
corr_all.to_csv(TAB_DIR / "correlation_matrix_all.csv", float_format="%.4f")
corr_red.to_csv(TAB_DIR / "correlation_matrix_red.csv", float_format="%.4f")
corr_white.to_csv(TAB_DIR / "correlation_matrix_white.csv", float_format="%.4f")

# ============================================================
# 3. OTIMIZAÇÃO VISUAL: HEATMAP TRIANGULAR (Padrão IEEE)
# ============================================================
def plot_triangular_heatmap(corr_matrix, title, filename):
    # Máscara para esconder a diagonal superior (informação duplicada)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .8},
                annot_kws={"size": 8})
    plt.title(title, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, format='pdf', bbox_inches='tight')
    plt.close()

plot_triangular_heatmap(corr_all, "Correlação de Pearson (Base Combinada)", "heatmap_all.pdf")
plot_triangular_heatmap(corr_red, "Correlação de Pearson (Vinho Tinto)", "heatmap_red.pdf")
plot_triangular_heatmap(corr_white, "Correlação de Pearson (Vinho Branco)", "heatmap_white.pdf")

# ============================================================
# 4. EXTRAÇÃO DE REDUNDÂNCIAS (Análise Bivariada)
# ============================================================
upper_pairs = []
for i, fx in enumerate(predictors):
    for fy in predictors[i+1:]:
        rho = corr_all.loc[fx, fy]
        upper_pairs.append({'Preditor_1': fx, 'Preditor_2': fy, 'rho': rho, 'abs_rho': abs(rho)})

top_pairs = pd.DataFrame(upper_pairs).sort_values('abs_rho', ascending=False)
top_pairs.head(10).to_csv(TAB_DIR / "top10_collinear_pairs.csv", index=False, float_format="%.4f")

# ============================================================
# 5. DISPERSÃO FOCADA EM COLINEARIDADE PARA O ARTIGO
# ============================================================
# Identificar as variáveis envolvidas nas top 3 maiores correlações para o Pairplot
top_features = list(pd.unique(top_pairs.head(3)[['Preditor_1', 'Preditor_2']].values.ravel()))

# Tratar quality como categoria para evitar mapeamento contínuo distorcido no Seaborn
df_all['quality_cat'] = df_all['quality'].astype('category')

g = sns.pairplot(
    df_all, 
    vars=top_features, 
    hue='quality_cat', 
    palette='viridis', 
    plot_kws={'alpha': 0.5, 's': 15, 'edgecolor': 'none'},
    diag_kws={'common_norm': False},
    height=2.5
)
g.fig.suptitle("Dispersão Bivariada dos Atributos de Maior Colinearidade", y=1.03)
g.savefig(FIG_DIR / "pairplot_top_redundant.pdf", format='pdf', bbox_inches='tight')
plt.close()

print("Execução finalizada. Insumos gráficos e tabulares gerados em /outputs/")

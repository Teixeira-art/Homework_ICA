import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations

# ============================================================
# CONFIGURAÇÃO DE DIRETÓRIOS
# ============================================================
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"


SCATTER_DIR = FIG_DIR / "scatter_pairs"

for folder in [FIG_DIR, TAB_DIR, SCATTER_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. INGESTÃO DE DADOS E CORREÇÃO METODOLÓGICA
# ============================================================
def load_and_merge():

    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    
    red["wine_type"] = "red"
    white["wine_type"] = "white"
    
    df = pd.concat([red, white], ignore_index=True)
    
    return df, red, white


df_all, df_red, df_white = load_and_merge()

predictors = [
    c for c in df_all.columns
    if c not in ["quality", "wine_type"]
]

# ============================================================
# 2. CÁLCULO MANUAL DE PEARSON
# ============================================================
def pearson_manual(x, y):
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)

    denom = np.sqrt(
        np.sum(x_c**2) *
        np.sum(y_c**2)
    )

    return (
        float(np.sum(x_c * y_c) / denom)
        if denom != 0
        else np.nan
    )


def compute_corr_matrix(data):
    corr = pd.DataFrame(
        index=predictors,
        columns=predictors,
        dtype=float
    )

    for fx in predictors:
        for fy in predictors:
            corr.loc[fx, fy] = pearson_manual(
                data[fx].to_numpy(),
                data[fy].to_numpy()
            )

    return corr


# Geração das matrizes segmentadas e global
corr_all = compute_corr_matrix(df_all)
corr_red = compute_corr_matrix(df_red)
corr_white = compute_corr_matrix(df_white)

# Salvar tabelas numéricas
corr_all.to_csv(
    TAB_DIR / "correlation_matrix_all.csv",
    float_format="%.4f"
)

corr_red.to_csv(
    TAB_DIR / "correlation_matrix_red.csv",
    float_format="%.4f"
)

corr_white.to_csv(
    TAB_DIR / "correlation_matrix_white.csv",
    float_format="%.4f"
)

# ============================================================
# 3. OTIMIZAÇÃO VISUAL: HEATMAP TRIANGULAR
# ============================================================
def plot_triangular_heatmap(corr_matrix, title, filename):

    # Máscara para esconder a diagonal superior
    # e evitar informação duplicada
    mask = np.triu(
        np.ones_like(corr_matrix, dtype=bool)
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 8}
    )

    plt.title(title, pad=15)

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=9
    )

    plt.yticks(fontsize=9)

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / filename,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close()


plot_triangular_heatmap(
    corr_all,
    "Correlação de Pearson (Base Combinada)",
    "heatmap_all.pdf"
)

plot_triangular_heatmap(
    corr_red,
    "Correlação de Pearson (Vinho Tinto)",
    "heatmap_red.pdf"
)

plot_triangular_heatmap(
    corr_white,
    "Correlação de Pearson (Vinho Branco)",
    "heatmap_white.pdf"
)

# ============================================================
# 4. EXTRAÇÃO DAS MAIORES CORRELAÇÕES
# ============================================================
upper_pairs = []

for i, fx in enumerate(predictors):
    for fy in predictors[i + 1:]:

        rho = corr_all.loc[fx, fy]

        upper_pairs.append({
            "Preditor_1": fx,
            "Preditor_2": fy,
            "rho": rho,
            "abs_rho": abs(rho)
        })


top_pairs = (
    pd.DataFrame(upper_pairs)
    .sort_values(
        "abs_rho",
        ascending=False
    )
)

top_pairs.head(10).to_csv(
    TAB_DIR / "top10_collinear_pairs.csv",
    index=False,
    float_format="%.4f"
)

# ============================================================
# 5. SCATTER PLOTS DOS 55 PARES DE PREDITORES
# ============================================================

# Existem C(11,2) = 55 pares distintos
pairs = list(combinations(predictors, 2))

# Quality usada apenas para identificação visual das classes
df_all["quality_cat"] = (
    df_all["quality"]
    .astype("category")
)

for i, (feature_x, feature_y) in enumerate(
    pairs,
    start=1
):

    plt.figure(figsize=(6, 4.5))

    sns.scatterplot(
        data=df_all,
        x=feature_x,
        y=feature_y,
        hue="quality_cat",
        palette="viridis",
        alpha=0.45,
        s=15,
        edgecolor=None
    )

    rho = corr_all.loc[
        feature_x,
        feature_y
    ]

    plt.title(
        f"{feature_x} × {feature_y}\n"
        f"Pearson $\\rho$ = {rho:.3f}"
    )

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)

    plt.legend(
        title="Quality",
        fontsize=7,
        title_fontsize=8,
        ncol=2
    )

    plt.grid(alpha=0.15)

    plt.tight_layout()

    name_x = feature_x.replace(" ", "_")
    name_y = feature_y.replace(" ", "_")

    plt.savefig(
        SCATTER_DIR /
        f"{i:02d}_{name_x}__vs__{name_y}.pdf",
        format="pdf",
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 6. DISPERSÃO FOCADA NAS MAIORES CORRELAÇÕES PARA O ARTIGO
# ============================================================

# Identificar as variáveis envolvidas nas top 3 maiores correlações
# para produzir uma figura compacta para o artigo
top_features = list(
    pd.unique(
        top_pairs
        .head(3)[
            ["Preditor_1", "Preditor_2"]
        ]
        .values
        .ravel()
    )
)

g = sns.pairplot(
    df_all,
    vars=top_features,
    hue="quality_cat",
    palette="viridis",
    plot_kws={
        "alpha": 0.5,
        "s": 15,
        "edgecolor": "none"
    },
    diag_kws={
        "common_norm": False
    },
    height=2.5
)

g.fig.suptitle(
    "Dispersão Bivariada dos Atributos de Maior Colinearidade",
    y=1.03
)

g.savefig(
    FIG_DIR / "pairplot_top_redundant.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FINALIZAÇÃO
# ============================================================

print("Execução finalizada.")
print(
    f"Scatter plots gerados: "
    f"{len(pairs)}"
)

print(
    f"Scatter plots salvos em: "
    f"{SCATTER_DIR}"
)

print(
    "Heatmaps, pairplot e tabelas "
    "também foram gerados em /outputs/"
)
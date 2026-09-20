from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Tarefa 5 — PCA FROM SCRATCH 
# ============================================================
DATA_DIR = Path("dataset")
OUT_DIR = Path("PCA/outputs")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

RED_FILE = DATA_DIR / "winequality-red.csv"
WHITE_FILE = DATA_DIR / "winequality-white.csv"
CLASS_COL = "quality"
PREDICTORS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# FUNÇÕES DE APOIO
# ------------------------------------------------------------
def save_latex_table(df, filename, caption, label, first_col_name="Preditor"):
    """Exporta DataFrame para formato LaTeX (IEEEtran compatível)."""
    cols = [str(c) for c in df.columns]
    col_spec = "l" + "r" * len(cols)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\columnwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        f"{first_col_name} & " + " & ".join(cols).replace("_", r"\_") + r" \\",
        r"\hline"
    ]
    
    for idx, row in df.iterrows():
        vals = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append(str(idx).replace("_", r"\_") + " & " + " & ".join(vals) + r" \\")
        
    lines += [r"\hline", r"\end{tabular}%", "}", r"\end{table}", ""]
    (TAB_DIR / filename).write_text("\n".join(lines), encoding="utf-8")

# ------------------------------------------------------------
# 1. LEITURA COM METADADOS
# ------------------------------------------------------------
red = pd.read_csv(RED_FILE, sep=";")
white = pd.read_csv(WHITE_FILE, sep=";")
red['wine_type'] = 'Tinto'
white['wine_type'] = 'Branco'
df = pd.concat([red, white], ignore_index=True)

X = df[PREDICTORS].to_numpy(dtype=float)
N, D = X.shape

# ------------------------------------------------------------
# 2. PADRONIZAÇÃO E COVARIÂNCIA (Correção N-1)
# ------------------------------------------------------------
mean = X.mean(axis=0)
std = X.std(axis=0, ddof=1)
Z = (X - mean) / std

# Estimador não enviesado da matriz de covariância
Sigma = (Z.T @ Z) / (N - 1)

# ------------------------------------------------------------
# 3. DECOMPOSIÇÃO ESPECTRAL E PROJEÇÃO
# ------------------------------------------------------------
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

# Ordenação decrescente
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Convenção de sinal (para reprodutibilidade)
for j in range(D):
    if eigenvectors[np.argmax(np.abs(eigenvectors[:, j])), j] < 0:
        eigenvectors[:, j] *= -1

explained_ratio = eigenvalues / eigenvalues.sum()
cumulative_ratio = np.cumsum(explained_ratio)
W = eigenvectors[:, :2]
X_pca = Z @ W

# Criar DataFrame com as componentes projetadas
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca['quality'] = df[CLASS_COL].astype('category')
df_pca['wine_type'] = df['wine_type'].astype('category')

# ------------------------------------------------------------
# 4. EXPORTAÇÃO NUMÉRICA E TABELAS (CSV e LaTeX)
# ------------------------------------------------------------
# Padronização e Covariância
pd.DataFrame({"feature": PREDICTORS, "mean": mean, "std_sample": std}).to_csv(TAB_DIR / "standardization_parameters.csv", index=False)
pd.DataFrame(Sigma, index=PREDICTORS, columns=PREDICTORS).to_csv(TAB_DIR / "covariance_matrix_standardized.csv")

# Autovalores e Variância Explicada
df_eigen = pd.DataFrame({
    "Componente": [f"PC{i+1}" for i in range(D)],
    "Autovalor": eigenvalues,
    "Variância (%)": 100 * explained_ratio,
    "Acumulada (%)": 100 * cumulative_ratio,
})
df_eigen.to_csv(TAB_DIR / "eigenvalues_explained_variance.csv", index=False)
save_latex_table(df_eigen.set_index("Componente"), "pca_eigenvalues.tex", "Autovalores e variância explicada pelas componentes principais.", "tab:pca_eigenvalues", "Componente")

# Carregamentos (Loadings)
df_loadings = pd.DataFrame(W, index=PREDICTORS, columns=["PC1", "PC2"])
df_loadings.to_csv(TAB_DIR / "pca_loadings_PC1_PC2.csv")
save_latex_table(df_loadings, "pca_loadings.tex", "Carregamentos das duas primeiras componentes principais.", "tab:pca_loadings")

# Projeções e Centroides
df_pca.to_csv(TAB_DIR / "pca_projection.csv", index=False)

rows = []
for cls in sorted(df[CLASS_COL].unique()):
    mask = df[CLASS_COL].to_numpy() == cls
    scores = X_pca[mask]
    rows.append({
        "quality": int(cls), "N": int(mask.sum()),
        "PC1_mean": scores[:, 0].mean(), "PC2_mean": scores[:, 1].mean(),
        "PC1_std": scores[:, 0].std(ddof=1), "PC2_std": scores[:, 1].std(ddof=1),
    })
pd.DataFrame(rows).to_csv(TAB_DIR / "class_centroids_PC1_PC2.csv", index=False)

# ------------------------------------------------------------
# 5. PLOTS OTIMIZADOS
# ------------------------------------------------------------
pc1_var = explained_ratio[0] * 100
pc2_var = explained_ratio[1] * 100

# Gráfico 1: Exigência do Professor (Colorido por Qualidade)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="quality", palette="viridis", s=15, alpha=0.6, edgecolor=None)
plt.xlabel(f"PC1 ({pc1_var:.2f}% da variância)")
plt.ylabel(f"PC2 ({pc2_var:.2f}% da variância)")
plt.title("Projeção PCA - Agrupamento por Qualidade")
plt.legend(title="Qualidade", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.2)
plt.savefig(FIG_DIR / "pca_by_quality.pdf", bbox_inches="tight")
plt.close()

# Gráfico 2: Descoberta Académica (Colorido por Tipo de Vinho)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="wine_type", palette="Set1", s=15, alpha=0.6, edgecolor=None)
plt.xlabel(f"PC1 ({pc1_var:.2f}% da variância)")
plt.ylabel(f"PC2 ({pc2_var:.2f}% da variância)")
plt.title("Projeção PCA - Agrupamento por Tipo de Vinho")
plt.legend(title="Tipo", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.2)
plt.savefig(FIG_DIR / "pca_by_winetype.pdf", bbox_inches="tight")
plt.close()

# Scree Plot
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, D + 1), explained_ratio * 100, marker="o", color='#1f77b4')
plt.xlabel("Componente Principal")
plt.ylabel("Variância Explicada (%)")
plt.title("Variância Explicada por Componente")
plt.xticks(np.arange(1, D + 1))
plt.grid(alpha=0.2)
plt.savefig(FIG_DIR / "scree_plot.pdf", bbox_inches="tight")
plt.close()

# Gráficos de Loadings (PC1 e PC2)
for i, pc in enumerate(["PC1", "PC2"]):
    loadings = df_loadings[pc].sort_values()
    plt.figure(figsize=(7, 4))
    sns.barplot(x=loadings.values, y=loadings.index, color="#1f77b4")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel("Carregamento")
    plt.ylabel("Preditor")
    plt.title(f"Carregamentos {pc}")
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"loadings_{pc}.pdf", bbox_inches="tight")
    plt.close()


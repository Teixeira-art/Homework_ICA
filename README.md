# Homework 1 — ICA

Trabalho desenvolvido para a disciplina de Inteligência Computacional Aplicada, utilizando o Wine Quality Dataset.

O objetivo do projeto é realizar uma análise exploratória dos dados, investigar a separabilidade entre as classes de qualidade, estudar relações lineares entre os atributos físico-químicos e aplicar Análise de Componentes Principais (PCA) para redução de dimensionalidade.

## Dataset

Foram utilizados conjuntamente os subconjuntos:

- winequality-red.csv
- winequality-white.csv

Após a concatenação, a base possui:

- N = 6497 observações
- D = 11 preditores físico-químicos
- L = 7 classes observadas de qualidade: 3, 4, 5, 6, 7, 8 e 9

A variável quality é utilizada como rótulo de classe.

O tipo de vinho (red ou white) é preservado apenas como metadado auxiliar em algumas análises complementares e não é considerado um dos 11 preditores.

Os 11 preditores utilizados são:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

## Estrutura do projeto

A organização principal do repositório é dividida de acordo com as etapas do trabalho:

```text
Homework_ICA/
│
├── dataset/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality.names
│
├── mono-variate analysis/
│   ├── mono_variate_script.py
│   └── outputs/
│       ├── resultados_membros_1_2.tex
│       ├── figures/
│       │   ├── unconditional/
│       │   └── conditional/
│       └── tables/
│
├── bivariado/
│   ├── bivariado.py
│   └── outputs/
│       ├── figures/
│       │   ├── scatter_pairs/
│       │   │    ├── 01_fixed_acidity__vs__volatile_acidity.pdf
│       │   │    ├── 02_fixed_acidity__vs__citric_acid.pdf
│       │   │    ├── 03_fixed_acidity__vs__residual_sugar.pdf
│       │   │    ├── (...)
│       │   │    └── 55_sulphates_vs_alcohol.pdf
│       │   ├── heatmap_all.pdf
│       │   ├── heatmap_red.pdf
│       │   ├── heatmap_white.pdf
│       │   └── pairplot_top_redundant.pdf
│       └── tables/
│
├── PCA/
│   ├── pca_script.py
│   └── outputs/
│       ├── resultados_membro_4.tex
│       ├── figures/
│       └── tables/
│
├── requirements.txt
└── README.md
```

## Tarefas 1, 2 e 3 — Análise monovariada

A primeira parte do trabalho realiza a descrição do conjunto de dados e as análises monovariadas incondicional e condicionada à classe.

Para cada um dos 11 preditores foram calculados:

- média amostral
- desvio padrão amostral
- coeficiente de assimetria
- histogramas
- box-plots

Na análise condicionada, essas medidas são calculadas separadamente para cada uma das sete classes de qualidade.

Como existem 11 preditores e 7 classes, são analisadas 77 combinações preditor-classe.

As tabelas condicionais completas estão disponíveis em:

```text
mono-variate analysis/outputs/tables/
```

Entre os arquivos gerados estão:

```text
conditional_means.tex
conditional_stds.tex
conditional_skew.tex
```

As figuras completas das análises incondicionais e condicionais também são mantidas no repositório, mesmo quando apenas uma seleção é apresentada no artigo devido ao limite de páginas.

## Tarefa 4 — Análise bivariada

A análise bivariada investiga a dependência linear entre os preditores por meio do coeficiente de correlação de Pearson.

Com 11 preditores, existem 55 pares distintos de variáveis.

Foram produzidos:

- matriz de correlação da base combinada
- matrizes complementares para vinhos tintos e brancos
- heatmaps das matrizes de correlação
- 55 scatter plots, um para cada par distinto de preditores
- pairplot dos atributos envolvidos nas maiores correlações em módulo
- tabela com os pares de maior correlação

Os 55 scatter plots estão disponíveis em:

```text
bivariado analysis/outputs/figures/scatter_pairs/
```

Os principais resultados tabulares encontram-se em:

```text
bivariado analysis/outputs/tables/
```

## Tarefa 5 — Análise de Componentes Principais

A PCA foi implementada diretamente por operações matriciais, sem utilizar uma implementação pronta de PCA.

O procedimento inclui:

- padronização dos 11 preditores
- cálculo da matriz de covariância
- cálculo dos autovalores e autovetores
- ordenação das componentes principais
- seleção das duas primeiras componentes
- projeção dos dados no espaço PC1 × PC2
- cálculo da variância explicada
- análise dos loadings
- visualização das observações por classe de qualidade
- análise complementar utilizando o tipo de vinho como metadado

Entre os resultados produzidos estão:

- projeção PCA por qualidade
- projeção PCA por tipo de vinho
- scree plot
- loadings de PC1 e PC2
- autovalores e variância explicada
- matriz de covariância padronizada
- projeções das observações no espaço PCA

## Instalação

É necessário possuir Python 3 instalado.

Na raiz do projeto, instale as dependências com:

```bash
pip install -r requirements.txt
```

As principais bibliotecas utilizadas nos scripts são:

- NumPy
- pandas
- Matplotlib
- Seaborn

## Execução

Os scripts geram automaticamente suas respectivas tabelas e figuras nas pastas outputs/.

Como alguns scripts utilizam caminhos relativos, recomenda-se executar cada análise a partir de seu respectivo diretório.

### Análise monovariada

Entre na pasta:

```bash
cd "mono-variate analysis"
```

e execute o script Python responsável pelas Tarefas 1, 2 e 3 presente nesse diretório.

### Análise bivariada

Entre na pasta:

```bash
cd "bi-variate analysis"
```

e execute:

```bash
python script_geral_4.py
```

### PCA

Entre na pasta:

```bash
cd PCA
```

e execute o script Python responsável pela Tarefa 5 presente nesse diretório.

Após a execução, os resultados são armazenados automaticamente nas respectivas pastas outputs/.

## Reprodutibilidade

Os códigos utilizados no trabalho geram automaticamente as estatísticas, tabelas e figuras utilizadas na análise.

Os resultados completos são mantidos no repositório. O artigo apresenta apenas uma seleção dos resultados devido ao limite de páginas, enquanto as análises completas permanecem disponíveis nas pastas de saída de cada etapa.

## Contribuições

- Lucas Martins e Lucas Teixeira — Tarefas 1, 2 e 3: descrição do dataset e análises monovariadas incondicional e condicionada às classes
- João Victor Falcão — Tarefa 4: análise bivariada, cálculo da correlação de Pearson, heatmaps e scatter plots
- João Victor de Abreu — Tarefa 5: implementação do PCA, cálculo das componentes principais e análise dos resultados

Todos os integrantes participaram da interpretação dos resultados, organização do trabalho e revisão do artigo.

## Uso de Inteligência Artificial

Ferramentas de inteligência artificial generativa foram utilizadas como apoio à organização do relatório, revisão textual e depuração de código.

O uso dessas ferramentas foi realizado como apoio ao desenvolvimento, permanecendo sob responsabilidade dos integrantes a verificação dos cálculos, da metodologia, dos resultados e do conteúdo apresentado no artigo.

Os prompts e respostas relevantes utilizados durante o desenvolvimento devem ser disponibilizados no repositório conforme as orientações da disciplina.

## Artigo

O relatório final foi desenvolvido no formato de conferência IEEE.

O artigo apresenta de forma resumida os principais resultados das análises monovariada, bivariada e multivariada, enquanto os resultados completos permanecem disponíveis neste repositório.

## Referência do dataset

P. Cortez, A. Cerdeira, F. Almeida, T. Matos e J. Reis, “Modeling wine preferences by data mining from physicochemical properties”, Decision Support Systems, vol. 47, no. 4, pp. 547–553, 2009.

# HW2 — Modelos de regressão para estimativa da solubilidade de compostos químicos

Projeto desenvolvido para o **Homework II** da disciplina **TI0175 — Inteligência Computacional Aplicada**, do Departamento de Engenharia de Teleinformática (DETI), Universidade Federal do Ceará (UFC).

O objetivo é avaliar diferentes modelos de regressão para estimar a solubilidade de compostos químicos a partir de descritores moleculares. O projeto está organizado em quatro partes, correspondentes às etapas do trabalho: análise exploratória e preparação dos dados; modelos lineares; PLS e rede neural; e análise estatística e integração dos resultados.

## 1. Dados utilizados

Foi utilizada a **Alternativa 1** do enunciado, com a base de solubilidade disponibilizada no pacote `AppliedPredictiveModeling` do R.

- **Observações:** 1.267 compostos químicos.
- **Treinamento:** 951 observações.
- **Teste:** 316 observações.
- **Preditores:** 228 descritores moleculares — 208 fingerprints binários, 16 variáveis de contagem e 4 variáveis contínuas.
- **Variável resposta:** solubilidade do composto, representada em escala logarítmica.

A divisão original entre treinamento e teste é preservada. O conjunto de teste é reservado para a avaliação final dos modelos, enquanto a seleção de hiperparâmetros é realizada com validação cruzada somente no conjunto de treinamento.

## 2. Organização e contribuição de cada integrante

O trabalho foi dividido em quatro partes. Cada integrante ficou responsável por uma etapa específica do fluxo de análise.

### Parte 1 - Análise exploratória, métricas e pré-processamento

**Responsável:** João Victor Marques Falcao - 567357.

Esta etapa prepara a base que será utilizada pelos demais modelos. Inclui:

- leitura e organização dos dados de treinamento e teste;
- identificação dos 208 fingerprints e das 20 variáveis não binárias;
- análise das distribuições e das assimetrias dos preditores;
- transformação Box-Cox das variáveis não binárias, com deslocamento para valores positivos;
- padronização dos 228 preditores;
- cálculo das métricas RMSE e R²;
- análise das correlações entre preditores e entre cada preditor e a variável resposta;
- geração de tabelas e figuras da análise exploratória.

O pré-processamento é ajustado apenas nos dados de treinamento disponíveis em cada etapa. Durante a validação cruzada, isso significa ajustar as transformações novamente dentro do treino de cada fold, evitando vazamento de informação.

**Principais módulos:** `data.py`, `metrics.py`, `preprocessing.py`, `plot_utils.py` e `eda.py`.

### Parte 2 - OLS, validação cruzada e Ridge

**Responsável:** Lucas Teixeira Holanda - 568254.

Esta etapa desenvolve e avalia os modelos lineares exigidos nos itens 1 e 2 do enunciado:

- implementação da validação cruzada de 10 folds e geração da partição compartilhada pelos modelos;
- implementação da regressão linear ordinária (OLS);
- implementação da regressão Ridge, com regularização L2;
- seleção do parâmetro de regularização λ do Ridge por validação cruzada, utilizando RMSE;
- comparação das implementações com as funções equivalentes do scikit-learn;
- avaliação dos modelos no conjunto de teste;
- geração dos arquivos de resultados e do perfil de validação cruzada do Ridge, em RMSE e R².

Os mesmos folds são utilizados nas etapas seguintes, permitindo uma comparação consistente entre os modelos.

**Principais módulos:** `cv.py`, `ols.py`, `ridge.py` e `run_lineares.py`.

### Parte 3 - PLS e rede neural

**Responsável:** João Victor de Abreu Silva - 568274.

Esta etapa desenvolve e avalia os modelos dos itens 3 e 4 do enunciado:

- treinamento da regressão PLS (*Partial Least Squares*), que constrói componentes latentes considerando a relação entre os preditores e a resposta;
- seleção do número de componentes do PLS por validação cruzada, utilizando exclusivamente os dados de treinamento;
- registro do perfil de validação cruzada do PLS em RMSE e R² e geração da Figura 3(b);
- construção de uma rede neural MLP (*Multi-Layer Perceptron*) para regressão, com uma camada oculta;
- busca de hiperparâmetros da MLP por validação cruzada, usando os mesmos folds das etapas anteriores;
- repetição dos treinamentos com diferentes sementes aleatórias para avaliar a variabilidade das redes;
- ajuste final com múltiplas sementes e geração das previsões individuais e da previsão média (*ensemble*) no conjunto de teste;
- preparação dos trechos LaTeX de Métodos e Resultados referentes ao PLS e à MLP.

A média das previsões das redes treinadas com diferentes sementes é utilizada como ensemble. A comparação descritiva do desempenho da MLP com os modelos lineares é apresentada nesta etapa; a avaliação da significância estatística das diferenças fica para a Parte 4.

**Principais módulos:** `pls.py`, `mlp.py` e `run_pls_mlp.py`.

### Parte 4 - Testes estatísticos e integração final

**Responsável:** Lucas Martins Menezes - 565788.

Esta etapa consolida os resultados das quatro famílias de modelos (OLS, Ridge, PLS e MLP) e realiza a análise comparativa final:

- conferência da consistência das previsões e das métricas produzidas pelas etapas anteriores;
- comparação dos modelos por RMSE e R²;
- aplicação de bootstrap pareado no conjunto de teste para estimar intervalos de confiança das diferenças de RMSE;
- aplicação do teste de Wilcoxon pareado aos resultados de validação cruzada dos modelos lineares, com correção de Holm;
- geração da tabela consolidada de comparação e da figura de valores preditos versus observados;
- integração dos trechos LaTeX, referências, resumo, introdução e conclusão;
- revisão da documentação, reprodutibilidade e organização final do repositório.

A conclusão sobre superioridade estatística de um modelo deve seguir os intervalos de confiança e os testes realizados nesta etapa. Uma diferença numérica entre métricas, isoladamente, não é suficiente para afirmar que um modelo é estatisticamente superior.


## 3. Fluxo metodológico do projeto

A sequência geral de execução e análise é:

1. **Preparação dos dados:** carregar os arquivos fornecidos, examinar as distribuições, transformar as variáveis assimétricas e padronizar os preditores.
2. **Modelos lineares:** ajustar OLS e Ridge, selecionar λ por validação cruzada e avaliar os resultados.
3. **Modelos de redução de dimensão e não lineares:** selecionar o número de componentes do PLS e ajustar a MLP com busca de hiperparâmetros e múltiplas sementes.
4. **Comparação final:** reunir as previsões dos quatro modelos, comparar as métricas e avaliar a incerteza das diferenças por métodos estatísticos.

A validação cruzada utiliza **10 folds compartilhados**, definidos com semente fixa. Em cada fold, o pré-processamento é ajustado apenas na parcela de treinamento daquele fold. Depois da seleção dos hiperparâmetros, cada modelo é reajustado utilizando todo o conjunto de treinamento e avaliado no conjunto de teste reservado.

As métricas principais são:

- **RMSE (Root Mean Squared Error):** mede a magnitude média dos erros de previsão, penalizando mais fortemente erros maiores. Valores menores indicam menor erro.
- **R² (coeficiente de determinação):** calculado como `1 - SSE/SST`, mede a proporção da variação da resposta explicada pelo modelo. Valores maiores indicam melhor ajuste. Neste projeto, R² não é calculado como o quadrado da correlação entre valores observados e previstos.

## 4. Uso de ferramentas de inteligência artificial

Ferramentas de inteligência artificial foram utilizadas como apoio ao desenvolvimento (fonte de pesquisa), à revisão de código, à análise dos métodos e à documentaçã.


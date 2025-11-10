# Análise de Clusterização de Países

Este projeto realiza uma análise de clusterização para agrupar países com características socioeconômicas semelhantes. São utilizados os algoritmos K-Means e Clusterização Hierárquica.

## Pré-requisitos

- Python 3.9+
- `pip` e `venv` (geralmente inclusos na instalação do Python)

## Instalação

1. **Clone o repositório ou baixe os arquivos do projeto.**

2. **Crie e ative um ambiente virtual:**

   ```bash
   # No Windows
   python -m venv .venv
   .venv\Scripts\activate

   # No macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. **Baixe o conjunto de dados:**

   O arquivo `Country-data.csv` é necessário. Faça o download a partir deste link:
   [https://www.kaggle.com/datasets/rohan0y20/country-data](https://www.kaggle.com/datasets/rohan0y20/country-data)

2. **Posicione o arquivo:**

   Coloque o arquivo `Country-data.csv` na raiz do diretório do projeto (na mesma pasta que o arquivo `requirements.txt`).

3. **Execute o script:**

   O script principal está localizado em `trabls/trabalho.py`. Para executá-lo, navegue até a raiz do projeto e use o seguinte comando:

   ```bash
   python trabls/trabalho.py
   ```

## Resultados

Ao executar o script, você verá no terminal:

- A análise e interpretação dos clusters.
- A comparação entre os resultados do K-Means e da Clusterização Hierárquica.

Além disso, três arquivos de imagem serão gerados e salvos na raiz do projeto:

- `boxplot_distribuicao_antes.png`: Mostra a distribuição dos dados antes do pré-processamento.
- `boxplot_distribuicao_depois.png`: Mostra a distribuição dos dados após a normalização.
- `dendograma_hierarquico.png`: O dendograma gerado pelo algoritmo de clusterização hierárquica.



## Referências


"""
---------------------------------------------------------------------
--- Parte 4: Escolha de algoritmos ---
---------------------------------------------------------------------

1 - Escreva em tópicos as etapas do algoritmo de K-médias até sua convergência.

O algoritmo K-Médias (K-Means) funciona da seguinte forma:

1.  **Escolha de K:** O usuário define o número de clusters (K) desejado.
2.  **Inicialização dos Centróides:** O algoritmo escolhe aleatoriamente K pontos dos dados (ou K pontos no espaço dos dados) para serem os centróides iniciais.
3.  **Atribuição (Passo E - Expectation):** Para cada ponto de dado:
    * Calcula-se a distância (geralmente Euclidiana) deste ponto a *cada um* dos K centróides.
    * O ponto é atribuído ao cluster do centróide mais próximo.
4.  **Atualização (Passo M - Maximization):** Após todos os pontos serem atribuídos:
    * Para cada um dos K clusters, calcula-se um novo centróide.
    * O novo centróide é o **baricentro** (a média aritmética) de *todos* os pontos que foram atribuídos àquele cluster na etapa anterior.
5.  **Convergência:** O algoritmo repete os passos 3 (Atribuição) e 4 (Atualização) iterativamente. A convergência ocorre quando:
    * A atribuição dos pontos aos clusters não muda mais entre iterações;
    * OU os centróides se movem muito pouco (abaixo de um limiar de tolerância);
    * OU um número máximo de iterações é atingido.


2 - Refaça o algoritmo... [garantir que o cluster seja representado pelo dado mais próximo ao seu baricentro]... esse novo algoritmo, o dado escolhido será chamado medóide.

Este algoritmo é conhecido como **K-Medóides** (ou, mais especificamente, uma variação dele, como o PAM - Partitioning Around Medoids, embora a descrição seja uma mistura de K-Means e K-Medoids). As etapas seriam:

1.  **Escolha de K:** O usuário define o número de clusters (K).
2.  **Inicialização dos Medóides:** O algoritmo escolhe K pontos *reais* do conjunto de dados para serem os medóides iniciais.
3.  **Atribuição:** Para cada ponto de dado (que não seja um medóide):
    * Calcula-se a distância deste ponto a *cada um* dos K medóides.
    * O ponto é atribuído ao cluster do medóide mais próximo.
4.  **Atualização (Etapa de Troca ou Otimização, conforme a descrição):**
    * **a. Cálculo do Baricentro:** Para cada um dos K clusters, calcula-se o **baricentro** (a média aritmética, ou centróide) de todos os pontos atribuídos a ele (assim como no K-Médias).
    * **b. Seleção do Novo Medóide:** O algoritmo *não* usa esse baricentro (que pode não ser um dado real) como novo representante. Em vez disso, ele procura dentro do cluster (entre os pontos *reais* atribuídos a ele) qual é o ponto que está **mais próximo** daquele baricentro calculado.
    * **c. Atualização:** Esse ponto real (o dado mais próximo do baricentro) torna-se o **novo medóide** daquele cluster para a próxima iteração.
5.  **Convergência:** Repete os passos 3 e 4 até que os medóides não mudem mais entre as iterações (ou atinja a convergência).

*(Obs: O algoritmo K-Medóides clássico, PAM, usa uma etapa de "troca" (swap) testando todos os não-medóides para ver se a troca reduz o custo total, mas a descrição da pergunta se encaixa no processo 4a-4c).*


3 - O algoritmo de K-médias é sensível a outliers nos dados. Explique.

O K-Médias é sensível a outliers porque seu objetivo é minimizar a **Inércia**, que é a *soma das distâncias quadradas* de cada ponto ao centróide do seu cluster.

1.  **Distância Quadrada:** Um outlier é um ponto muito distante da maioria dos outros pontos. A distância desse ponto a qualquer centróide será grande. Ao elevar essa distância ao quadrado (d²), o outlier contribui com um valor desproporcionalmente enorme para a soma total (Inércia).
2.  **Atualização do Centróide:** Na etapa de atualização (Passo M), o centróide é calculado como a *média* de todos os pontos do cluster. Como em qualquer cálculo de média, um valor extremo (o outlier) "puxa" a média significativamente em sua direção.
3.  **Resultado:** Para minimizar o enorme erro quadrático causado pelo outlier, o algoritmo deslocará o centróide do cluster para mais perto desse outlier, distorcendo o centróide e fazendo com que ele deixe de representar o verdadeiro "centro" dos dados não-atípicos (o "corpo" principal do cluster).


4 - Por que o algoritmo de DBScan é mais robusto à presença de outliers?

O DBSCAN (Density-Based Spatial Clustering of Applications with Noise) é robusto a outliers porque ele define clusters com base na **densidade** de pontos, e não em centróides ou médias.

1.  **Definição de Cluster:** O DBSCAN define um cluster como uma área de alta densidade de pontos, onde "alta densidade" é definida por dois parâmetros:
    * **Eps (ε):** Um raio de vizinhança.
    * **MinPts:** O número mínimo de pontos que devem existir dentro desse raio (Eps) para que um ponto seja considerado um "ponto central" (core point).
2.  **Como ele Trata Outliers:** Um outlier, por definição, é um ponto isolado. Ele está em uma região de baixa densidade.
    * Quando o DBSCAN analisa um outlier, ele verifica sua vizinhança (raio Eps).
    * Ele não encontrará MinPts vizinhos.
    * Portanto, o outlier não será classificado como um "ponto central" (core point) e nem será "alcançável por densidade" (density-reachable) a partir de nenhum cluster.
3.  **Classificação de Ruído (Noise):** O DBSCAN possui uma categoria específica para esses pontos: **Ruído (Noise)**. Os outliers são explicitamente classificados como ruído e *não são atribuídos a nenhum cluster*.

Dessa forma, os outliers não têm influência no cálculo ou na forma dos clusters. Os clusters são formados apenas pelas regiões densas, e os outliers são simplesmente ignorados e rotulados separadamente.
"""
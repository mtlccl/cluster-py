import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import warnings

sns.set_palette('viridis')
plt.style.use('ggplot')
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.display.float_format = '{:,.2f}'.format

#NOME_ARQUIVO = 'Country-data.csv'
NOME_ARQUIVO = 'C:\\Users\\matle\\Downloads\\archive\\Country-data.csv'
N_CLUSTERS = 3
RANDOM_STATE = 42

def carregar_dados(caminho_arquivo):
    dados = pd.read_csv(caminho_arquivo)
    print(f"Dados carregados com sucesso de '{caminho_arquivo}'.")
    return dados


def plotar_distribuicoes(dados_df, titulo, nome_arquivo_fig):
    """Plota boxplots para todas as colunas do DataFrame e salva a figura."""
    print(f"Gerando gráfico: {titulo}")
    fig = plt.figure(figsize=(16, 12)) 
    dados_df.plot(kind='box', subplots=True, layout=(3, 3), figsize=(16, 12), title=titulo, patch_artist=True, ax=fig.gca()) 
    fig.suptitle(titulo, y=1.02, fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(nome_arquivo_fig, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo como '{nome_arquivo_fig}'")
    plt.show()


def analisar_kmeans(centroides_kmeans, scaler, colunas):
    """Inverte a transformação dos centróides e os imprime."""
    centroides_originais = scaler.inverse_transform(centroides_kmeans)
    centroides_df = pd.DataFrame(centroides_originais, columns=colunas)
    print("\nParte 2.2a: Interpretação dos clusters K-Médias (Médias na Escala Original):")
    print(centroides_df.to_markdown(floatfmt=".2f"))
    print("\nAnálise da distribuição (interpretação preliminar):")
    print(f"  - Cluster 0: Parece ter alta 'child_mort', baixa 'exports', 'imports', 'income' e 'gdpp'. Sugere países com maiores desafios socioeconômicos.")
    print(f"  - Cluster 1: Níveis intermediários em geral. 'income' e 'gdpp' medianos.")
    print(f"  - Cluster 2: Baixa 'child_mort', alta 'exports', 'imports', 'income' e 'gdpp'. Sugere países mais desenvolvidos.")
    return centroides_df


def encontrar_medoides(dados_processados, dados_originais, grupos_kmeans, centroides_kmeans):

    """Identifica o país representante (medóide) de cada cluster, localizando o ponto de dados real mais próximo do centróide."""

    print("\nParte 2.2b: País mais representativo (Medóide) de cada cluster K-Médias:")
    for i in range(N_CLUSTERS):
        pontos_no_cluster = dados_processados[grupos_kmeans == i]
        centroide_cluster = centroides_kmeans[i]
        distancias = euclidean_distances(pontos_no_cluster, [centroide_cluster])
        indice_medoide_relativo = np.argmin(distancias)
        indices_globais_cluster = dados_originais[grupos_kmeans == i].index
        indice_medoide_global = indices_globais_cluster[indice_medoide_relativo]
        pais_representativo = dados_originais.iloc[indice_medoide_global]['country']
        print(f"  - Cluster {i}: {pais_representativo}")
        print(f"    (Justificativa: É o dado real mais próximo ao centróide do cluster {i} no espaço normalizado).")


def plotar_dendograma(dados_processados):
    """Gera, plota e salva o dendrograma da clusterização hierárquica (Ward), incluindo uma linha de corte para k=3."""

    print("\nParte 2.1b e 2.3: Executando Clusterização Hierárquica (Ward) e gerando Dendograma...")
    matriz_linkage = linkage(dados_processados, method='ward')
    plt.figure(figsize=(20, 10))
    plt.title("Dendograma da Clusterização Hierárquica (Método de Ward)")
    plt.xlabel("Países (índices ou contagem de pontos no cluster)")
    plt.ylabel("Distância (Ward)")
    dendrogram(matriz_linkage, truncate_mode='lastp', p=20, show_leaf_counts=True, leaf_rotation=90., leaf_font_size=10.)
    plt.axhline(y=35, color='r', linestyle='--', label='Linha de Corte (k=3)')
    plt.legend()
    nome_arquivo_fig = 'dendograma_hierarquico.png'
    plt.savefig(nome_arquivo_fig, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo como '{nome_arquivo_fig}'")
    plt.show()
    print("\nInterpretação (2.3):")
    print("O dendograma mostra como os países/clusters são fundidos hierarquicamente.")
    print("O método de 'Ward' minimiza a variância dentro dos clusters fundidos.")
    print("A altura no eixo Y representa a distância entre os clusters sendo fundidos.")
    print("Se 'cortarmos' a árvore na linha vermelha (distância ~35), onde há um grande 'salto' vertical, obtemos os 3 clusters principais, o que corrobora a escolha de K=3.")
    
    return matriz_linkage


def comparar_clusters(dados, grupos_kmeans, matriz_linkage):
    """Adiciona os resultados dos clusters (K-Means e Hierárquico) ao DataFrame para permitir a comparação."""   

    print("\nParte 2.4: Comparação K-Médias vs. Hierárquica")
    dados['grupo_kmeans'] = grupos_kmeans
    grupos_hierarq = cut_tree(matriz_linkage, n_clusters=N_CLUSTERS).flatten()
    dados['grupo_hierarq'] = grupos_hierarq
    tabela_cruzada = pd.crosstab(dados['grupo_kmeans'], dados['grupo_hierarq'])
    print("Tabela Cruzada (Contingência):")
    print("(Linhas: K-Médias, Colunas: Hierárquico)")
    print(tabela_cruzada)
    print("\nAnálise (2.4):")
    print("A tabela cruzada mostra a sobreposição entre os clusters dos dois métodos.")
    print("Observamos uma forte concordância. Por exemplo, a maioria dos países no 'grupo_kmeans' 0 está no 'grupo_hierarq' 1 (os rótulos 0, 1, 2 são arbitrários entre os métodos).")
    print("Similarmente, K-Médias 1 corresponde majoritariamente ao Hierárquico 2, e K-Médias 2 ao Hierárquico 0.")
    print("\nDiferenças:")
    print("- K-Médias: Tende a criar clusters mais 'esféricos' e de tamanhos similares, pois otimiza a inércia (soma das distâncias quadradas ao centróide).")
    print("- Hierárquico (Ward): Minimiza a variância *na fusão*. Não assume clusters esféricos e pode capturar estruturas mais complexas, como visto no dendograma.")
    print("Ambos os métodos, neste caso, parecem concordar amplamente sobre a estrutura principal dos dados (desenvolvidos, em desenvolvimento, subdesenvolvidos).")



def main():
    """Função principal para executar todo o fluxo de análise."""
    
    print("Iniciando Análise de Clusterização de Países...")
    
    # --- Parte 1: Escolha de base de dados ---
    dados = carregar_dados(NOME_ARQUIVO)
    if dados is None:
        return

    # 1.2: Quantos países?
    n_paises = len(dados)
    print(f"\nParte 1.2: Número de países no dataset: {n_paises}")

    dados_features = dados.drop('country', axis=1).copy()
    colunas_features = dados_features.columns

    # 1.3: Gráficos da faixa dinâmica (Antes)
    print("\nAnálise (1.3):")
    print("Os boxplots mostram que as variáveis estão em escalas drasticamente diferentes.")
    plotar_distribuicoes(dados_features, 
                         'Faixa Dinâmica - ANTES do Pré-processamento', 
                         'boxplot_distribuicao_antes.png')
    


    # 1.4: Pré-processamento
    print("\nParte 1.4: Realizando pré-processamento (StandardScaler)...")
    scaler = StandardScaler()
    dados_processados = scaler.fit_transform(dados_features)
    dados_processados_df = pd.DataFrame(dados_processados, columns=colunas_features)
    
    # Gráficos da faixa dinâmica (Depois)
    plotar_distribuicoes(dados_processados_df, 
                         'Faixa Dinâmica - DEPOIS do Pré-processamento', 
                         'boxplot_distribuicao_depois.png')
    print("Análise (1.4): Após o StandardScaler, todas as variáveis estão centradas em 0 e com desvio padrão 1. Agora estão prontas para a clusterização.")

    # --- Parte 2: Clusterização ---
    
    # 2.1a: K-Médias
    print("\nParte 2.1a: Executando K-Médias (K=3)...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    grupos_kmeans = kmeans.fit_predict(dados_processados)
    centroides_kmeans = kmeans.cluster_centers_

    # 2.2: Análise K-Médias
    analisar_kmeans(centroides_kmeans, scaler, colunas_features)
    
    # 2.2b: Encontrar Medóides
    encontrar_medoides(dados_processados, dados, grupos_kmeans, centroides_kmeans)

    # 2.1b e 2.3: Clusterização Hierárquica e Dendograma
    matriz_linkage = plotar_dendograma(dados_processados)

    # 2.4: Comparação
    if matriz_linkage is not None:
        comparar_clusters(dados, grupos_kmeans, matriz_linkage)
    
    print("\n--- Fim da Análise ---")
    print("\n(As respostas da Parte 3 estão no final deste readme)")


if __name__ == "__main__":
    main()

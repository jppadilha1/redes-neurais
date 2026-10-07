## Questão 1 – Configuração dos treinamentos

Foram executados 5 treinamentos de uma rede Perceptron Multicamadas com 3 entradas (x1, x2, x3), 10 neurônios na camada escondida e 1 neurônio na camada de saída, usando o algoritmo backpropagation (Regra Delta Generalizada) com os 200 padrões do conjunto de treinamento.

- Função de ativação: logística, em todos os neurônios.
- Taxa de aprendizagem: η = 0,1.
- Precisão: ε = 10⁻⁶. O treinamento termina quando |EQM_atual − EQM_anterior| ≤ ε.
- Pesos iniciais: valores aleatórios entre 0 e 1 nas matrizes W1 (10 × 4, incluindo o bias) e W2 (1 × 11, incluindo o bias), com o gerador de números aleatórios reiniciado com uma semente diferente em cada treinamento (sementes 1 a 5).

## Questão 2 – Resultados dos treinamentos

| Treinamento | Semente | Erro Quadrático Médio | Número de Épocas |
|:-----------:|:-------:|:---------------------:|:----------------:|
| 1º (T1) | 1 | 0.000782 | 106 |
| 2º (T2) | 2 | 0.000779 | 115 |
| 3º (T3) | 3 | 0.000774 | 123 |
| 4º (T4) | 4 | 0.000769 | 108 |
| 5º (T5) | 5 | 0.000789 | 105 |

## Questão 3 – Gráficos do EQM por época

Os dois treinamentos com maior número de épocas foram o T3 (123 épocas) e o T2 (115 épocas).

![Gráficos do EQM em função das épocas para os treinamentos T3 e T2](graficos_eqm_mlp.png)

Nos dois treinamentos, o EQM parte de valores próximos de 0,01, cai rapidamente nas primeiras épocas e depois diminui cada vez mais devagar, até estabilizar em torno de 0,00077. O treinamento é encerrado quando a variação do EQM entre duas épocas consecutivas fica menor ou igual a ε = 10⁻⁶.

## Questão 4 – Por que o EQM e o número de épocas variam?

O EQM final e o número de épocas variam porque cada treinamento começa com matrizes de pesos iniciais diferentes, sorteadas aleatoriamente. Isso faz com que cada um parta de um ponto diferente da superfície de erro (o EQM em função de todos os pesos da rede).

Diferentemente do ADALINE, cuja superfície de erro tem um único mínimo, a superfície de erro do Perceptron Multicamadas é não convexa, por causa das funções de ativação não lineares e da camada escondida. Ela possui vários mínimos locais, regiões planas e trechos com inclinações diferentes. Assim:

- **O EQM final varia** porque, dependendo do ponto de partida, o backpropagation converge para mínimos diferentes, cada um com um valor de erro ligeiramente diferente.
- **O número de épocas varia** porque o caminho até cada mínimo tem comprimento e inclinação diferentes. Pontos de partida mais próximos do mínimo, ou em regiões mais íngremes, reduzem o erro mais rapidamente; pontos mais distantes, ou que passam por regiões mais planas, levam mais épocas até que a variação do EQM fique abaixo de ε.

Além disso, há várias combinações de pesos equivalentes: os 10 neurônios escondidos podem dividir a tarefa de formas diferentes, de modo que não existe uma única solução ótima.

Nos resultados obtidos, o EQM final variou entre 0,000769 e 0,000789, e o número de épocas entre 105 e 123. A variação pequena indica que todos os treinamentos convergiram para mínimos de qualidade semelhante, ou seja, todos obtiveram boas aproximações da função.

## Questão 5 – Validação com o conjunto de teste

Erro relativo de cada amostra: |d − y_rede| / |d| × 100. O erro relativo médio e a variância foram calculados sobre os 20 padrões de teste.

| Amostra | x1 | x2 | x3 | d | y rede (T1) | y rede (T2) | y rede (T3) | y rede (T4) | y rede (T5) |
|:-------:|:--:|:--:|:--:|:-:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 1  | 0.0611 | 0.2860 | 0.7464 | 0.4831 | 0.4858 | 0.4829 | 0.4857 | 0.4840 | 0.4825 |
| 2  | 0.5102 | 0.7464 | 0.0860 | 0.5965 | 0.5887 | 0.5930 | 0.5917 | 0.5905 | 0.5907 |
| 3  | 0.0004 | 0.6916 | 0.5006 | 0.5318 | 0.5252 | 0.5260 | 0.5271 | 0.5253 | 0.5242 |
| 4  | 0.9430 | 0.4476 | 0.2648 | 0.6843 | 0.7044 | 0.7060 | 0.7038 | 0.7042 | 0.7061 |
| 5  | 0.1399 | 0.1610 | 0.2477 | 0.2872 | 0.2917 | 0.2894 | 0.2895 | 0.2934 | 0.2914 |
| 6  | 0.6423 | 0.3229 | 0.8567 | 0.7663 | 0.7520 | 0.7499 | 0.7512 | 0.7503 | 0.7520 |
| 7  | 0.6492 | 0.0007 | 0.6422 | 0.5666 | 0.5682 | 0.5662 | 0.5689 | 0.5661 | 0.5667 |
| 8  | 0.1818 | 0.5078 | 0.9046 | 0.6601 | 0.6784 | 0.6753 | 0.6775 | 0.6754 | 0.6773 |
| 9  | 0.7382 | 0.2647 | 0.1916 | 0.5427 | 0.5310 | 0.5331 | 0.5336 | 0.5313 | 0.5318 |
| 10 | 0.3879 | 0.1307 | 0.8656 | 0.5836 | 0.6024 | 0.5990 | 0.6023 | 0.5991 | 0.6000 |
| 11 | 0.1903 | 0.6523 | 0.7820 | 0.6950 | 0.6893 | 0.6871 | 0.6886 | 0.6873 | 0.6888 |
| 12 | 0.8401 | 0.4490 | 0.2719 | 0.6790 | 0.6719 | 0.6736 | 0.6721 | 0.6715 | 0.6735 |
| 13 | 0.0029 | 0.3264 | 0.2476 | 0.2956 | 0.3001 | 0.2980 | 0.2979 | 0.3014 | 0.2991 |
| 14 | 0.7088 | 0.9342 | 0.2763 | 0.7742 | 0.7802 | 0.7820 | 0.7793 | 0.7807 | 0.7811 |
| 15 | 0.1283 | 0.1882 | 0.7253 | 0.4662 | 0.4657 | 0.4624 | 0.4653 | 0.4641 | 0.4624 |
| 16 | 0.8882 | 0.3077 | 0.8931 | 0.8093 | 0.8177 | 0.8166 | 0.8170 | 0.8182 | 0.8181 |
| 17 | 0.2225 | 0.9182 | 0.7820 | 0.7581 | 0.7782 | 0.7757 | 0.7763 | 0.7776 | 0.7780 |
| 18 | 0.1957 | 0.8423 | 0.3085 | 0.5826 | 0.5878 | 0.5905 | 0.5903 | 0.5890 | 0.5886 |
| 19 | 0.9991 | 0.5914 | 0.3933 | 0.7938 | 0.7967 | 0.7979 | 0.7952 | 0.7977 | 0.7979 |
| 20 | 0.2299 | 0.1524 | 0.7353 | 0.5012 | 0.4963 | 0.4932 | 0.4963 | 0.4943 | 0.4933 |
| **Erro Relativo Médio (%)** | | | | | **1.4066** | **1.2938** | **1.2564** | **1.4411** | **1.3974** |
| **Variância (%)** | | | | | **0.8059** | **0.6953** | **0.7657** | **0.6842** | **0.7389** |

Os cinco treinamentos apresentaram erro relativo médio entre 1,26% e 1,44%, com saídas muito próximas dos valores desejados em todos os padrões de teste. Isso mostra que a rede foi capaz de generalizar bem para dados que não foram usados no treinamento.

## Questão 6 – Configuração mais adequada

A configuração mais adequada para o sistema de ressonância magnética é a do **T3**, por apresentar o **menor erro relativo médio** sobre o conjunto de teste (1,2564%). Como o conjunto de teste contém padrões não utilizados no treinamento, esse é o principal indicador da capacidade de generalização da rede.

A variância do T3 (0,7657%) é intermediária e muito próxima à dos demais treinamentos (entre 0,6842% e 0,8059%), indicando que seus erros também são consistentes entre os padrões. O T4 tem a menor variância, mas também o maior erro relativo médio (1,4411%), ou seja, erra de forma uniforme, porém mais do que os outros. O T2 seria a segunda melhor opção, com o segundo menor erro médio (1,2938%) e baixa variância (0,6953%).

Vale observar que o T4 obteve o menor EQM no treinamento (0,000769), mas o pior desempenho na validação. Isso mostra que o menor erro no conjunto de treinamento não garante a melhor generalização, e por isso a escolha deve ser feita com base no conjunto de teste.
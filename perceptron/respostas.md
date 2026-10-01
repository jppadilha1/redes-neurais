## Questão 2 – Resultados dos treinamentos

Taxa de aprendizagem η = 0,01. Os pesos iniciais foram sorteados entre 0 e 1, com o gerador de números aleatórios reiniciado com uma semente diferente em cada treinamento (sementes 1 a 5).

| Treinamento | Semente | w0 inicial | w1 inicial | w2 inicial | w3 inicial | w0 final | w1 final | w2 final | w3 final | Nº de Épocas |
|:-----------:|:-------:|:----------:|:----------:|:----------:|:----------:|:--------:|:--------:|:--------:|:--------:|:------------:|
| 1º (T1) | 1 | 0.5118 | 0.9505 | 0.1442 | 0.9486 | -2.8882 | 1.3980 | 2.3916 | -0.6919 | 337 |
| 2º (T2) | 2 | 0.2616 | 0.2985 | 0.8142 | 0.0919 | -3.0584 | 1.5650 | 2.4589 | -0.7299 | 398 |
| 3º (T3) | 3 | 0.0856 | 0.2368 | 0.8013 | 0.5822 | -3.1544 | 1.6158 | 2.5500 | -0.7522 | 437 |
| 4º (T4) | 4 | 0.9431 | 0.5113 | 0.9762 | 0.0808 | -3.1169 | 1.6202 | 2.5313 | -0.7455 | 459 |
| 5º (T5) | 5 | 0.8050 | 0.8079 | 0.5153 | 0.2858 | -2.9950 | 1.4575 | 2.4428 | -0.7165 | 387 |

Os cinco treinamentos convergiram, ou seja, terminaram com uma época sem nenhum erro de classificação no conjunto de treinamento. Isso mostra que as duas classes são linearmente separáveis.

## Questão 3 – Classificação das amostras de óleo

Convenção: **-1 = classe C1** e **+1 = classe C2**.

| Amostra | x1 | x2 | x3 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
|:-------:|:--:|:--:|:--:|:------:|:------:|:------:|:------:|:------:|
| 1  | -0.3565 | 0.0620  | 5.9891 | -1 | -1 | -1 | -1 | -1 |
| 2  | -0.7842 | 1.1267  | 5.5912 | +1 | +1 | +1 | +1 | +1 |
| 3  | 0.3012  | 0.5611  | 5.8234 | +1 | +1 | +1 | +1 | +1 |
| 4  | 0.7757  | 1.0648  | 8.0677 | +1 | +1 | +1 | +1 | +1 |
| 5  | 0.1570  | 0.8028  | 6.3040 | +1 | +1 | +1 | +1 | +1 |
| 6  | -0.7014 | 1.0316  | 3.6005 | +1 | +1 | +1 | +1 | +1 |
| 7  | 0.3748  | 0.1536  | 6.1537 | -1 | -1 | -1 | -1 | -1 |
| 8  | -0.6920 | 0.9404  | 4.4058 | +1 | +1 | +1 | +1 | +1 |
| 9  | -1.3970 | 0.7141  | 4.9263 | -1 | -1 | -1 | -1 | -1 |
| 10 | -1.8842 | -0.2805 | 1.2548 | -1 | -1 | -1 | -1 | -1 |

As amostras 1, 7, 9 e 10 foram classificadas como C1, e as amostras 2, 3, 4, 5, 6 e 8 como C2. Os cinco treinamentos concordaram em todas as amostras. Embora os pesos finais sejam um pouco diferentes, os planos de separação encontrados são equivalentes na prática para esses dados.

## Questão 4 – Por que o número de épocas varia?

Porque os pesos iniciais são aleatórios, e cada treinamento parte de um ponto diferente. Alguns pontos de partida estão mais próximos de uma solução e exigem menos correções; outros estão mais distantes e exigem mais. Além disso, existem infinitos planos capazes de separar as classes, e o algoritmo para no primeiro que classifica corretamente todas as amostras. Por isso cada treinamento termina com pesos finais e número de épocas diferentes (de 337 a 459 épocas neste trabalho).

## Questão 5 – Principal limitação do perceptron

O perceptron só consegue resolver problemas **linearmente separáveis**, isto é, problemas em que as classes podem ser separadas por uma única reta (2D), plano (3D) ou hiperplano. Se as classes se sobrepõem ou exigem uma fronteira não linear, como no problema do XOR, o treinamento não converge. Para esses casos é necessário usar redes com múltiplas camadas, como o Perceptron Multicamadas (MLP).
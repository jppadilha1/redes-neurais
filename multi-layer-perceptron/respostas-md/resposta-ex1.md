## Backpropagation – Sequência e ajuste das matrizes W1, W2 e W3

### 1. Notação e dimensões das matrizes

- **x** = [x0, x1, ..., xN]: vetor de entrada, com x0 = -1 (entrada do limiar/bias).
- **g(·)**: função de ativação (ex.: sigmoide logística), com derivada g'(·).
- **η**: taxa de aprendizagem; **d**: saída desejada do padrão.
- **I**: soma ponderada (potencial de ativação); **Y**: saída após a ativação.
- Índice **i** = entradas (0 a N); índice **j** = neurônios escondidos (1 a N1).

| Matriz | Liga | Dimensão | Elemento |
|:------:|:----:|:--------:|:--------:|
| **W1** | 1ª camada (entrada) → 2ª camada (escondida) | N1 × (N+1) | W1_ji: peso da entrada i para o neurônio escondido j |
| **W2** | 2ª camada (escondida) → 3ª camada (saída) | 1 × (N1+1) | W2_j: peso do neurônio escondido j para a saída |
| **W3** | 1ª camada (entrada) → 3ª camada (saída) | 1 × N | W3_i: peso da entrada i direto para a saída |

Cada linha de W1 corresponde ao vetor de pesos de um neurônio escondido. O limiar do neurônio de saída já está em W2 (coluna j = 0), por isso W3 não possui coluna de bias.

### 2. Fase forward (propagação adiante)

Para cada padrão k (k = 1, ..., P), o sinal é propagado camada a camada, sem alterar os pesos.

**Camada escondida:**

```
I1_j = Σ (i = 0..N) W1_ji · x_i          para j = 1, ..., N1
Y1_j = g(I1_j)
Y1_0 = -1                                (bias para a camada de saída)
```

**Camada de saída** (recebe a camada escondida via W2 e as entradas diretamente via W3):

```
I2 = Σ (j = 0..N1) W2_j · Y1_j  +  Σ (i = 1..N) W3_i · x_i
Y2 = g(I2)
```

**Erro do padrão k:**

```
E(k) = ½ · (d(k) − Y2)²
```

### 3. Fase backward (retropropagação do erro)

Os pesos são ajustados pela descida do gradiente, w ← w − η · ∂E/∂w, aplicando a regra da cadeia da saída em direção à entrada.

**3.1 Gradiente local da camada de saída:**

```
δ2 = (d − Y2) · g'(I2)
```

**3.2 Gradiente local da camada escondida.** Os neurônios escondidos não possuem saída desejada, então recebem o erro da saída proporcional ao peso que os liga a ela:

```
δ1_j = δ2 · W2_j · g'(I1_j)              para j = 1, ..., N1
```

A matriz W3 não participa do cálculo de δ1, pois liga a entrada diretamente à saída e não passa pela camada escondida.

**3.3 Ajuste de W2 (escondida → saída).** Pela regra da cadeia, ∂E/∂W2_j = −(d − Y2) · g'(I2) · Y1_j = −δ2 · Y1_j. Logo:

```
W2_j ← W2_j + η · δ2 · Y1_j              para j = 0, ..., N1
```

**3.4 Ajuste de W3 (entrada → saída).** Como I2 depende diretamente de x_i via W3, ∂E/∂W3_i = −δ2 · x_i. Logo:

```
W3_i ← W3_i + η · δ2 · x_i               para i = 1, ..., N
```

**3.5 Ajuste de W1 (entrada → escondida).** O erro percorre o caminho saída → neurônio j → peso W1_ji:

```
∂E/∂W1_ji = −(d − Y2) · g'(I2) · W2_j · g'(I1_j) · x_i = −δ1_j · x_i
```

Logo:

```
W1_ji ← W1_ji + η · δ1_j · x_i           para j = 1, ..., N1 e i = 0, ..., N
```

**Forma matricial** (T = transposta):

```
W2 ← W2 + η · δ2 · (Y1)ᵀ                 Y1 = [Y1_0, ..., Y1_N1]
W3 ← W3 + η · δ2 · (x̃)ᵀ                  x̃ = [x1, ..., xN]  (sem o bias)
W1 ← W1 + η · δ1 · xᵀ                    δ1 = [δ1_1, ..., δ1_N1]ᵀ
```

### 4. Sequência completa do algoritmo

1. Inicializar W1, W2 e W3 com valores aleatórios pequenos; definir η e a precisão ε.
2. Repetir (cada repetição é uma época):
   1. Guardar EQM_anterior.
   2. Para cada padrão k = 1, ..., P:
      1. **Forward:** calcular I1, Y1, I2 e Y2.
      2. **Backward:** calcular δ2 e, em seguida, δ1 (usando W2 *antes* de ser ajustada).
      3. **Ajustar** W2, W3 e W1.
   3. Calcular o erro quadrático médio da época:
```
      EQM = (1/P) · Σ (k = 1..P) ½ · (d(k) − Y2(k))²
```
3. Parar quando |EQM_atual − EQM_anterior| ≤ ε.

Resumindo: em cada padrão, o sinal percorre a rede da entrada para a saída (forward); o erro é medido na saída e propagado de volta (backward), primeiro gerando δ2, que ajusta diretamente W2 e W3, e depois gerando δ1, que ajusta W1.
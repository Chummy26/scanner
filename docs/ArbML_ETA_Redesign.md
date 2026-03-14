# ArbML — Análise de ETA: Diagnóstico, Literatura e Plano de Redesign

## 1. O que o ETA actual faz

### 1.1 Treino

```python
# ml_model.py:168-172 — ETA head
self.eta_out = nn.Sequential(
    nn.Linear(hidden_size, head_hidden),
    nn.SiLU(),
    nn.Dropout(dropout),
    nn.Linear(head_hidden, 1),       # ← 1 único neurónio de saída
)

# train_model.py:64-75 — ETA loss
def _eta_loss(criterion, eta_raw, target_eta_seconds, target_class):
    mask = target_class > 0.5        # ← só positivos (y=1)
    if not torch.any(mask):
        return eta_raw.new_tensor(0.0)
    target = torch.log1p(target_eta_seconds[mask])   # ← log-transform
    pred = eta_raw.squeeze(1)[mask]
    return criterion(pred, target)    # ← SmoothL1Loss
```

### 1.2 Label

```python
# ml_dataset.py:848
"y_eta": float(first_qualified.end_ts) - float(current_ts)
# Para positivos: tempo em segundos até o episódio qualificado FECHAR
# Para negativos: y_eta = 0.0 (mascarado no loss)
```

### 1.3 Inferência

```python
# ml_analyzer.py:340
eta_seconds = int(max(60.0, math.expm1(max(float(eta_raw.item()), 0.0))))
# Inverte log1p → 1 ponto único em segundos
# Clamp mínimo 60s, máximo = prediction_horizon
```

### 1.4 Dashboard

```
┌─────────────────────────────────────────┐
│  ETA (model):     32m                   │  ← ponto único do modelo
│  ETA empírico:    Q25=18m Q50=32m Q75=61m│  ← estatísticas históricas
└─────────────────────────────────────────┘
```

Os quantis Q25/Q50/Q75 no dashboard NÃO vêm do modelo. Vêm de
`get_pair_recurring_context()` que calcula percentis da duração de
episódios passados daquele par. São médias históricas, não condicionais
ao estado actual das features.

---

## 2. Os 5 problemas fundamentais

### P1 — CRÍTICO: Output ponto único sem incerteza

O modelo produz 1 valor: "ETA = 32 minutos". O operador não sabe
se isso significa "quase certamente 32min" ou "pode ser 5min ou 3h".

A Uber descobriu que diferentes use cases precisam de diferentes
estimativas pontuais: média para cálculo de tarifas, mediana para UX,
P95 para planeamento. O DeepETA usa uma loss parametrizada
(Asymmetric Huber) que pode produzir qualquer quantil mudando um
parâmetro omega, sem retreinar o modelo.

A DoorDash modela a distribuição completa via Weibull, reconhecendo
que tempos de entrega seguem distribuição de cauda longa que não pode
ser capturada por Gaussiana ou exponencial.

O ArbML tem o MESMO problema: episódios de spread têm cauda longa
(p50=42s, mas outliers até horas). Um ponto único é insuficiente.

### P2 — CRÍTICO: Quantis empíricos desconectados do modelo

```
ESTADO ACTUAL:
  Modelo → ponto único (SmoothL1 em log-space)
  Dashboard → Q25/Q50/Q75 de episódios históricos (NÃO do modelo)
  
  Os quantis NÃO consideram:
  - O spread actual (se está a 5%, ETA deveria ser mais curto)
  - A velocidade de mudança (delta_entry alto = episódio mais rápido)
  - A hora do dia (liquidez varia)
  - O par específico no contexto actual

O CORRECTO:
  Modelo → distribuição condicional P(tempo|features)
  Dashboard → Q25/Q50/Q75 extraídos DESSA distribuição
  
  Os quantis CONSIDERAM tudo que as features capturam.
```

### P3 — ALTO: Right-censoring ignorado

```python
# Samples com y=0 recebem y_eta=0.0 e são mascarados:
mask = target_class > 0.5
# Resultado: modelo treina ETA apenas com ~120K positivos (4.57%)
# Os 2.5M negativos são completamente ignorados para ETA

# MAS esses negativos contêm informação:
# y=0, horizonte=4h → "em 4h não houve episódio qualificado"
# Isso é RIGHT-CENSORING: sabemos que T > 4h, não que T = ∞
```

A survival analysis trata isto com censoring correction. O modelo
actual desperdiça 95% dos dados para treino de ETA.

Na literatura, o paper "Deep learning for survival analysis" (Springer,
2024) documenta extensivamente como métodos como DeepHit, Nnet-survival,
e Weibull Time-to-Event RNN (WTTE-RNN) usam observações censuradas
para melhorar a estimativa da distribuição de tempo até evento.

### P4 — ALTO: SmoothL1 é loss inadequada

```
SmoothL1(log1p(pred), log1p(target))
```

Problemas:
1. Simétrica — trata sobre-estimação e sub-estimação igualmente.
   Na arbitragem, sub-estimar ETA (dizer 10min quando demora 1h)
   é PIOR que sobre-estimar (dizer 1h quando demora 10min).
   O operador que espera 10min e não vê resultado desiste.

2. Não modela a distribuição — SmoothL1 produz algo próximo da
   mediana condicional, mas não dá informação sobre a dispersão.

3. Log-transform ajuda mas não resolve — log1p comprime a cauda
   longa mas a loss continua simétrica e pontual.

A Uber usa Asymmetric Huber Loss com parâmetros delta (robustez
a outliers) e omega (assimetria). O DoorDash usa distribuição de
Weibull que naturalmente modela cauda longa.

### P5 — MÉDIO: Label é tempo até FECHAR, não até ATINGIR

```python
# ml_dataset.py:848
y_eta = float(first_qualified.end_ts) - float(current_ts)
# end_ts = quando o episódio FECHOU (spread voltou ao baseline)
# NÃO quando o episódio ATINGIU o pico
```

O operador quer saber: "quanto tempo até o spread chegar a X%?"
O label actual mede: "quanto tempo até o episódio inteiro acabar".
São coisas diferentes. O pico pode acontecer em 5min mas o episódio
só fecha em 45min.

---

## 3. O que a literatura recomenda

### 3.1 Uber DeepETA — Asymmetric Huber Loss

```
L(e) = {
  ω * 0.5 * e²/δ         se |e| < δ  (sub-estimação)
  (1-ω) * 0.5 * e²/δ     se |e| < δ  (sobre-estimação)
  ω * (|e| - 0.5*δ)       se |e| >= δ (sub-estimação)
  (1-ω) * (|e| - 0.5*δ)   se |e| >= δ (sobre-estimação)
}

- δ controla robustez a outliers (interpola entre L2 e L1)
- ω controla assimetria (custo relativo de sub vs sobre-estimação)
- Variando ω, produz qualquer quantil sem retreinar
```

### 3.2 DoorDash — Weibull Distribution

```
P(T ≤ t) = 1 - exp(-(t/λ)^k)

- O modelo prediz parâmetros (λ, k) em vez de ponto único
- λ = scale (ETA típico), k = shape (dispersão)
- k > 1: probabilidade aumenta com tempo (episódio "amadurece")
- k < 1: probabilidade diminui (se não aconteceu rápido, demora)
- Cauda longa natural — ideal para tempos de evento
```

### 3.3 Quantile Regression (Koenker & Bassett, 1978)

```
Pinball Loss:
L_τ(e) = max(τ*e, (τ-1)*e)

- Para τ=0.5 → mediana
- Para τ=0.25 → Q25
- Para τ=0.75 → Q75

Multi-quantile head:
  Modelo → [Q10, Q25, Q50, Q75, Q90]
  5 neurónios de saída em vez de 1
  Loss = soma de pinball losses para cada quantil
```

### 3.4 Survival Analysis — Censored Quantile Regression

O paper "Censored Quantile Regression Neural Networks" (NeurIPS, 2022)
mostra como combinar quantile regression com NNs em dados censurados
(onde o evento pode não ter sido observado). Usa o estimador de Portnoy
adaptado para redes neurais.

O paper "DeepQuantReg" propõe quantile regression com check function
ajustada por IPCW (inverse probability of censoring weights) para
dados com right-censoring — exactamente o caso do ArbML.

### 3.5 WTTE-RNN — Weibull Time-to-Event RNN

Martinsson (2016) propõe LSTM que prediz parâmetros de Weibull
directamente, tratando censoring nativamente. Cada sample contribui
para o loss, censurado ou não:

```
Censurado (y=0):   L = -log(S(t)) = (t/λ)^k     (sobreviveu até t)
Não-censurado (y=1): L = -log(f(t)) = log hazard  (evento em t)
```

---

## 4. Redesign proposto para ArbML ETA

### 4.1 Arquitectura: Multi-Quantile Head

Substituir o ETA head de 1 neurónio por 5 neurónios (Q10/Q25/Q50/Q75/Q90):

```python
# ANTES (ml_model.py):
self.eta_out = nn.Sequential(
    nn.Linear(hidden_size, head_hidden),
    nn.SiLU(),
    nn.Dropout(dropout),
    nn.Linear(head_hidden, 1),
)

# DEPOIS:
self.eta_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
self.eta_out = nn.Sequential(
    nn.Linear(hidden_size, head_hidden),
    nn.SiLU(),
    nn.Dropout(dropout),
    nn.Linear(head_hidden, len(self.eta_quantiles)),  # 5 outputs
)
```

### 4.2 Loss: Pinball Loss com Censoring Correction

```python
def quantile_loss_censored(pred_quantiles, target_eta, target_class, 
                            horizon_sec, quantiles=[0.10, 0.25, 0.50, 0.75, 0.90]):
    """
    pred_quantiles: (batch, num_quantiles) em log-space
    target_eta: seconds até evento (0.0 se censurado)
    target_class: 1.0 se evento observado, 0.0 se censurado
    horizon_sec: horizonte de predição (4h = 14400s)
    """
    total_loss = 0.0
    for i, tau in enumerate(quantiles):
        pred_q = pred_quantiles[:, i]  # log-space
        
        # Não-censurados (y=1): pinball loss normal
        uncensored = target_class > 0.5
        if uncensored.any():
            target_log = torch.log1p(target_eta[uncensored])
            error = target_log - pred_q[uncensored]
            loss_unc = torch.where(
                error >= 0,
                tau * error,
                (tau - 1.0) * error
            ).mean()
            total_loss = total_loss + loss_unc
        
        # Censurados (y=0): sabemos que T > horizon
        # Se pred_q < log1p(horizon): penalizar (prediz cedo demais)
        censored = target_class <= 0.5
        if censored.any():
            horizon_log = math.log1p(horizon_sec)
            # Se o quantil predito é menor que o horizonte,
            # está a sub-estimar o tempo (episódio não aconteceu em 4h)
            error_cens = horizon_log - pred_q[censored]
            # Só penalizar se pred < horizon (sub-estimação)
            loss_cens = torch.where(
                error_cens > 0,
                tau * error_cens,
                torch.zeros_like(error_cens)
            ).mean() * 0.1  # peso menor para censurados
            total_loss = total_loss + loss_cens
    
    return total_loss / len(quantiles)
```

### 4.3 Label: Tempo até ATINGIR pico (não até fechar)

```python
# ANTES:
y_eta = float(first_qualified.end_ts) - float(current_ts)
# Tempo até o episódio FECHAR

# DEPOIS (dois targets):
y_eta_to_peak = float(first_qualified.peak_ts) - float(current_ts)
y_eta_to_close = float(first_qualified.end_ts) - float(current_ts)
# O modelo prediz distribuição de tempo até ATINGIR o pico
# (mais útil para o operador — "quando posso entrar/sair")
```

Nota: peak_ts precisa de ser adicionado ao TrackerEpisode. Actualmente
só guarda start_ts e end_ts. O peak_ts é quando peak_entry_spread
foi observado.

### 4.4 Inferência: Quantis condicionais

```python
# ANTES:
eta_seconds = int(max(60.0, math.expm1(max(float(eta_raw.item()), 0.0))))

# DEPOIS:
logits, eta_quantile_raw = self.model(tensor)
# eta_quantile_raw shape: (1, 5) → [Q10, Q25, Q50, Q75, Q90]
eta_quantiles_sec = [
    int(max(30.0, math.expm1(max(float(q), 0.0))))
    for q in eta_quantile_raw[0]
]
# Enforce monotonicity (Q10 <= Q25 <= Q50 <= Q75 <= Q90)
for i in range(1, len(eta_quantiles_sec)):
    eta_quantiles_sec[i] = max(eta_quantiles_sec[i], eta_quantiles_sec[i-1])
```

### 4.5 Dashboard: Distribuição visual

```
ANTES:
  ETA: 32m

DEPOIS:
  ETA: 18m / 32m / 61m  (rápido / típico / lento)
  ├──────|════════|──────┤
  Q10    Q50      Q90
  
  Incerteza: ±22min (Q75-Q25)
  
  Se incerteza < 10min → "confiança alta"
  Se incerteza > 45min → "confiança baixa"
```

O operador vê imediatamente:
- Cenário optimista (Q10): 10min
- Cenário típico (Q50): 32min
- Cenário pessimista (Q90): 75min
- Dispersão (Q75-Q25): quão incerto é o ETA

---

## 5. Monotonicity constraint

Quantis devem ser ordenados: Q10 ≤ Q25 ≤ Q50 ≤ Q75 ≤ Q90.
Duas abordagens:

**A) Post-hoc sort (simples):**
```python
eta_sorted = torch.sort(eta_quantile_raw, dim=1).values
```

**B) Incremental parametrization (melhor):**
```python
# O modelo prediz Q10 + 4 deltas positivos
# Q10 = eta_out[0]
# Q25 = Q10 + softplus(eta_out[1])
# Q50 = Q25 + softplus(eta_out[2])
# Q75 = Q50 + softplus(eta_out[3])
# Q90 = Q75 + softplus(eta_out[4])
```

A opção B garante monotonicity por construção e é usada na
literatura (MNQ-LSTM paper).

---

## 6. Métricas de avaliação

### Calibração de quantis

```python
# Para cada quantil tau, verificar:
# "tau% dos valores reais caem abaixo do quantil predito?"
coverage_tau = (y_true_eta <= pred_quantile_tau).mean()
# Se tau=0.50 e coverage=0.50 → perfeitamente calibrado
# Se tau=0.50 e coverage=0.30 → modelo sobre-estima (optimista demais)
```

### Pinball Score (Quantile Score)

```python
def pinball_score(y_true, y_pred_quantile, tau):
    error = y_true - y_pred_quantile
    return np.where(error >= 0, tau * error, (tau - 1) * error).mean()
```

### Interval Width (sharpness)

```python
# Intervalos mais estreitos = mais informativos
width = pred_Q90 - pred_Q10  # intervalo de 80%
# Queremos width pequeno MAS com boa cobertura
```

### Winkler Score (combina calibração + sharpness)

```python
alpha = 0.20  # para intervalo de 80% (Q10 a Q90)
width = pred_Q90 - pred_Q10
penalty = 0
if y_true < pred_Q10:
    penalty = (2/alpha) * (pred_Q10 - y_true)
elif y_true > pred_Q90:
    penalty = (2/alpha) * (y_true - pred_Q90)
winkler = width + penalty
# Penaliza intervalos largos E violações de cobertura
```

---

## 7. Fases de implementação

### Fase ETA-1: Multi-quantile head + Pinball loss (~3h)
**Scope:** Mudar arquitectura + loss. Sem censoring correction ainda.

Ficheiros:
- `ml_model.py`: ETA head 1→5 neurónios + monotonicity
- `train_model.py`: Pinball loss em vez de SmoothL1
- `train_model.py`: _eta_metrics expandido com calibração
- `ml_analyzer.py`: Inferência com 5 quantis

**Gate:** Calibração coverage para Q50 deve estar entre 0.40-0.60.

### Fase ETA-2: Censoring correction (~2h)
**Scope:** Usar samples censurados (y=0) no loss de ETA.

Ficheiros:
- `train_model.py`: Loss censurada (penaliza sub-estimação quando T > horizon)
- `ml_dataset.py`: Passar horizon_sec para o loss

**Gate:** ETA MAE deve melhorar (mais dados para treinar).

### Fase ETA-3: Label peak_ts (~2h)
**Scope:** Gravar peak_ts nos episódios e usar como target alternativo.

Ficheiros:
- `spread_tracker.py`: Adicionar peak_ts ao TrackerEpisode
- `ml_dataset.py`: y_eta_to_peak como target principal
- `train_model.py`: Treinar com peak_ts

**Gate:** Q50 mais curto e com melhor calibração (pico é mais previsível que close).

### Fase ETA-4: Dashboard + Métricas de calibração (~2h)
**Scope:** Mostrar distribuição no dashboard e monitorar calibração.

Ficheiros:
- `ml_analyzer.py`: _eta_payload com Q10/Q25/Q50/Q75/Q90
- `soak_runbook.py`: Gate de calibração ETA no Stage 2
- Dashboard frontend: Visualização de intervalo

**Gate:** Winkler score < threshold (definir após baseline).

### Fase ETA-5 (futuro): Weibull head
**Scope:** Substituir quantile head por Weibull paramétrica para
censoring nativo e distribuição contínua.

---

## 8. Comparação antes/depois

```
ANTES (actual):
┌───────────────────────────────────────────────────┐
│ Modelo: 1 ponto (SmoothL1 em log-space)           │
│ Label: tempo até episódio FECHAR                   │
│ Censoring: ignorado (y=0 mascarado)                │
│ Dashboard: ponto único + quantis empíricos          │
│ Assimetria: nenhuma                                 │
│ Dados usados: ~120K (4.57% — só positivos)          │
│ Métricas: MAE, RMSE (sem calibração)                │
│ O que o operador vê: "ETA: 32m" (sem intervalo)     │
└───────────────────────────────────────────────────┘

DEPOIS (v3):
┌───────────────────────────────────────────────────┐
│ Modelo: 5 quantis (Pinball loss com censoring)      │
│ Label: tempo até PICO do episódio                   │
│ Censoring: samples censurados contribuem para loss   │
│ Dashboard: Q10/Q25/Q50/Q75/Q90 condicionais         │
│ Assimetria: pinball loss naturalmente assimétrica    │
│ Dados usados: ~2.6M (100% — censurados incluídos)   │
│ Métricas: calibração, pinball score, Winkler         │
│ O que o operador vê: "18m / 32m / 61m" + confiança  │
└───────────────────────────────────────────────────┘
```

---

## 9. Referências

- Wu & Wu (2019). DeepETA: Spatial-Temporal Sequential NN for ETA. AAAI.
- Uber Blog (2022). DeepETA: How Uber Predicts Arrival Times.
- Hu et al. (2022). DeeprETA: An ETA Post-processing System at Scale. arXiv:2206.02127.
- DoorDash (2024). Precision in Motion: Deep Learning for Smarter ETA Predictions.
- Martinsson (2016). WTTE-RNN: Weibull Time to Event RNN. Master's Thesis.
- Koenker & Bassett (1978). Regression Quantiles. Econometrica.
- Jia & Jeong (2021). Deep learning for quantile regression under right censoring. Computational Statistics & Data Analysis.
- Qin et al. (2022). Censored Quantile Regression Neural Networks. NeurIPS.
- Lee et al. (2018). DeepHit: A Deep Learning Approach to Survival Analysis. AAAI.
- Springer (2024). Deep learning for survival analysis: a review. AI Review.
- arXiv (2024). An Introduction to Deep Survival Analysis Models. arXiv:2410.01086.

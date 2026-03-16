# ArbML — Analise State-of-the-Art e Plano de Reconstrucao

**15 de Marco de 2026 — v3.0**
**Baseado em: 32 recursos do guia + 60+ papers + 7 agentes PhD + 3 agentes de verificacao + ETA Redesign doc + Post-V3 Roadmap**

---

## Sumario Executivo

O ArbML e um signal engine de arbitragem cross-exchange de crypto baseado em LSTM com temporal attention. Esta analise cruza 32 recursos do guia original com pesquisa complementar de 60+ papers (2024-2026), verificados contra fontes primarias. O documento mapeia **26 acoes de reconstrucao (R1-R26)** com evidencia quantitativa, codigo proposto, e gates de aceitacao.

**Parametros do dominio sao VARIAVEIS** — interval de recording (atualmente 15s), sequence length (15), prediction horizon (4h), hidden size (64 inference / 128 treino) podem todos ser redesenhados se a evidencia justificar. Esta flexibilidade reabilita propostas que foram inicialmente desclassificadas por "domain mismatch".

**A mudanca de maior impacto verificado**: O ETA head treina com apenas **4.57% dos dados** (confirmado no codigo: `mask = target_class > 0.5`). O pipeline ETA (R19-R22) corrige isto com censoring Portnoy + multi-quantile + peak_ts label + Asymmetric Huber Loss.

---

# PARTE 0 — PARAMETROS VARIAVEIS DO DOMINIO

Os seguintes parametros foram tratados como fixos nas versoes anteriores mas sao redesenhaveis:

| Parametro | Valor Atual | Range Considerado | Impacto se mudar |
|-----------|-------------|------------------|-----------------|
| Record interval | 15s | 1s - 5min | Muda viabilidade de OBI (1-3s), HMM (1-5min), seq coverage |
| Sequence length | 15 steps | 10 - 100+ | VSN viavel com seq≥60, MHA 4-head com seq≥30 |
| Prediction horizon | 4h (14400s) | 30min - 8h | OBI viavel com horizon≤1h, funding rate diff mais/menos relevante |
| Hidden size | 64 (inference), 128 (treino) | 64 - 256 | MHA 4-head: 32-dim/head com hidden=128 (razoavel) |
| Num features | 40 (V3) | 40 - 50 | +3 features (V4 volume ratio + funding) = 43. Mais se OBI viavel |

**Implicacao**: Propostas R1 (VSN), R2 (MHA), R3 (OBI), R5 (HMM) que foram desclassificadas por "domain mismatch" ganham viabilidade sob parametros diferentes. Cada R agora especifica **sob quais parametros e viavel**.

---

# PARTE I — ARQUITETURA DO MODELO

## 1. LSTM vs Transformers: O Veredito Empirico

### O Benchmark Definitivo (arXiv:2603.01820, Marco 2026)

O maior benchmark publicado de deep learning para series temporais financeiras testou 10+ arquiteturas em dados diarios de futuros (2010-2025):

| Rank | Modelo | Sharpe | CAGR | Max Drawdown |
|------|--------|--------|------|-------------|
| 1 | **VLSTM** (VSN + LSTM) | **2.40** | 26.3% | -22.9% |
| 2 | LPatchTST (LSTM + PatchTST) | 2.31 | 25.5% | -17.4% |
| 3 | TFT (Temporal Fusion Transformer) | 2.27 | 24.0% | -23.2% |
| 4 | xLSTM | 1.79 | 19.4% | -14.1% |
| 7 | LSTM vanilla | 1.48 | — | — |
| 8 | Mamba2 | 0.78 | — | — |
| 10 | **iTransformer** | **0.38** | — | — |

**Fato critico**: VLSTM Sharpe **6.3x** > iTransformer no mesmo dataset.

**Mecanismo**: "Architectures maintaining explicit recurrent state representations consistently outperformed purely attention-based models" porque:
1. LSTM gates adaptam-se a sinais intermitentes em dados ruidosos
2. Self-attention e suscetivel a overfitting em outliers (fat tails)
3. Transformers degradam em mudancas de regime

### Confirmacoes adicionais

- **"Transformers versus LSTMs for Electronic Trading" (OpenReview)**: Em sequencias diferenciais (spreads), "LSTM-based models demonstrate superior and more consistent performance"
- **Zeng et al. (AAAI 2023)**: DLinear (1 camada linear) superou Informer/Autoformer/FEDformer em benchmarks long-horizon
- **Foundation Models descartados**: Chronos fine-tuned Sharpe 3.97 bruto → **negativo apos 3bps slippage** (Valeyre & Aboura, arXiv:2412.09394). TimesFM 500M zero-shot: retorno **-1.47%** (arXiv:2511.18578)

### Estado atual vs proposto

| Aspecto | ArbML Atual | Proposto | Condicao de viabilidade |
|---------|-------------|----------|------------------------|
| Arquitetura | LSTM (h=64/128, 2 layers) | **VSN reduced + LSTM** | Seq≥30 e hidden≥128 para VSN. L1 reg como alternativa zero-cost |
| Attention | Single-head Bahdanau (2K params) | **2-4 head MHA** ou **temp scaling** | 4-head viavel se hidden=128 (32-dim/head). 2-head mais seguro |
| Feature gating | Nenhum | **VSN micro (h=16, ~60K params)** ou **L1 reg** | VSN micro ~0.7x modelo. L1 = zero overhead |
| BiLSTM | Nao tem | **REJEITADO** | Look-ahead bias fatal em inference (ICS-LSTM confirmou) |
| sLSTM (xLSTM) | Nao tem | **REJEITADO** | API incompativel, geometric ergodicity provada (arXiv:2408.10006), zero evidencia seq<96 |
| Sequence length | 15 fixo | **Validar com PACF** | Computar PACF por par antes de ablation |

---

## 2. R1: VSN (Variable Selection Network)

### O que e
VSN e uma soft-gating layer que aprende a importancia de cada feature em cada timestep. Core do VLSTM vencedor do benchmark 2026.

### Implementacao (GRN + VSN)

```python
class GatedResidualNetwork(nn.Module):
    """Bloco GRN do TFT (Lim et al., Google Research 2021, Int J Forecasting)."""
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.skip = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = torch.sigmoid(self.gate(h))
        return self.layernorm(gate * h + (1 - gate) * residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: soft-gating per-feature per-timestep."""
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.flattened_grn = GatedResidualNetwork(input_size, hidden_size, dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        B, T, F = x.shape
        flat = x.reshape(B * T, F)
        weights = self.softmax(self.output_proj(self.flattened_grn(flat)))
        weights = weights.reshape(B, T, F)
        return weights * x
```

### Parametros verificados (Keras source)

| Variante | N=40 features | Params | vs modelo atual (83K) |
|----------|--------------|--------|----------------------|
| VSN full h=64 | ~851K | **10.2x** | Risco de overfitting |
| VSN reduced h=32 | ~217K | **2.6x** | Viavel com dataset grande |
| **VSN micro h=16** | **~60K** | **0.7x** | **Recomendado** |
| L1 regularization | 0 | **0x** | Alternativa zero-cost |

### Viabilidade por parametros

- **Parametros atuais** (seq=15, h=64): VSN micro viavel. L1 reg como alternativa mais segura
- **Se seq≥60 e h=128**: VSN reduced viavel (benchmark testou nestas condicoes)

### Riscos verificados

- Softmax gating collapse com features correlacionadas (arXiv:2405.13997)
- Benchmark e daily futures, nao 15s crypto. Transferencia NAO validada
- Wang 2025 (arXiv:2506.05764): "Better inputs matter more than stacking another hidden layer"

---

## 3. R2: Multi-Head Attention

### Estado atual
`TemporalAttention`: Linear(64→32) → Tanh → Linear(32→1) → Softmax → weighted sum. **Pooling scorer**, NAO self-attention encoder.

### Nota de verificacao
NeurIPS 2024 (arXiv:2405.16877, CATS) encontrou self-attention "potentially harmful" para time series. **MAS**: essa critica e sobre **encoder self-attention**, NAO pooling attention como a do ArbML. Testou **seq≥96**, nao 15. A critica nao se aplica diretamente.

### Opcoes

```python
# OPCAO A: Learnable temperature (1 param, menor risco)
self.attn_temperature = nn.Parameter(torch.ones(1) * 1.0)
# forward: weights = torch.softmax(scores / self.attn_temperature, dim=1)

# OPCAO B: 2-head (32-dim/head com h=64, ou 64-dim/head com h=128)
self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=2, batch_first=True)

# OPCAO C: 4-head (requer hidden≥128 para 32-dim/head minimo)
self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
```

### Viabilidade por parametros
- **h=64**: Opcao A (temp) ou B (2-head). Opcao C arriscada (16-dim/head)
- **h=128**: Opcao C viavel (32-dim/head). Recomendada

---

## 4. Sequence Length

| Estudo | Dominio | Window Otima | Resolucao |
|--------|---------|-------------|-----------|
| KCS-LSTM (PMC 11639137) | Commodity spread | **11** | Minutos |
| ICS-LSTM (PMC 11784865) | Futures spread | **3** | HF |
| Pratica geral stocks | Equities | 60 | Diario |

**Recomendacao**: Computar PACF por par ANTES do ablation. Se PACF decai no step 8-10 com interval=15s, testar {8,10,12,15}. Se interval mudar para 5s, testar {20,30,40,60}.

---

# PARTE II — FEATURES

## 5. Features V3 Atuais + Propostas V4

### V3 atual (40 features): Validado

V1 micro (10) + V2 multiscale 30m/2h/8h (15) + V3 exit-aware (15).

### V4 proposto: +3 features (43 total)

**R3-A: Volume Ratio** (evidencia forte, custo zero):
```python
"log_volume_ratio": log(volume_buy_exchange / max(volume_sell_exchange, 1e6))
```
- Price discovery literature (Hasbrouck 1995, arXiv:2506.08718): exchange com maior volume lidera
- Dados ja no `market_data.py` — `get_volume(exchange, symbol, market_type)`

**R3-B: Funding Rate Differential** (precisa companion feature):
```python
"funding_rate_diff": rate_buy_exchange - rate_sell_exchange,
"time_to_settlement_frac": (current_utc_ts % 28800) / 28800.0,
```
- Sem `time_to_settlement_frac`, funding_rate_diff e confounded (Zhivkov, MDPI Mathematics 2025/2026)
- Dados ja cached — `get_funding_rate(exchange, symbol)`

### R3-C: OBI (Order Book Imbalance) — CONDICIONAL

> **Status**: Rejeitado nos parametros atuais (OBI decai ~60s, horizonte 4h). **Restaurado condicionalmente**: se interval cair para 1-5s E horizonte encurtar para ≤1h, OBI ganha viabilidade.

- Top SHAP feature em arXiv:2602.00776 (3-second Binance prediction)
- Cont et al. 2014: R²=65% para mid-price contemporaneo
- **Operacionalmente viavel apenas com WebSocket depth** (nao REST polling)
- Formula: `OBI_5L = (sum_bid_5 - sum_ask_5) / (sum_bid_5 + sum_ask_5)`

### SHAP para feature pruning (R11)

TreeSHAP em LightGBM nao transfere perfeitamente para LSTM temporal dynamics (paper 2024 electricity load confirmou). **Usar como screening pre-filter**, depois validar com captum IntegratedGradients no LSTM. Features candidatas a redundancia: `episode_count_30m/2h/8h`, `episode_close_rate_2h` vs `mean_reversion_speed_8h`.

---

# PARTE III — ETA REDESIGN COMPLETO

> **Integrado do ArbML_ETA_Redesign.md + verificacoes v2.0**

## 6. Os 5 Problemas Fundamentais do ETA Atual

### P1 — CRITICO: Output ponto unico sem incerteza
O modelo produz 1 valor: "ETA = 32 minutos". O operador nao sabe se e "quase certamente 32min" ou "pode ser 5min ou 3h". A Uber (DeepETA) descobriu que diferentes use cases precisam de diferentes estimativas pontuais. A DoorDash modela distribuicao completa via Weibull.

### P2 — CRITICO: Quantis empiricos desconectados do modelo
Dashboard mostra Q25/Q50/Q75 de episodios historicos, NAO do modelo. Esses quantis NAO consideram spread atual, velocidade de mudanca, hora do dia, ou par especifico.

### P3 — ALTO: Right-censoring ignorado (95.43% dos dados descartados)
```python
# train_model.py linha 104 — VERIFICADO NO CODIGO
mask = target_class > 0.5  # APENAS 4.57% dos dados treinam ETA
```
Os 2.5M samples negativos sao completamente ignorados para ETA. Mas eles contem informacao: "em 4h nao houve episodio qualificado" e RIGHT-CENSORING (T > 4h, nao T = ∞).

### P4 — ALTO: SmoothL1 e loss inadequada
SmoothL1 e simetrica — trata sub-estimacao e sobre-estimacao igualmente. Na arbitragem, sub-estimar ETA (dizer 10min quando demora 1h) e PIOR que sobre-estimar. O operador que espera 10min e nao ve resultado desiste.

### P5 — MEDIO: Label e tempo ate FECHAR, nao ate ATINGIR pico
```python
y_eta = float(first_qualified.end_ts) - float(current_ts)  # tempo ate FECHAR
# O operador quer: "quanto tempo ate o spread chegar ao pico?"
# peak_ts JA EXISTE na DB (TrackerEpisode.peak_ts)
```

## 7. Literatura de ETA

| Abordagem | Fonte | O que faz | Aplicabilidade |
|-----------|-------|-----------|---------------|
| **Asymmetric Huber** | Uber DeepETA (Wu & Wu 2019, AAAI) | Loss com omega (assimetria) e delta (robustez). Qualquer quantil sem retreinar | **ALTA** — resolve P4 diretamente |
| **Weibull Distribution** | DoorDash (2024) | Prediz (lambda, k) parametricos. Cauda longa natural | MEDIA — futuro (ETA-5) |
| **Pinball Loss** | Koenker & Bassett (1978) | Multi-quantile: L_tau(e) = max(tau*e, (tau-1)*e) | **ALTA** — resolve P1+P2 |
| **Censored QR** | arXiv:2205.13496 (NeurIPS 2022) | Portnoy EM weights para dados censurados | **ALTA** — resolve P3 |
| **WTTE-RNN** | Martinsson (2016) | LSTM → Weibull params. Censoring nativo | MEDIA — alternativa a Weibull |
| **DeepHit** | Lee et al. (2018, AAAI) | Survival analysis com deep learning | MEDIA — referencia |

## 8. Pipeline ETA: 5 Fases

### ETA-1 (R19): Multi-Quantile Head + Pinball Loss (~3h)

**Ficheiros**: `ml_model.py`, `train_model.py`, `ml_analyzer.py`

```python
# ANTES (ml_model.py):
self.eta_out = nn.Sequential(
    nn.Linear(hidden_size, head_hidden),
    nn.SiLU(), nn.Dropout(dropout),
    nn.Linear(head_hidden, 1),  # ← 1 output
)

# DEPOIS:
self.eta_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
self.eta_out = nn.Sequential(
    nn.Linear(hidden_size, head_hidden),
    nn.SiLU(), nn.Dropout(dropout),
    nn.Linear(head_hidden, len(self.eta_quantiles)),  # ← 5 outputs
)
```

**Monotonicity** (softplus-cumsum, verificado matematicamente: softplus(x) > 0 para todo x finito):
```python
# base = raw[0]; deltas = softplus(raw[1:4]); quantiles = cumsum([base, deltas])
```

**Loss**: Pinball loss substitui SmoothL1:
```python
def pinball_loss(pred, target, tau):
    error = target - pred
    return torch.where(error >= 0, tau * error, (tau - 1) * error).mean()
```

**Gate**: AUC ≥ baseline - 0.02. Q50 coverage entre 0.40-0.60. Pinball Q50 < MAE baseline.

### ETA-2 (R20): Censoring Correction com Portnoy (~1d)

> **Verificado**: Fixed-horizon censoring (T=14400s) significa G(t)=1 para todo t<14400 (PMC5568678). IPCW weights = 1 para uncensored. Censored observations usam Portnoy EM redistribution (arXiv:2205.13496 usa Portnoy, NAO IPCW como metodo principal).

```python
def censored_pinball_loss(pred_quantiles, target_eta, target_class, tau_levels, horizon_sec):
    """Portnoy-style censoring for fixed-horizon at 14400s."""
    uncensored = target_class > 0.5
    censored = ~uncensored
    loss = torch.tensor(0.0, device=pred_quantiles.device)

    for k, tau in enumerate(tau_levels):
        q_k = pred_quantiles[:, k]

        # Uncensored: standard pinball (IPCW weight = 1, trivial)
        if uncensored.any():
            err = torch.log1p(target_eta[uncensored]) - q_k[uncensored]
            loss += torch.where(err >= 0, tau * err, (tau - 1) * err).mean()

        # Censored: Portnoy redistribution — penaliza se Q_k < horizon
        if censored.any():
            log_h = math.log1p(horizon_sec)
            below = q_k[censored] < log_h
            if below.any():
                err_c = log_h - q_k[censored][below]
                loss += ((1 - tau) * err_c).mean()

    return loss / len(tau_levels)
```

**Impacto**: Dados para ETA: 4.57% → **100%**. Head passa de data-starved para data-rich.

### ETA-3 (R21): Label peak_ts (~1-2h)

> **Restaurado** (desclassificacao injusta na v2.0 — ETA Redesign doc ja tem plano completo com fallback).

`TrackerEpisode.peak_ts` **JA EXISTE** na DB. `NormalizedEpisode` precisa propagar:

```python
# feature_contracts.py
@dataclass(slots=True)
class NormalizedEpisode:
    end_ts: float
    start_ts: float = 0.0
    peak_ts: float = 0.0  # ← JA EXISTE NA DB
    # ...

# ml_dataset.py — relabel:
# ANTES:  y_eta = end_ts - current_ts (tempo ate close)
# DEPOIS: y_eta = peak_ts - current_ts (tempo ate pico)
# Fallback: se peak_ts == 0 ou peak_ts <= start_ts, usar end_ts
```

**Pre-requisito**: Verificar distribuicao de `(peak_ts - start_ts)/(end_ts - start_ts)` no SQLite. Se concentrada em 0.3-0.7, peak_ts e label estavel. Se bimodal ou >0.9, manter end_ts.

### ETA-4 (R22): Dashboard + Calibracao (~2h)

```
ANTES:   "ETA: 32m"
DEPOIS:  "ETA: 18m / 32m / 61m  (rapido / tipico / lento)"
         ├──────|════════|──────┤
         Q10    Q50      Q90

         Incerteza: ±22min (Q75-Q25)
         Se incerteza < 10min → "confianca alta"
         Se incerteza > 45min → "confianca baixa"
```

Adiciona `eta_uncertainty = Q75 - Q25` ao `compute_signal_score`:
```python
uncertainty_penalty = 1.0 / (1.0 + max(eta_uncertainty_min, 0.0) / 30.0)
score = prob * viability * support * strength * drift * eta_discount * uncertainty_penalty
```

### ETA-5 (futuro): Weibull Head

Substituir quantile head por Weibull parametrica (lambda, k) para censoring nativo e distribuicao continua. Apenas se pinball loss for insuficiente apos ETA-1 a ETA-4.

## 9. Metricas de ETA

| Metrica | Formula | O que mede |
|---------|---------|-----------|
| **Coverage** | `mean(y_true <= Q_tau_pred)` para cada tau | Calibracao por quantil |
| **Pinball Score** | `mean(max(tau*e, (tau-1)*e))` | Qualidade de cada quantil |
| **Interval Width** | `Q90 - Q10` | Sharpness (mais estreito = melhor) |
| **Winkler Score** | `width + penalty_se_fora_do_intervalo` | Combina calibracao + sharpness |

---

# PARTE IV — REGIME DETECTION E THRESHOLDS ADAPTATIVOS

## 10. Episode Detection: Validacao

Thresholds atuais (codigo confirmado):
- Activation: `baseline_median + max(1.0 * MAD, 0.05)` → ~1.48 std
- Release: `baseline_median + max(0.25 * MAD, 0.02)` → ~0.37 std

Paper arXiv:2412.12555: theta_in otimo = 1.42 std (±0.30), theta_out = 0.37 std (±0.13). ArbML activation +4.4% acima (dentro do desvio). Release exato. Nota: conversao MAD*1.4826=std assume normalidade; em fat tails, discrepancia pode ser menor.

**Porem**: thresholds otimizados estaticamente = mesmo retorno que nao-otimizados no test set → regime-adaptation necessaria.

## 11. R4: OU Theta como Feature

RAmmStein (arXiv:2602.19419): theta = primary regime signal. Rolling OLS 1800s:
```
theta = -beta / delta_t (onde X_{t+1}-X_t = alpha + beta*X_t + epsilon)
```

**Bias**: OLS theta tem +22% upward bias em n=1800 (Hudson & Thames). Marriott-Pope corrige parcialmente:
```python
phi_corrected = phi * (n+1)/(n-1) + (1-phi)/(n-1)
theta_corrected = -log(phi_corrected) / dt
```

**Condicao**: Validar offline: (1) theta variance > 15% do range, (2) corr com zscore_vs_8h < 0.5.

## 12. R5: HMM 3-State

> **Restaurado** (desclassificacao injusta — se interval subir para 1-5min, HMM fica viavel. Digital Finance 2024 confirma BIC→3 estados em daily crypto).

- `hmmlearn.GaussianHMM(n_components=3, covariance_type='full')` para fitting
- Custom `OnlineHMMFilter` O(K²) para real-time. Memory: 24 bytes/par = 24KB total
- **Condicao**: Verificar se BIC seleciona 3 estados em dados de 15s. Se nao, usar interval maior ou adiar.

## 13. R6: CUSUM + BH-FDR

CUSUM no z-score residual para structural break detection. BH-FDR para controlar false discovery rate em 10K pares.

**Limitacoes verificadas**: Gaussian ARL nao vale para fat-tails crypto. BH assume independencia violada por pares correlacionados. **Solucao**: Winsorized CUSUM + calibracao empirica de ARL + BH por cluster de exchange.

---

# PARTE V — GRAPH NEURAL NETWORKS

## 14. R16: GNN Layer

### Verificacao: GraphSAGE ≠ MLP em graphs com features heterogeneas

O argumento "WL prova que GNN=MLP em K_6" foi verificado como **framing errado**. WL aplica-se a graph discrimination, nao node representations. Com features heterogeneas (nosso caso: 6 exchanges com embeddings distintos), message passing em K_6 captura informacao relacional que MLP flat nao captura.

### Atlantis Press (2025): F1=0.90 em crypto cross-exchange

GraphSAGE com edge fusion. 78ms inference CPU. 36 edges (6 assets × 5 exchanges). Porem: **classificacao** (arbitragem sim/nao), NAO regressao (spread value).

### Proposta: Ablation MLP vs GAT

Testar ambos e comparar:
- **MLP fusion**: concat 6 LSTM states (384-dim) → MLP 384→128→64
- **GAT 2-layer**: per-symbol 6-node subgraphs com edge features

Se GAT > MLP por >1pp AUC, GNN tem merito. Se nao, MLP e suficiente.

---

# PARTE VI — REINFORCEMENT LEARNING

## 15. Yang & Malik (2024): O Unico Estudo com Numeros em Crypto

| Config | Algoritmo | Retorno | Nota |
|--------|-----------|---------|------|
| Tradicional Gatev | N/A | 8.33% | Baseline |
| RL1 (timing only) | A2C | 9.94% | +1.6pp |
| **RL2 (timing + sizing)** | **A2C** | **31.53%** | **+23.2pp** |
| RL2 | PPO | **-77.81%** | Destruiu valor |
| RL2 | SAC | **-87.12%** | Destruiu valor |

**Licoes**: Sizing > timing (+23pp vs +1.6pp). Apenas A2C funcionou. PPO e SAC destruiram valor.

### Entropy Paradox

SAC ganha em portfolios (alta entropia otima) mas destroi valor em pairs (baixa entropia otima). Principio: **entropia otima da policy determina o algoritmo**.

| Tarefa | Algo Otimo | Contra-indicado |
|--------|-----------|----------------|
| Per-pair sizing | **A2C** | SAC (-87%), PPO (-77%) |
| Cross-pair allocation | **SAC** | — |
| Regime switching | **TimesNet + HMM** | — |

## 16. Caminho Pratico para RL

### Fase 0: Sizing heuristico (R14, ~2h)
```python
def suggested_position_size(signal_score, viability_score, max_viability=5.0):
    raw = min(1.0, max(0.0, viability_score / max_viability))
    return round(raw * min(1.0, signal_score / 0.5), 3)
```

### Fase 1-3: A2C (R17, requer 3000+ sinais resolvidos)
Pre-requisito alto. Interim: supervised GBT em outcome_total.

## 17. R24: Asymmetric Loss (NAO loss aversion RL)

> **Restaurado e corrigido**: O agente PhD confundiu "reward shaping RL" (BBAPT/loss aversion) com "asymmetric loss supervisionada" (Uber DeepETA). Sao coisas diferentes.
>
> - **arXiv:2601.08247 diz "loss aversion no reward RL nao funciona"** — verdade para RL reward shaping
> - **Uber DeepETA Asymmetric Huber Loss funciona** — loss supervisionada, nao RL reward

```python
# Asymmetric Huber Loss (Uber DeepETA, Wu & Wu 2019)
def asymmetric_huber(pred, target, delta=1.0, omega=0.7):
    """
    omega > 0.5: penaliza sub-estimacao mais (operador prefere sobre-estimar)
    delta: robustez a outliers
    """
    error = target - pred
    abs_err = torch.abs(error)
    is_under = (error >= 0).float()
    weight = omega * is_under + (1 - omega) * (1 - is_under)
    loss = torch.where(abs_err < delta, 0.5 * error**2 / delta, abs_err - 0.5 * delta)
    return (weight * loss).mean()
```

**Aplicacao**: Na loss do ETA head (substituir SmoothL1 para o quantil Q50) OU na loss de probabilidade (penalizar FN mais que FP).

---

# PARTE VII — CALIBRACAO E TRAINING PIPELINE

## 18. R7: Signal Ledger Calibration Report

Ledger tem `prob_at_signal` + `outcome_status`. SQL aggregation trivial. Usar SmoothECE (ICML 2024). **NAO conectar a retrain trigger ate N_min=30/bucket.**

## 19. R8: Episode Window 5→7 dias

14 dias causa label starvation (14d sem labels no boundary). 7 dias e sweet spot para crypto 24/7.

## 20. R9: PSI 0.20→0.25

Industry standard. 8h schedule e safety net.

## 21. R10: Platt Stability Check

Guard: se `platt_scale < 0.3` ou `> 3.0`, rejeitar calibracao. Previne falha silenciosa.

---

# PARTE VIII — CLSTM CONTEXT VECTOR FUSION

## 22. R23: 19 Context Features → Modelo

`build_recurring_context_from_episodes()` produz 19+ features usadas no dashboard mas NAO no modelo:

**Continuas (8)**: support_short, support_long, entry_median_2h, exit_median_2h, median_total_spread, episode_density_per_hour, median_episode_duration_sec, entry_exit_coherence_ratio

**Binarias (2)**: entry_coherent, exit_coherent

**One-hot (9)**: context_strength (3), range_status (3), entry_position (3)

### Implementacao

```python
class ContextualSpreadLSTM(nn.Module):
    def __init__(self, input_sz=40, hidden_sz=64, context_sz=19, ...):
        super().__init__()
        self.lstm = nn.LSTM(input_sz, hidden_sz, ...)
        self.temporal_attention = TemporalAttention(hidden_sz)
        self.norm = nn.LayerNorm(hidden_sz)
        self.context_proj = nn.Sequential(
            nn.Linear(context_sz, hidden_sz // 2), nn.SiLU(), nn.Dropout(0.2),
        )
        fused_sz = hidden_sz + hidden_sz // 2
        head_hidden = max(8, fused_sz // 2)
        self.prob_out = nn.Sequential(
            nn.Linear(fused_sz, head_hidden), nn.SiLU(),
            nn.Dropout(0.3), nn.Linear(head_hidden, 1),
        )
        self.eta_out = nn.Sequential(
            nn.Linear(fused_sz, head_hidden), nn.SiLU(),
            nn.Dropout(0.3), nn.Linear(head_hidden, 5),
        )

    def forward(self, x, context_vector):
        out, _ = self.lstm(x)
        pooled = self.temporal_attention(out)
        pooled = self.norm(self.dropout(pooled))
        ctx = self.context_proj(context_vector)
        fused = torch.cat([pooled, ctx], dim=-1)
        return self.prob_out(fused), self.eta_out(fused)
```

**Gate**: PR-AUC ≥ baseline v3. ECE ≤ 0.10. Se CLSTM nao bater v3, descartar.

### Abordagem faseada
Comecar com **5 ETA quantiles empiricos** (empirical_eta_q10..q90) como subset. Se ETA MAE melhora → expandir para 19 completos.

---

# PARTE IX — MAPEAMENTO CODIGO ↔ RECONSTRUCOES

## 23. Inventario de Infraestrutura Existente

| Componente | Ficheiro | O que ja existe |
|------------|----------|----------------|
| Order book | `orderbook.py`, `spread_engine.py` | OrderBookSnapshot com bids/asks multi-nivel. Dados para OBI disponiveis |
| Episodes | `spread_tracker.py` | `compute_closed_episodes()` com MAD thresholds. **peak_ts ja gravado** |
| Recurring context | `spread_tracker.py` | 19+ features. Usado no dashboard, NAO no modelo |
| Features | `feature_contracts.py` | v1(10)→v2(25)→v3(40) com hash versionado |
| Signal ledger | `signal_ledger.py` | SQLite com prob/outcome. Dados para calibracao ja acumulando |
| Auto-retrain | `auto_retrain.py` | Slots 0/8/16h UTC. PSI ja calculado |
| Certification | `training_certification.py` | 12 gates, 204 tests |
| Platt scaling | `train_model.py` | LogisticRegression pos-LSTM |

---

# PARTE X — PAPERS RL (4 novos, 2025-2026)

## 24. BBAPT (Nature Scientific Reports 2026)
3 agentes comportamentais + TimesNet regime switcher. Loss-averse: melhor Sharpe + menor drawdown em bear. Overconfident: maiores retornos em bull + maiores drawdowns. BBAPT integrado: melhor risk-adjusted full cycle.

## 25. Risk-Aware DRL: PPO Collapse (arXiv:2511.11481)
PPO Sharpe 1.41→0.13. 3 failure modes: entropy collapse under clipping, reward misalignment, non-stationarity overfitting.

## 26. SAC vs DDPG Crypto (arXiv:2511.20678)
SAC Sharpe 5x DDPG (0.067 vs 0.013). 3-LSTM + DSR reward. SAC ganha em portfolios via entropy-driven diversification.

## 27. Meta-Analysis 167 Papers (arXiv:2512.10913)
**NOTA**: Scores 0.92/0.85/0.45 sao **rubrica de praticantes** (Table 7), NAO coeficientes de regressao. OLS nao encontrou NENHUM preditor significativo. RF feature importance: complexity score 0.31, algorithm family **0.08**.

"Success in finance is mostly the result of implementation quality, data pre-processing, and domain knowledge instead of algorithmic complexity"

### Transaction Costs Alert
Maioria dos 167 papers usa custos zero. FinRL com 0.1% costs: crypto Sharpe cai para 0.15-0.28. ArbML ja modela costs (cost_estimate_pct=0.30) — vantagem competitiva.

---

# PARTE XI — TABELA MESTRA FINAL (v3.0, R1-R26)

| ID | Acao | Prioridade | Esforco | Condicao |
|----|------|-----------|---------|---------|
| **R10** | Platt stability check | **S** | 30min | Nenhuma |
| **R9** | PSI 0.20→0.25 | **S** | 5min | Nenhuma |
| **R8** | Episode window 5→7d | **S** | 30min | Nenhuma |
| **R7** | Ledger calibration report | **S** | 4h | Nenhuma |
| **R20** | ETA censoring (Portnoy) | **A** | 1d | Nenhuma |
| **R19** | ETA multi-quantile head | **A** | 3h | Requer R20 |
| **R21** | ETA label peak_ts | **A** | 1-2h | Verificar distribuicao no SQLite |
| **R22** | Dashboard ETA bands | **A** | 2h | Requer R19+R20 validados |
| **R3** | Features V4 (volume ratio + funding rate) | **B** | 1d | Nenhuma |
| **R12** | Sequence length ablation | **B** | 4h | PACF primeiro |
| **R14** | Kelly sizing heuristico | **B** | 4h | Backtest |
| **R4** | OU theta feature | **B** | 1d | Validacao offline (bias, variance, corr) |
| **R24** | Asymmetric Huber Loss (ETA) | **B** | 4h | Apos R19-R22 |
| **R23** | CLSTM context fusion | **B** | 8-12h | Requer R19-R22 |
| **R13** | Ensemble LSTM+XGB | **B** | 2d | Refit Platt no combinado |
| **R1** | VSN (micro h=16 ou L1 reg) | **C** | 1-3d | Ablation. Se h=128, VSN reduced viavel |
| **R2** | Attention upgrade (temp ou 2-head) | **C** | 4h | Se h=128, 4-head viavel |
| **R5** | HMM 3-state | **C** | 3d | BIC check em dados reais |
| **R11** | SHAP feature analysis | **C** | 1d | Usar captum primeiro |
| **R15** | Pair embeddings | **C** | 1d | Precisa contrastive fine-tuning |
| **R6** | CUSUM + BH-FDR | **C** | 2d | Winsorized CUSUM + calibracao empirica |
| **R16** | GNN (MLP fusion vs GAT ablation) | **C** | 3d | Ablation |
| **R25** | Offline Sortino entre ciclos | **C** | 4h | Computed batch, nao step reward |
| **R17** | A2C sizing policy | **D** | 3 semanas | 3000+ sinais resolvidos |
| **R18** | TFC contrastive pre-training | **D** | 4 semanas | Se cold-start for problema |
| **R26** | 2-level RL (A2C+SAC) | **F** | — | REJEITADO: zero precedente, latency impossivel |

**Legenda**: S=implementar agora, A=unidade ETA, B=com validacao, C=ablation/condicional, D=adiar, F=rejeitar

---

# PARTE XII — SEQUENCIAMENTO

```
SEMANA 1 (Tier S):
  R10 (Platt check)     — 30min
  R9  (PSI 0.25)        — 5min
  R8  (window 5→7d)     — 30min
  R7  (ledger report)   — 4h

SEMANA 2-3 (Tier A — ETA pipeline):
  R20 (censoring Portnoy) — 1d
  R19 (quantile head)     — 3h
  R21 (peak_ts label)     — 1-2h (se distribuicao ok)
  → Validar coverage rates Q10-Q90
  R22 (dashboard bands)   — 2h

SEMANA 3-4 (Tier B):
  R3  (V4 features)    — 1d
  R24 (Asymmetric Huber)— 4h
  R12 (PACF→ablation)  — 4h
  R14 (Kelly sizing)   — 4h

SEMANA 5-6 (Tier B):
  R4  (OU theta)       — 1d
  R23 (CLSTM)          — 8-12h
  R13 (ensemble)       — 2d

APOS VALIDACAO (Tier C — ablations):
  R1/R2 (VSN/attention, depende de h e seq)
  R5 (HMM, depende de interval)
  R6 (CUSUM, depende de calibracao)
  R16 (MLP vs GAT ablation)

FUTURO (Tier D):
  R17 (A2C, 3000+ sinais)
  R18 (TFC, se cold-start)
```

---

# PARTE XIII — REFERENCIAS

## Papers com resultados quantitativos

1. arXiv:2603.01820 — Benchmark 2026. VLSTM Sharpe 2.40
2. arXiv:2412.09394 — Valeyre & Aboura, LLMs for StatArb. Chronos negativo apos custos
3. arXiv:2511.18578 — Foundation Models. TimesFM -1.47%
4. arXiv:2407.16103 — Yang & Malik, A2C +31.53% vs PPO -77.81%
5. arXiv:2403.12180 — Ning & Lee, EMRT
6. arXiv:2510.11616 — Epstein et al., Attention Factors Sharpe >4
7. arXiv:2602.19419 — RAmmStein, 85% reducao rebalancing
8. arXiv:2508.14784 — Hong & Klabjan, Graph Learning FX +61.89%
9. arXiv:2601.04602 — Fanshawe et al., THGNN Sharpe 1.837
10. arXiv:2602.10711 — FASCL, asset retrieval Sharpe 5.33
11. arXiv:2106.04028 — Guijarro-Ordonez, Deep Learning StatArb
12. arXiv:2412.12555 — Pair Trading Params Optimization
13. Frontiers 2026 — DWE crypto pairs
14. Wiley 2026 — Okasova, feature importance crypto arbitrage
15. PMC 11639137 — KCS-LSTM optimal window 11
16. Nature 2026 — BBAPT, behavioral DRL
17. arXiv:2511.11481 — Risk-Aware DRL, PPO collapse
18. arXiv:2511.20678 — SAC vs DDPG crypto
19. arXiv:2512.10913 — Meta-analysis 167 papers
20. arXiv:2405.16877 — CATS, self-attention unnecessary (NeurIPS 2024)
21. arXiv:2408.10006 — P-sLSTM, geometric ergodicity
22. arXiv:2405.04517 — xLSTM (Beck et al., NeurIPS 2024)
23. arXiv:2205.13496 — Censored QR (NeurIPS 2022)
24. Wu & Wu 2019 — DeepETA, Asymmetric Huber (AAAI)
25. Martinsson 2016 — WTTE-RNN
26. Koenker & Bassett 1978 — Quantile Regression
27. arXiv:2206.08496 — TFC, contrastive time series pre-training

---

*— Fim do Documento (v3.0 — 15/Mar/2026) —*

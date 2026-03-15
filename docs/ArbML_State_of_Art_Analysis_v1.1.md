# ArbML — Analise State-of-the-Art e Plano de Reconstituicao

**15 de Marco de 2026 — v1.1 (revisada)**
**Baseado em 32 recursos do guia + pesquisa complementar profunda (60+ fontes)**
**v1.1: +ETA pipeline, +CLSTM, +mapeamento codigo, +RL status**
**v1.1r: correcoes PPO→A2C, VSN snippet completo, nuance MAD→std, contagem R1-R23**

---

## Sumario Executivo

Esta analise cruza os 32 recursos catalogados no guia com pesquisa adicional de 40+ papers e benchmarks publicados em 2024-2026. A conclusao central e que a **arquitetura base do ArbML (LSTM + temporal attention)** esta empiricamente validada como a escolha correta para spread prediction cross-exchange. No entanto, existem **6 reconstituicoes arquiteturais maiores** (R1-R6) e **17 melhorias complementares** (R7-R23, total 23 acoes) que elevariam o sistema ao estado da arte, cada uma com evidencia quantitativa forte.

**v1.1 adiciona**: pipeline ETA multi-quantile de 4 fases (secoes 19-23), CLSTM context vector fusion com 19 features existentes (secoes 24-25), mapeamento completo codigo↔reconstrucoes (secoes 26-27), e investigacao de Reinforcement Learning com caminho pratico (secoes 28-31). Tabela mestra expandida de R18 → R23.

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
| 9 | Mamba | 0.64 | — | — |
| 10 | **iTransformer** | **0.38** | — | — |

**Fato critico**: O melhor LSTM (VLSTM) tem Sharpe **6.3x** maior que o melhor Transformer puro (iTransformer) no mesmo dataset.

**Explicacao mecanistica do paper**: "Architectures maintaining explicit recurrent state representations consistently outperformed purely attention-based models" porque:
1. LSTM gates adaptam-se a sinais intermitentes em dados ruidosos
2. Self-attention e suscetivel a overfitting em outliers (fat tails)
3. Transformers degradam em mudancas de regime — exatamente o cenario de arbitragem cross-exchange

### Confirmacao: "Transformers versus LSTMs for Electronic Trading" (OpenReview)

Testou ambas arquiteturas em dados de order book de alta frequencia:
- **Previsao de preco absoluto**: Transformers com "only marginal advantage"
- **Sequencias diferenciais (spreads)**: "LSTM-based models demonstrate superior and more consistent performance"

**Spread data e por definicao uma sequencia diferencial** — suporte direto para LSTM.

### O Alerta de Zeng et al. (AAAI 2023)

DLinear (uma unica camada linear) superou Informer, Autoformer e FEDformer em quase todos os benchmarks long-horizon. Implicacao: complexidade Transformer nao confere vantagem automatica quando ruido domina o sinal.

### Foundation Models: Descartados para Producao

**Valeyre & Aboura (arXiv:2412.09394)** — Chronos para Statistical Arbitrage:
- Chronos fine-tuned: Sharpe bruto **3.97** (PCA neutralized)
- CNN-Transformer com **169 parametros**: Sharpe **5.01** (supera Chronos)
- **Apos 3 basis points de slippage: todos os Sharpe de Chronos ficam negativos**

**Re(Visiting) Foundation Models (arXiv:2511.18578)**:
- TimesFM 500M zero-shot: retorno anualizado **-1.47%**
- Chronos retrained: Sharpe 5.42, mas accuracy 51.74% (pouco acima de CatBoost 51.16%)
- "Generic time series pre-training does not directly transfer to financial domains"

**Veredito**: Foundation models (Chronos, TimesFM, Lag-Llama, PatchTST puro) **nao sao viaveis** para ArbML. Custo computacional incompativel com inferencia sub-segundo, e transaction costs eliminam o alpha bruto.

### O que o ArbML tem hoje vs o que deveria ter

| Aspecto | ArbML Atual | State-of-the-Art |
|---------|-------------|------------------|
| Arquitetura base | LSTM unidirecional | **VLSTM** (VSN + LSTM) |
| Attention | Single-head Bahdanau | **Multi-head** (2-4 heads) |
| Feature gating | Nenhum | **Variable Selection Network** |
| Direcao LSTM | Unidirecional | **BiLSTM** (com cuidado no inference) |
| Sequence length | 15 fixo | **10-15 adaptativo** (por half-life) |

---

## 2. RECONSTITUICAO #1: VLSTM (Variable Selection Network + LSTM)

### O que e

VLSTM e a arquitetura vencedora do benchmark 2026. Consiste em:
1. **Variable Selection Network (VSN)**: soft-gating layer que aprende a importancia de cada feature em cada timestep
2. **LSTM encoder**: processa a sequencia filtrada pelo VSN
3. **Temporal attention**: pesa os hidden states (o ArbML ja tem isso)

### Por que e critico para o ArbML

O ArbML tem 40 features (V3 contract). Nem todas sao igualmente relevantes em todos os momentos. O VSN aprende automaticamente:
- "Neste spread, entry_spread e exit_spread importam mais que volume features"
- "Neste regime, as features de 8h importam mais que as de 30m"

### Implementacao pratica

```python
class GatedResidualNetwork(nn.Module):
    """Bloco GRN do Temporal Fusion Transformer (Lim et al., DeepMind)."""
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
    """VSN: aprende soft-gating per-feature per-timestep (core do VLSTM)."""
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.flattened_grn = GatedResidualNetwork(input_size, hidden_size, dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        B, T, F = x.shape
        flat = x.reshape(B * T, F)
        weights = self.softmax(self.output_proj(self.flattened_grn(flat)))
        weights = weights.reshape(B, T, F)
        return weights * x  # element-wise gating (batch, seq_len, input_size)
```

**Impacto estimado**: +0.5-0.9 Sharpe baseado na diferenca VLSTM vs LSTM vanilla no benchmark (2.40 vs 1.48).

---

## 3. RECONSTITUICAO #2: Multi-Head Attention

### Evidencia

O ArbML usa single-head Bahdanau attention. A literatura mostra:
- AUC-ROC improvement de **+6-13%** em financial classification tasks (SenT-In, 2024)
- PR-AUC improvement de **+9%** em media
- PLSTM-TAL: PR-AUC de **0.9455** em S&P500 com attention

Multi-head attention permite ao modelo atender simultaneamente a:
- Head 1: padroes de mean-reversion
- Head 2: sinais de volume/momentum
- Head 3: coerencia entry-exit

### Implementacao

Trocar o attention atual do ArbML (`SpreadSequenceLSTM`) de:
```python
# Atual: single-head
attn_weights = F.softmax(self.attn_layer(torch.tanh(self.attn_proj(lstm_out))), dim=1)
context = (attn_weights * lstm_out).sum(dim=1)
```

Para:
```python
# Proposto: multi-head (4 heads)
self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
context, attn_weights = self.multihead_attn(lstm_out, lstm_out, lstm_out)
context = context.mean(dim=1)  # pool across sequence
```

---

## 4. Sequence Length: 15 e Near-Optimal

### Evidencia empirica

| Estudo | Dominio | Window Otima | Resolucao |
|--------|---------|-------------|-----------|
| KCS-LSTM (PMC 11639137) | Commodity spread | **11** timesteps | Minutos |
| ICS-LSTM (PMC 11784865) | Futures spread | **3** timesteps | HF |
| Pratica geral LSTM stocks | Equities | 60 timesteps | Diario |

**Consenso**: Para spread prediction intraday, windows otimas sao **significativamente mais curtas** que para directional price prediction, porque:
1. Mean reversion e um processo de memoria curta (half-life em minutos a horas)
2. Non-stationarity degrada janelas longas
3. SNR cai com distancia do ponto de decisao

**Recomendacao**: Testar {10, 12, 15, 20}. Hipotese baseada na literatura: 10-15 supera 20+ por non-stationarity. O 15 atual esta no range otimo.

**Abordagem ideal (paper FFT)**: Computar a frequencia dominante da autocorrelacao do spread e setar o lookback para 1-2 ciclos dessa frequencia. Isso tornaria o sequence length data-adaptive.

---

# PARTE II — FEATURES E ENGENHARIA DE SINAIS

## 5. Feature Engineering State-of-the-Art

### O que Okasova et al. (2026, Wiley) descobriu

O paper expandido do IEEE ICBC 2024 testou feature importance para predicao de arbitragem cross-exchange crypto (Binance/Bybit, BTCUSDT/ETHUSDT, 1/5/15min). As **features mais preditivas**:

1. **Spread velocity** (taxa de variacao do spread)
2. **Volume ratio** (volume exchange A / volume exchange B)
3. **Order book imbalance** (OBI = bid_depth - ask_depth / total_depth)

### Comparacao com features do ArbML V3

| Categoria | ArbML V3 tem? | Evidencia de valor |
|-----------|---------------|-------------------|
| Micro (spread, delta, zscore) | Sim (10) | Baseline essencial |
| Multi-escala (30m, 2h, 8h) | Sim (15) | Alta — multiscale e confirmado |
| Exit-aware (exit P10, cycle) | Sim (15) | Alta — unico diferencial |
| **Order book imbalance** | **NAO** | **MUITO ALTA** — top feature em Okasova |
| **Volume ratio** | **NAO** | **ALTA** — segundo top feature |
| **Spread velocity** | Parcial (delta) | ALTA — delta captura parcialmente |
| Funding rate differential | Via pipeline, nao feature | MEDIA — complementar |
| Cross-exchange correlation | NAO | MEDIA — papers GNN confirmam |
| Volatility (realized vol) | NAO | MEDIA — HAR model superior a GARCH para intraday |

### RECONSTITUICAO #3: Adicionar Features Criticas

**Prioridade 1 — Order Book Imbalance (OBI)**:
```python
# O ArbML ja tem OrderBookSnapshot com bids/asks
# Adicionar ao feature_contracts.py:
"buy_obi": (buy_bid_depth - buy_ask_depth) / (buy_bid_depth + buy_ask_depth),
"sell_obi": (sell_bid_depth - sell_ask_depth) / (sell_bid_depth + sell_ask_depth),
"obi_diff": buy_obi - sell_obi,
```

**Prioridade 2 — Volume Ratio**:
```python
"volume_ratio": buy_volume_24h / max(sell_volume_24h, 1e-8),
"log_volume_ratio": log(buy_volume_24h / max(sell_volume_24h, 1e-8)),
```

**Prioridade 3 — Spread Acceleration**:
```python
"spread_acceleration": delta2_entry,  # ja existe como delta2_entry
"spread_jerk": delta3_entry,          # terceira derivada — novo
```

### Sobre dimensionalidade: 40 features e adequado?

A literatura indica que:
- SHAP-based feature selection com gradient boosting funciona bem com 20-150 features
- PCA com 15 componentes como upper bound (Berkhin 2006, ArbitrageLab)
- Para LSTM sequences de 15 timesteps, 40 features gera tensor 15x40=600 — gerenciavel

**Recomendacao**: Expandir para ~46-48 features (adicionando OBI, volume ratio, spread_jerk), entao usar VSN (Reconstituicao #1) para soft-gate automaticamente. Isso e superior a feature selection manual.

---

# PARTE III — REGIME DETECTION E THRESHOLDS ADAPTATIVOS

## 6. O Problema Central: Gates que Quebram em Escala

O ArbML documentou que "gates desenhados para dezenas de pares quebram catastroficamente em ~10.000 pares". Isso e um **problema de regime** — thresholds estaticos nao capturam que o mesmo spread pode ter dinamicas completamente diferentes em regimes diferentes.

### RAmmStein (arXiv:2602.19419, Marco 2026)

**Contexto**: DRL para liquidity provision em DEXes, usando Ornstein-Uhlenbeck process.

**Resultados chave**:
- **85% reducao em rebalancing** vs greedy approaches
- 1.60% net ROI (melhor entre estrategias nao-oniscientes)
- Greedy competitors perderam ate **-8.4%** em gas costs
- Testado com Coinbase 1Hz data, 6.8M trades

**Mecanismo**: O agente aprende a "separar o espaco de estados em regioes de acao e inacao" — thresholds se ajustam implicitamente conforme o DRL aprende boundaries otimos por regime.

### Ning & Lee (Purdue, 2024): EMRT — Empirical Mean Reversion Time

**Inovacao**: Metrica model-free que mede velocidade de mean-reversion:
```
EMRT = (2/N) * Sum(tau_n - tau_{n-1}) para n par
```

**Estado do agente**: Discretizacao de mudancas percentuais recentes em 4^l estados (l=4 → 256 estados), capturando momentum sem dependencia de parametros.

**Resultados**: Sharpe diario ~0.10, retorno cumulativo ~19.2%, max drawdown 0-3%. Consistentemente superior a Distance Method e OU-based.

### RECONSTITUICAO #4: Regime-Aware Adaptive Gates

**O que implementar**:

1. **Online regime detection via HMM** (2-3 estados: low-vol/normal/high-vol):
```python
from hmmlearn import GaussianHMM
# Fit em spread returns rolling
hmm = GaussianHMM(n_components=3, covariance_type="diag")
hmm.fit(spread_returns_rolling)
current_regime = hmm.predict(latest_window)[-1]
```

2. **Thresholds por regime**:
```python
REGIME_THRESHOLDS = {
    0: {"activation": median + 1.5*MAD, "release": median + 0.5*MAD},  # low-vol
    1: {"activation": median + 1.0*MAD, "release": median + 0.25*MAD}, # normal
    2: {"activation": median + 0.5*MAD, "release": median + 0.1*MAD},  # high-vol
}
```

3. **Half-life adaptativo**: Usar EMRT para calibrar o episode detection do `spread_tracker.py`, substituindo os thresholds fixos:
```python
# Atual: activation_threshold = baseline_median + max(1.0 * baseline_mad, 0.05)
# Proposto: adaptar multiplicadores baseado no regime
activation_mult = REGIME_MULTIPLIERS[current_regime]["activation"]
activation_threshold = baseline_median + max(activation_mult * baseline_mad, 0.05)
```

**Impacto**: Resolve diretamente o problema de escalabilidade de gates para 10.000+ pares, porque cada par auto-calibra seus thresholds baseado em seu proprio regime.

---

# PARTE IV — GRAPH NEURAL NETWORKS

## 7. Modelagem de Exchanges como Grafo

### Hong & Klabjan (Northwestern, 2025): FX StatArb com Graph Learning

**Estrutura do grafo**:
- **Nodes**: Moedas (no caso do ArbML: exchanges)
- **Edge features**: Taxas de cambio / spreads
- **Node features**: Taxas de juros / funding rates

**Resultados**:
- Information ratio **61.89% superior** ao benchmark
- Sortino ratio **45.51% superior**
- **Prova formal** de que o metodo satisfaz arbitrage constraints

**Inovacao critica**: Modela explicitamente o **observation-execution time lag** — problema real em crypto arbitrage.

### Fanshawe et al. (2026): THGNN para Correlacao de Equities

- Transformer encoder temporal + edge-aware GAT
- Predicao em Fisher-z space (correlacoes)
- Sharpe de **1.837** out-of-sample (2019-2024)
- Forward-looking baskets que performam especialmente bem em **stress de mercado**

### Como adaptar para o ArbML

**Grafo ArbML**:
```
Nodes = 6 exchanges (mexc, bingx, gate, kucoin, xt, bitget)
Edges = pares de exchanges para cada simbolo
Edge features = [entry_spread, exit_spread, volume_ratio, book_age_diff, obi_diff]
Node features = [avg_latency, circuit_breaker_status, total_symbols, avg_funding_rate]
Temporal = snapshots a cada 15s (tracker interval)
```

**Arquitetura proposta (GNN-LSTM hibrido)**:
1. GAT processa o grafo de exchanges por simbolo → embedding de contexto
2. Embedding de contexto concatenado com features do LSTM sequence
3. LSTM processa sequencia enriquecida → predicao

### RECONSTITUICAO #5: GNN Layer (Medio Prazo)

**Nao e prioridade imediata** — requer mais dados e infra. Mas a modelagem de exchanges como grafo e a generalizacao natural mais poderosa para o ArbML. O principal valor:
- Captura **lead-lag relationships** entre exchanges (quando gate se move, mexc segue em X ms)
- Modela **network effects** (se 3 exchanges convergem, a 4a provavelmente seguira)
- Trata o **time lag** observacao-execucao explicitamente

---

# PARTE V — REINFORCEMENT LEARNING

## 8. Dynamic Scaling: A Proxima Fronteira

### Yang & Malik (2024): RL Pair Trading com BTC

**Setup**: BTC-GBP vs BTC-EUR, 1 minuto, 263.520 observacoes.

**Resultados**:
| Metodo | Retorno Anualizado |
|--------|-------------------|
| Tradicional (Gatev) | 8.33% |
| RL1 (timing only) | 9.94% |
| **RL2 (timing + quantity)** | **31.53%** |

**Mecanismo**: O agente decide nao apenas **quando** entrar/sair, mas **quanto** alocar. Acao continua [-1, +1] representando % do portfolio.

**Implicacao para ArbML**: Hoje o ArbML emite sinais categoricos (EXECUTE/STRONG_EXECUTE/WAIT). Dynamic scaling transformaria em sinais continuos com sizing otimizado.

### Guijarro-Ordonez et al. (Stanford, 2021/2022): Deep Learning StatArb Framework

O framework "3 pilares" que o ArbML implicitamente segue:
1. **Conditional factor model** → spread residual (ArbML: episodios)
2. **Temporal signal extraction** → LSTM (ArbML: SpreadSequenceLSTM)
3. **Optimal trading policy** → dashboard signal (ArbML: SignalScore)

Sharpe out-of-sample "consistently high" em US equities por 24 anos. Evolucao: Epstein et al. 2025 com Attention Factors alcancou **Sharpe > 4** bruto, **2.3 liquido**.

### Amdev-5 Multi-Agent (2026): Referencia de Arquitetura

Arquitetura de 3 agentes + orquestrador:
- **Quant Agent**: cointegration, Z-score, OBI signals
- **Sentiment Agent**: Gemini 2.5 para regime detection via news
- **Risk Agent**: position sizing, stop-loss, drawdown limits
- **Orchestrator**: consensus-based execution

Performance targets: Sharpe > 1.5, win rate > 55%, drawdown < 20%.

### RECONSTITUICAO #6: Signal Score como RL Policy (Longo Prazo)

Substituir o `compute_signal_score()` heuristico atual:
```python
# Atual: formula manual
score = prob * viability * support * strength * drift * eta_discount
```

Por um agente RL (**A2C**, nao PPO) que aprende a politica otima:
```python
# Proposto: A2C com action space continuo
# ATENCAO: PPO produziu -77.81% e SAC -87.12% em pairs trading (Yang 2024).
# Apenas A2C funcionou: menos trades, maior lucro por trade.
state = [prob, net_capture_median, support_2h, support_24h, context_strength,
         eta_minutes, drift_status, regime, spread_velocity, obi_diff]
action = agent.predict(state)  # [-1, +1] sizing + timing
```

**Evidencia**: RL2 (Yang) com A2C dynamic scaling: **31.53%** vs 8.33% tradicional = **3.8x melhoria**. PPO (-77.81%) e SAC (-87.12%) destruiram valor no mesmo teste.

---

# PARTE VI — CALIBRACAO E TRAINING PIPELINE

## 9. Probability Calibration

### Estado atual do ArbML

Platt scaling (logistic regression) pos-LSTM. Isso e adequado mas pode ser melhorado:

**Temperatura scaling** (Guo et al., 2017): Mais simples e frequentemente superior a Platt para redes neurais:
```python
temperature = nn.Parameter(torch.ones(1) * 1.5)
calibrated_prob = torch.sigmoid(logits / temperature)
```

**Isotonic regression**: Nao-parametrica, superior quando ha dados suficientes (>1000 amostras).

**Metricas que o ArbML deve trackear**:
- ECE (Expected Calibration Error) < 0.05
- Brier score (ja trackeado) < 0.25
- Reliability diagram (prob predita vs freq real por bin)

### Concept Drift Detection

O ArbML faz auto-retrain a cada 8h (slots 0h/8h/16h UTC). A literatura sugere:

**PSI (Population Stability Index)** como trigger:
- PSI < 0.10: Sem drift significativo
- 0.10 < PSI < 0.25: Drift moderado — monitorar
- PSI >= 0.25: **Retreinar imediatamente**

```python
def compute_psi(expected, actual, bins=10):
    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

Isso e superior ao schedule fixo de 8h porque retreina **quando necessario**, nao por clock.

## 10. Episode Detection: Validacao

O ArbML usa activation/release thresholds baseados em median + MAD rolling (32 records). Isso esta alinhado com a literatura:

**Validacao pelo paper de parametros de pair trading (arXiv:2412.12555)**:
- Theta_in otimo medio: **1.42 std** (ArbML: 1.0 MAD ≈ 1.48 std, convertendo MAD * 1.4826)
- Theta_out otimo medio: **0.37 std** (ArbML: 0.25 MAD ≈ 0.37 std)

**Os thresholds do ArbML estao quase exatamente nos valores otimos da literatura!**

Porem, o paper tambem encontrou que parametros otimizados produziram **o mesmo retorno cumulativo** (5.2%) que nao-otimizados no test set, sugerindo overfitting. Isso reforça a necessidade de thresholds adaptativos por regime (Reconstituicao #4) ao inves de otimizacao estatica.

## 11. Training Pipeline: Frontiers 2026 DWE

O paper mais recente de crypto pairs trading (Frontiers, Jan 2026) encontrou que **Dynamic Weighted Ensemble (DWE)** — combinando DNN + LSTM com pesos adaptativos — alcanca melhor accuracy que qualquer modelo solo.

**Implicacao para ArbML**: Considerar ensemble leve no auto-retrain:
1. Treinar VLSTM principal (peso 70%)
2. Treinar XGBoost nas mesmas features (peso 30%) — tree-based complementa LSTM em features tabulares (confirmado pelo paper de Hong & Klabjan sobre RNConv)
3. Weighted average da probabilidade calibrada

---

# PARTE VII — CONTRASTIVE LEARNING E EMBEDDINGS

## 12. Future-Aligned Soft Contrastive Learning (FASCL, 2026)

### Detalhes tecnicos completos (arXiv:2602.10711)

**Loss function**: KL divergence entre distribuicoes target e predita:
```
L_SC = (1/B) * Sum(D_KL(p_i || q_i))
```

Onde:
- p_ij: softmax com temperatura sobre correlacoes futuras de retorno (Pearson)
- q_ij: similaridade cosseno dos embeddings
- tau = 0.01 (embedding), tau_t = 0.05 (target)

**Arquitetura**: Patch-based Transformer (64-day windows, 6-channel OHLCV, 14M params)

**Resultados em 4,229 US equities**:
- FRC@K=1: 0.3837 (+12% vs Pearson baseline)
- **Spread trading Sharpe: 5.33** at K=20 (+28% vs melhor alternativa)
- Supera foundation models (MOMENT, Chronos) e SimStock em todas as metricas

**Aplicabilidade ao ArbML**: Nao para pair selection (ArbML opera com todos os simbolos), mas para **priorizar quais simbolos tem maior probabilidade de gerar episodios favoraveis**. Um embedding por simbolo, atualizado periodicamente, alimentaria o SignalScore com dimensao forward-looking.

---

# PARTE VIII — MAPA DE PRIORIDADES

## Reconstituicoes Ordenadas por Impacto/Esforco

| # | Reconstituicao | Impacto Estimado | Esforco | Prioridade |
|---|---------------|-----------------|---------|-----------|
| 1 | **VLSTM** (VSN + LSTM) | +0.5-0.9 Sharpe | Medio | **IMEDIATA** |
| 2 | **Multi-head attention** (4 heads) | +6-13% AUC | Baixo | **IMEDIATA** |
| 3 | **Features: OBI + Volume Ratio** | Top features em Okasova | Baixo | **IMEDIATA** |
| 4 | **Regime-aware adaptive gates** | 85% reducao rebalancing | Alto | **CURTO PRAZO** |
| 5 | **GNN layer** (exchanges como grafo) | +61% info ratio | Muito Alto | **MEDIO PRAZO** |
| 6 | **RL policy** (dynamic scaling) | 3.8x vs heuristico | Muito Alto | **LONGO PRAZO** |

### Complementares (baixo esforco, alto valor):

| Acao | Impacto | Esforco |
|------|---------|---------|
| Trocar Platt por Temperature Scaling | Melhor calibracao | 1 hora |
| Adicionar PSI drift detection | Retreino inteligente | 4 horas |
| Testar sequence lengths {10,12,15,20} | Validacao | 2 horas |
| Ensemble VLSTM + XGBoost | Robustez | 1 dia |

---

# PARTE IX — REFERENCIAS COMPLETAS

## Papers com resultados quantitativos citados

1. **arXiv:2603.01820** — Deep Learning for Financial Time Series Benchmark (2026). VLSTM Sharpe 2.40.
2. **arXiv:2412.09394** — Valeyre & Aboura, LLMs for StatArb. Chronos Sharpe 3.97 bruto, negativo apos custos.
3. **arXiv:2511.18578** — Re(Visiting) Foundation Models. TimesFM -1.47% retorno.
4. **arXiv:2407.16103** — Yang & Malik, RL Dynamic Scaling. 31.53% vs 8.33%.
5. **arXiv:2403.12180** — Ning & Lee, RL com EMRT. Sharpe diario ~0.10.
6. **arXiv:2510.11616** — Epstein et al., Attention Factors. Sharpe > 4, net 2.3.
7. **arXiv:2602.19419** — RAmmStein, Regime Adaptation. 85% reducao rebalancing.
8. **arXiv:2508.14784** — Hong & Klabjan, Graph Learning FX. +61.89% info ratio.
9. **arXiv:2601.04602** — Fanshawe et al., THGNN. Sharpe 1.837.
10. **arXiv:2602.10711** — FASCL, Contrastive Learning. Spread Sharpe 5.33.
11. **arXiv:2106.04028** — Guijarro-Ordonez et al., Deep Learning StatArb.
12. **arXiv:2412.12555** — Pair Trading Parameters Optimization. Theta_in=1.42, theta_out=0.37.
13. **Frontiers 2026** — Tsoku et al., DWE for crypto pairs. Dynamic ensemble.
14. **Quantitative Finance 2023** — Tan & Plataniotis, CNN-LSTM StatArb. 26-year backtest.
15. **MDPI Forecasting 2024** — BiLSTM + Attention + TCN + MADDPG.
16. **Wiley 2026** — Okasova et al., Feature importance: OBI, spread velocity, volume ratio.
17. **PMC 11639137** — KCS-LSTM, optimal window 11 timesteps.
18. **PMC 11784865** — ICS-LSTM, optimal window 3 timesteps.

## Repositorios analisados

1. Hongshen-Yang/pair-trading-envs — RL crypto, Backtrader + Stable-Baselines3
2. seferlab/pairstrading — LSTM vs Transformers comparison
3. fiit-ba/ML-for-arbitrage-in-cryptoexchanges — ML classifiers cross-exchange
4. Amdev-5/crypto-pairs-trading-ai — Multi-agent, OBI, WebSocket sub-100ms
5. bkmulusew/ml_pairs_trading — BiLSTM + Attention + TCN + MADDPG
6. raktim-roychoudhury/pairs_trading — Clustering + DRL, Sharpe 1.15
7. prakhar1602/pair_trading_with_machine_learning — LSTM, +30% vs tradicional
8. achntj/statistical-arbitrage — K-Means + Cointegration + VaR/CVaR
9. ArbitrageLab (Hudson & Thames) — OPTICS/DBSCAN + Cointegration + Half-life

---

---

# PARTE X — ACHADOS PROFUNDOS DAS PESQUISAS COMPLEMENTARES

## 13. Regime Detection: Achados Criticos

### RAmmStein — Detalhes Tecnicos Completos

O paper resolve LPing em Uniswap v3 como **impulse control estocastico** via HJB-QVI. O "Stein Signal" e o theta do OU estimado via OLS rolling em **janela de 1800 segundos (30 min)**:

```
S_{t+1} - S_t = alpha + beta*S_t + epsilon
theta = -beta / delta_t
mu = -alpha / beta
```

**Estado do agente (8 dimensoes)**:
1. Normalized price deviation: S_t/c - 1
2. Distance to range edge: (S_t - c)/(u - c)
3. **Stein Signal (theta)**: truncado para [0, 1]
4. Mean deviation: (mu - S_t)/S_t
5. Normalized sigma: clipped at 0.1
6. Active fraction: cumulative in-range time ratio
7. Rolling volatility: janela 300s, clipped at 0.1
8. In-range indicator: binario

**O que o agente aprendeu**: Todas as 50 rebalancings observadas satisfizeram AMBAS: `|d_edge| > 1` AND `theta < 0.015`. A boundary e nao-linear:
- theta ~ 0 (trending): rebalanceia apenas se `d_edge > 3`
- theta ~ 0.02 (mean-reverting): rebalanceia ja em `d_edge ~ 1.3`

**Theta mediano estimado**: 0.0056, correspondendo a half-life de **~124 segundos** para ETH-USD Coinbase.

### Online Changepoint Detection para 10.000+ Pares

**CUSUM (O(1) por update)**: Ideal para Tier 1 screening:
```
C_t = max(0, C_{t-1} + x_t - kappa)
# Alarme quando C_t > threshold h
```

**MBOC (Markovian BOCPD)**: Combina BOCPD com score-driven time-varying parameters:
```
rho_t = omega + alpha * s_{t-1} + beta * rho_{t-1}
```
Supera significativamente o BOCPD classico em dados financeiros fat-tailed.

**Arquitetura de gates em 3 tiers para 10.000 pares**:
- **Tier 1** (todos N pares): CUSUM no z-score residual — O(N) por tick
- **Tier 2** (~1% dos pares): BOCPD-MBOC para confirmar continuidade de regime
- **Tier 3** (<0.1% dos pares): OU parameter refresh completo + EMRT + ML scoring

### Correcao de Multiplas Hipoteses: BH-FDR

Quando 10.000 pares sao testados simultaneamente com threshold no percentil 97.5, esperam-se **250 falsos positivos** por puro acaso. Solucao: **Benjamini-Hochberg FDR control** com alpha = 0.05, limitando falsos a 5% dos ativados independente de N.

---

## 14. RL e Features: Achados Criticos

### Yang & Malik — Numeros Completos

| Config | Algoritmo | Retorno Anualizado |
|--------|-----------|-------------------|
| Tradicional Gatev | N/A | 8.33% |
| RL1 (timing only) | A2C | 9.94% |
| **RL2 (timing + sizing)** | **A2C** | **31.53%** |
| RL2 | PPO | **-77.81%** |
| RL2 | SAC | **-87.12%** |

**Fato critico**: PPO e SAC **destruiram valor** com dynamic sizing. Apenas A2C funcionou. Razao: SAC maximiza entropia (over-trades), PPO e conservador demais para rewards fat-tailed de spread trading. A2C faz menos trades, com maior lucro por trade.

### Epstein et al. — Detalhes da Arquitetura Attention Factors

- **LongConv** (nao LSTM) como sequence model: convolution kernels com decaimento geometrico, O(T log T) via FFT
- Lookback de **30 dias**, 32 hidden dims, 1 layer
- **Joint optimization** (fator + policy) e **decisivo**: Sharpe cai de 3.97 para 1.50 se fatores e policy sao treinados separadamente
- Remover dados de preco passado colapsa performance (3.97 → 1.50)
- **30+ weak factors > poucos strong factors**: diversidade de features vence selecao agressiva

### Feature Recommendations Consolidadas

**Features a ADICIONAR (alta evidencia)**:
| Feature | Tier | Fonte na Infra |
|---------|------|---------------|
| `entry_bidask_width` | Tier 1 | `orderbook.snapshot()`: asks[0][0] - bids[0][0] |
| `exit_bidask_width` | Tier 1 | Idem para exit exchange |
| `entry_top_depth_usd` | Tier 2 | `_top_level_qty()` * best_price, log-scaled |
| `exit_top_depth_usd` | Tier 2 | Idem |
| `funding_rate_diff` | Tier 3 | `market_data.get_funding_rate()` |

**Features a SCRUTINIZAR via SHAP** (possivel redundancia):
- `episode_count_30m/2h/8h` (3 features derivadas da mesma serie)
- `episode_close_rate_2h` vs `mean_reversion_speed_8h` (alta correlacao provavel)
- `max_entry_30m/2h/8h` (parcialmente redundante com position_in_range + mean + std)

**Target V4**: 44-45 features (adicionar 5), depois pruning via SHAP para 35-38 finais.

### Abordagem Intermediaria para Dynamic Sizing (sem RL)

Antes de implementar RL completo, mapear `viability_score` monotonicamente para position size:
```python
# Sem ML adicional — captura parte significativa do valor
position_pct = min(1.0, max(0.0, viability_score / viability_max))
```
Isso aproxima o RL2 de Yang sem infraestrutura adicional.

---

## 15. Calibracao e Training: Achados Criticos

### FASCL — Detalhes Tecnicos Completos

**Loss**: KL divergence com soft targets baseados em correlacao futura:
```
L_SC = (1/B) * Sum(D_KL(p_i || q_i))
p_ij = softmax(C_ij / tau_t)    # tau_t = 0.05
q_ij = softmax(cos_sim(z_i, z_j) / tau)  # tau = 0.01
```

**Arquitetura**: Patch Transformer — Conv1d patch embedding (4 timesteps → 1 token), 8 blocks, 384 dims, 8 heads, ~14M params.

**Resultados em 4,229 equities**:
- Spread trading Sharpe: **5.33** at K=20 (+28% vs melhor alternativa)
- Supera MOMENT, Chronos, SimStock em todas as metricas
- Tempo de embedding para 16,900 amostras: **2.78 segundos**

**Aplicacao pratica para ArbML**: Extrair embeddings por par do LSTM encoder (camadas antes de prob_out/eta_out). Pares com cosine similarity > 0.70 sao "co-behaving" → tratar como estatisticamente dependentes no sizing.

### Signal Ledger como Oracle de Calibracao

O `signal_ledger.py` ja captura `prob_at_signal` e resolve outcomes. Usar para calibracao empirica:

```python
def ledger_calibration_report():
    for bucket in [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.0)]:
        signals = query_resolved_in_bucket(bucket)
        empirical_hit_rate = sum(s.outcome == 'hit') / len(signals)
        # Se modelo prediz P=0.75 mas hit rate e 0.55 → recalibrar
```

Trigger: ECE > 0.15 com N >= 25 em qualquer bucket acima de 0.70 → recalibrar Platt.

### Episodios: Thresholds Validados

Paper de otimizacao de parametros (arXiv:2412.12555):
- theta_in otimo: **1.42 std** (desvio: 0.30)
- theta_out otimo: **0.37 std** (desvio: 0.13)

**Conversao para MAD**: 1.42 std ÷ 1.4826 = **0.958 MAD** (ArbML usa 1.0, +4.4% acima) | 0.37 std ÷ 1.4826 = **0.250 MAD** (ArbML usa 0.25, exato)

**Os thresholds do ArbML estao muito proximos dos otimos**: activation 4.4% acima (dentro do std=0.30 do paper), release exato. Porem, o mesmo paper encontrou que thresholds otimizados estaticamente produziram o **mesmo retorno** (5.2%) que nao-otimizados no test set — evidencia de que otimizacao estatica e insuficiente, motivando regime-adaptation (R4-R5).

### PSI e Drift: Melhoria sobre Schedule Fixo

PSI ja implementado no ArbML. Thresholds da literatura:
- PSI < 0.10: OK
- 0.10-0.25: Monitorar
- PSI >= 0.25: **Retreinar imediatamente**

Adicionar: **recalibration-only path** (refit Platt no signal ledger, sem retreino completo) quando ECE > 0.12 com 50+ resolved signals.

### Episode Window: Expandir de 5 para 10-14 dias

O paper coreano de triple barrier labeling (2025) encontrou que janelas de 29 dias otimizam calibracao. O ArbML usa 5 dias (`label_episode_window_days`), que e vulneravel a eventos extremos recentes. **Expandir para 10-14 dias** estabiliza os percentis de labeling.

---

## 16. GNN: Achados Criticos

### Atlantis Press (2025): GNN para Arbitragem Crypto Cross-Exchange

**O unico paper que resolve exatamente o problema do ArbML com GNN.**

- GraphSAGE com edge fusion customizado
- F1 = **0.90**, Precision = 0.89, Recall = 0.92, AUC = 0.94
- **78ms inference por grafo no CPU**
- 36 edges (6 assets x 5 exchanges)

### Integracao Pratica: Per-Symbol Subgraphs

```
Para cada simbolo (ex: BTC):
  Nodes = 6 (um por exchange)
  Edges = 30 (todas combinacoes bidirecionais)
  Node features = [mid_price, volume, OBI, funding_rate, book_age]
  Edge features = [spread, volume_ratio, fee_diff, latency_diff]

Total: 1000 subgraphs de 6 nodes cada
Batch PyG: ~5-30ms no GPU | ~50-200ms no CPU
```

### Integracao Sequential (Menor Disrupcao)

```
LSTM existente por par → embedding h_i (64-dim)
                              ↓
Build PyG graph com h_i como node features
                              ↓
GraphSAGE (2 layers, 64 hidden)
                              ↓
Enhanced embeddings h_i'
                              ↓
Edge prediction head → spread previsto
```

Loss: Huber no spread fee-adjusted. Backpropagation flui de volta para o LSTM.

### Risco de Over-Smoothing

Com 3+ layers GNN, nodes distantes influenciam demais, apagando sinais par-especificos do LSTM. **Comecar com 2 layers**. Verificar que o GNN nao degrada a performance do LSTM solo — se degradar, o grafo esta mal construido.

---

# PARTE XII — PIPELINE ETA MULTI-QUANTILE (ETA-1 a ETA-4)

> **Nota**: Esta secao cobre um pipeline de 4 fases ja planejado no roadmap que o documento original nao endereçava. O eta_out atual e single-point — limitacao seria para sinais operacionais.

## 19. Estado Atual do ETA

O `ml_model.py` tem um single-point ETA head:

```python
# ml_model.py linhas 165-169 (ATUAL)
self.eta_out = nn.Sequential(
    nn.Linear(self.hidden_size, head_hidden),
    nn.SiLU(),
    nn.Dropout(self.dropout_rate),
    nn.Linear(head_hidden, 1),  # ← SINGLE POINT
)
```

O `train_model.py` usa SmoothL1Loss apenas nos samples positivos (y_class > 0.5), o que significa que **apenas ~4.57% dos dados treinam o ETA head** — todos os samples censurados (sem episodio no horizonte) sao descartados.

```python
# train_model.py linhas 97-108 (ATUAL)
def _eta_loss(criterion, eta_raw, target_eta_seconds, target_class):
    mask = target_class > 0.5
    if not torch.any(mask):
        return eta_raw.new_tensor(0.0)
    target = torch.log1p(target_eta_seconds[mask])
    pred = eta_raw.squeeze(1)[mask]
    return criterion(pred, target)
```

O dashboard mostra "ETA: 32m" — um unico numero sem incerteza.

## 20. ETA-1: Multi-Quantile Head + Pinball Loss

**Ficheiros alvo**: `ml_model.py`, `train_model.py`, `ml_analyzer.py`

**O que muda**:

```python
# ml_model.py (PROPOSTO)
self.eta_out = nn.Sequential(
    nn.Linear(self.hidden_size, head_hidden),
    nn.SiLU(),
    nn.Dropout(self.dropout_rate),
    nn.Linear(head_hidden, 5),  # ← 5 QUANTIS: Q10/Q25/Q50/Q75/Q90
)

# Monotonicity via softplus deltas:
# base = raw[0]
# deltas = softplus(raw[1:4])
# quantiles = cumsum([base, deltas])
```

**Loss**: Pinball loss substitui SmoothL1:
```python
def pinball_loss(pred, target, tau):
    error = target - pred
    return torch.max(tau * error, (tau - 1) * error).mean()

# Total = media dos 5 quantis (tau = 0.10, 0.25, 0.50, 0.75, 0.90)
# Peso mantido: loss_prob + 0.25 * loss_eta
```

**Gate de aceitacao**:
- AUC >= baseline v3 - 0.02 (ETA loss nao deve degradar prob head)
- Q50 coverage entre 0.40 e 0.60
- Pinball score Q50 < MAE do baseline

**Esforco**: ~3h | **Dados**: mesmos

## 21. ETA-2: Censoring Correction

**Ficheiro alvo**: `train_model.py`

Samples censurados (y_class=0) passam a contribuir para ETA loss. Loss censurada: penalizar se predicao < horizonte (sub-estimacao):

```python
def censored_eta_loss(pred_q50, horizon_sec, target_class):
    censored_mask = target_class < 0.5
    if not torch.any(censored_mask):
        return pred_q50.new_tensor(0.0)
    pred = pred_q50[censored_mask]
    h = torch.full_like(pred, math.log1p(horizon_sec))
    return torch.clamp(h - pred, min=0).pow(2).mean()
```

**Impacto**: Usa **100% dos samples** em vez de ~4.57%. O ETA head passa de data-starved para data-rich.

**Esforco**: ~2h | **Dados**: mesmos

## 22. ETA-3: Label peak_ts

**Ficheiro alvo**: `ml_dataset.py`, `feature_contracts.py`

O `TrackerEpisode` no `spread_tracker.py` **ja grava peak_ts** (timestamp do pico do spread). O `NormalizedEpisode` no `feature_contracts.py` precisa propagar esse campo:

```python
# feature_contracts.py (PROPOSTO)
@dataclass(slots=True)
class NormalizedEpisode:
    end_ts: float
    start_ts: float = 0.0
    exit_spread: float = 0.0
    duration_sec: float = 0.0
    peak_entry_spread: float = 0.0
    total_spread: float = 0.0
    is_closed: bool = True
    peak_ts: float = 0.0  # ← NOVO: ja existe na DB

# ml_dataset.py — na relabel vectorizada:
# ANTES:  y_eta = end_ts - current_ts (tempo ate close)
# DEPOIS: y_eta = peak_ts - current_ts (tempo ate pico)
# Fallback: se peak_ts == 0 ou peak_ts <= start_ts, usar end_ts
```

**Impacto**: ETA passa de "tempo ate o episodio fechar" para "tempo ate o spread atingir o pico" — que e o que o operador realmente quer saber.

**Esforco**: ~1-2h | **Dados**: peak_ts JA EXISTE na DB

## 23. ETA-4: Dashboard ETA + Calibracao

**Ficheiros alvo**: `ml_analyzer.py`, `signal_ledger.py`, `soak_runbook.py`

Dashboard antes/depois:
```
ANTES:   "ETA: 32m"
DEPOIS:  "ETA: 18m / 32m / 61m (±21m)"
         Q10=18m  Q25=25m  Q50=32m  Q75=46m  Q90=61m
```

Adiciona `eta_uncertainty = Q75 - Q25` ao `compute_signal_score`:
```python
# ml_analyzer.py (PROPOSTO)
uncertainty_penalty = 1.0 / (1.0 + max(eta_uncertainty_min, 0.0) / 30.0)
score = prob * viability * support * strength * drift * eta_discount * uncertainty_penalty
```

**Esforco**: ~2h

---

# PARTE XIII — CLSTM CONTEXT VECTOR FUSION (Phase 4)

> **Nota**: O `build_recurring_context_from_episodes` em `spread_tracker.py` ja calcula 19 features de contexto que alimentam o dashboard mas NAO entram no modelo LSTM. Esta secao cobre a fusao dessas features no modelo.

## 24. Context Vector Existente (19 features)

O recurring_context ja e calculado e usado no `ml_analyzer.py` para o dashboard. Estas sao as 19 features disponiveis:

**Continuas (8)**:
- `support_short` — episodios nas ultimas 2h
- `support_long` — episodios nas ultimas 24h
- `entry_median_2h` — mediana de entry spread 2h
- `exit_median_2h` — mediana de exit spread 2h
- `median_total_spread` — mediana do total_spread
- `episode_density_per_hour` — frequencia de episodios
- `median_episode_duration_sec` — duracao mediana
- `entry_exit_coherence_ratio` — coerencia entry↔exit

**Binarias (2)**:
- `entry_coherent` — faixa de entrada estavel
- `exit_coherent` — faixa de saida estavel

**One-hot (9)**:
- `context_strength` (3): weak / normal / strong
- `range_status` (3): below / inside / above
- `entry_position` (3): low / mid / high

## 25. CLSTM: Fusao Pratica

**Ficheiros alvo**: `ml_model.py`, `ml_dataset.py`, `train_model.py`, `ml_analyzer.py`

```python
# ml_model.py (PROPOSTO)
class ContextualSpreadLSTM(nn.Module):
    def __init__(self, input_sz=40, hidden_sz=64, context_sz=19, ...):
        super().__init__()
        # Mesma arquitetura LSTM + attention existente
        self.lstm = nn.LSTM(input_sz, hidden_sz, ...)
        self.temporal_attention = TemporalAttention(hidden_sz)
        self.norm = nn.LayerNorm(hidden_sz)

        # NOVO: context projection
        self.context_proj = nn.Sequential(
            nn.Linear(context_sz, hidden_sz // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        # Heads recebem pooled + context
        fused_sz = hidden_sz + hidden_sz // 2
        head_hidden = max(8, fused_sz // 2)
        self.prob_out = nn.Sequential(
            nn.Linear(fused_sz, head_hidden), nn.SiLU(),
            nn.Dropout(0.3), nn.Linear(head_hidden, 1),
        )
        self.eta_out = nn.Sequential(
            nn.Linear(fused_sz, head_hidden), nn.SiLU(),
            nn.Dropout(0.3), nn.Linear(head_hidden, 5),  # multi-quantile
        )

    def forward(self, x, context_vector):
        out, _ = self.lstm(x)
        pooled = self.temporal_attention(out)
        pooled = self.norm(self.dropout(pooled))
        ctx = self.context_proj(context_vector)
        fused = torch.cat([pooled, ctx], dim=-1)
        return self.prob_out(fused), self.eta_out(fused)
```

**Mudancas no pipeline**:
- `ml_dataset.py`: `DatasetBundle` ganha campo `context_vectors: torch.Tensor` (N x 19)
- `train_model.py`: Forward pass inclui context_vector; config: `model_type="lstm"` vs `"clstm"`
- `ml_analyzer.py`: Na inferencia, extrair recurring_context → vetor 19-dim → passar ao modelo

**Gate de aceitacao**:
- PR-AUC >= baseline v3 (benchmark e v3 SEM context, NAO v2)
- ECE <= 0.10 (recalibrar Platt)
- Se CLSTM nao bater v3, **descartar**

**Esforco**: ~8-12h (3-4 sessoes) | **Dados**: mesmos (context ja existe)

---

# PARTE XIV — MAPEAMENTO CODIGO ↔ RECONSTRUCOES

> **Nota**: Cada R1-R18 mapeado para o que JA EXISTE no repo e o que precisa mudar.

## 26. Inventario de Infraestrutura Existente

| Componente | Ficheiro | O que ja existe |
|------------|----------|----------------|
| Order book completo | `orderbook.py`, `spread_engine.py` | `OrderBookSnapshot` com bids/asks multi-nivel. `_top_level_qty()` extrai quantidade. **Dados para OBI ja disponiveis, so nao sao features.** |
| Episode detection | `spread_tracker.py` | `compute_closed_episodes()` com activation/release thresholds adaptativos por par (median + MAD rolling 32 records). **peak_ts ja gravado.** |
| Recurring context | `spread_tracker.py` | `build_recurring_context_from_episodes()` retorna dict com 19+ chaves. **Usado no dashboard, NAO no modelo.** |
| Feature contracts | `feature_contracts.py` | v1(10) → v2(25) → v3(40) versionados com hash. `_RollingEntryWindow`, `_RollingExitWindow`, `_RollingEpisodeStats` com janelas 30m/2h/8h. |
| Signal ledger | `signal_ledger.py` | SQLite com `prob_at_signal`, `outcome_status`, cooldown. **Dados para calibracao empirica ja acumulando.** |
| Auto-retrain | `auto_retrain.py` | Slots 0h/8h/16h UTC, snapshots, manifests, state machine. **PSI ja calculado em `runtime_audit.py`.** |
| Training certification | `training_certification.py` | 6 camadas: ingest → treino. 204 tests. Gate-based. |
| Platt scaling | `train_model.py` | `LogisticRegression` pos-LSTM. `platt_scale` e `platt_bias` no metadata. |

## 27. Mapeamento R1-R18 → Codigo

| ID | O que ja existe no repo | O que falta implementar |
|----|------------------------|------------------------|
| R1 (VLSTM) | `SpreadSequenceLSTM` em `ml_model.py` com LSTM + `TemporalAttention` | Adicionar `VariableSelectionNetwork` (novo) + `GatedResidualNetwork` (novo) pre-LSTM |
| R2 (Multi-head) | `TemporalAttention` (single-head, `nn.Linear→Tanh→Linear→Softmax`) | Trocar por `nn.MultiheadAttention(num_heads=4)` |
| R3 (OBI features) | `OrderBookSnapshot` com bids/asks completos, `_top_level_qty()` | Calcular OBI, bidask_width, depth_ratio em `spread_engine.py`, propagar para `feature_contracts.py` v4 |
| R4 (OU theta) | `_RollingEntryWindow` com regression slope (OLS) em janela rolling | Adicionar estimacao OU (theta = -beta/dt) na janela de 30min, expor como feature |
| R5 (HMM regime) | Nenhum | Novo modulo `regime_detector.py` com `hmmlearn.GaussianHMM(n_components=3)` |
| R6 (CUSUM + BH-FDR) | `runtime_audit.py` com PSI | Adicionar CUSUM online O(1) no `spread_tracker.py`, BH-FDR no `ml_analyzer.py` |
| R7 (Ledger calibration) | `signal_ledger.py` com `prob_at_signal` e outcomes | Adicionar `ledger_calibration_report()` com ECE por bucket e reliability diagram |
| R8 (Episode window) | `label_episode_window_days` no config de treino | Mudar default de 5 para 10-14 |
| R9 (PSI retrain) | `_build_snapshot_psi_summary` em `runtime_audit.py` | Adicionar PSI threshold (>=0.25 → retreinar) em `auto_retrain.py` decision logic |
| R10 (Platt check) | `LogisticRegression` Platt em `train_model.py` | Adicionar guard: se coef < 0.1 ou intercept > 5.0, rejeitar calibracao |
| R11 (SHAP) | Nenhum | Novo script com LightGBM proxy + `shap.TreeExplainer` |
| R12 (Seq length) | `sequence_length=15` no config | Ablation loop testando {10,12,15,20} |
| R13 (Ensemble) | `SpreadSequenceLSTM` unico | Treinar XGBoost paralelo, weighted average prob |
| R14 (Sizing) | `compute_signal_score()` retorna score continuo | Mapear `viability_score → position_pct` (0.0 a 1.0) |
| R15 (Embeddings) | Pooled output (64-dim) do LSTM antes dos heads | Extrair e salvar embeddings; cosine similarity entre pares |
| R16 (GNN) | `spread_engine.py` com topology de exchanges | Novo modulo PyG com GraphSAGE, LSTM embeddings como node features |
| R17 (A2C sizing) | `compute_signal_score()` heuristico | Novo modulo RL com `stable-baselines3.A2C`, reward = net_capture |
| R18 (FASCL) | Nenhum | Novo modulo de contrastive learning com Patch Transformer |

---

# PARTE XV — REINFORCEMENT LEARNING: STATUS DA INVESTIGACAO

> **Nota**: Busca nas conversas anteriores do projeto confirma que **RL nunca foi discutido, prototipado ou implementado no ArbML**. Todo o historico de desenvolvimento focou em LSTM supervisionado, feature engineering, episode detection, soak testing e infraestrutura. Esta secao consolida o estado da investigacao e define o caminho pratico.

## 28. O que ja sabemos (da pesquisa, NAO do codigo)

A pesquisa compilada no guia de 32 recursos + este documento identificou 4 resultados concretos sobre RL para pairs trading:

### Yang & Malik (2024) — O unico estudo com numeros reais em crypto

| Config | Algoritmo | Retorno Anualizado | Resultado |
|--------|-----------|-------------------|-----------|
| Tradicional Gatev | N/A | 8.33% | Baseline |
| RL1 (timing only) | A2C | 9.94% | +1.6pp |
| **RL2 (timing + sizing)** | **A2C** | **31.53%** | **+23.2pp** |
| RL2 (timing + sizing) | PPO | **-77.81%** | Destruiu valor |
| RL2 (timing + sizing) | SAC | **-87.12%** | Destruiu valor |

**Licoes criticas**:
1. O valor do RL esta no **sizing** (quanto alocar), nao no timing (quando entrar/sair) — timing only ganhou apenas 1.6pp
2. **Apenas A2C funcionou**. PPO e SAC destruiram capital. A2C faz menos trades com maior lucro por trade
3. SAC maximiza entropia (over-trades em spreads fat-tailed), PPO e conservador demais

### RAmmStein (2026) — RL para threshold adaptation

- O agente aprendeu boundaries nao-lineares: theta ~ 0 (trending) → rebalanceia apenas se `d_edge > 3`; theta ~ 0.02 (mean-reverting) → rebalanceia ja em `d_edge ~ 1.3`
- **85% reducao em rebalancing** vs greedy
- **Nao e RL para trading**, e RL para **quando agir** — conceito transferivel ao ArbML para adaptive gates

### Ning & Lee (2024) — RL com EMRT

- Estado inclui tendencias recentes (nao apenas desvio da media)
- Reward customizado para mean-reversion: penaliza acoes contra reversao
- Sharpe diario ~0.10, max drawdown 0-3%

### Epstein et al. (2025) — Joint optimization e decisiva

- Treinar fatores e policy separadamente: Sharpe 1.50
- Treinar fatores e policy **conjuntamente**: Sharpe 3.97
- **Se o ArbML implementar RL, a integracao com o LSTM deve ser end-to-end, nao como modulo separado**

## 29. Por que RL NAO foi investigado ate agora

O caminho de desenvolvimento do ArbML priorizou (corretamente) em ordem:
1. Pipeline de dados confiavel (WebSocket → SQLite → certificacao)
2. Modelo supervisionado funcional (LSTM + attention + FocalLoss)
3. Feature engineering progressivo (v1→v2→v3)
4. Infraestrutura de producao (soak test, auto-retrain, signal ledger)

RL requer que os passos 1-4 estejam solidos porque:
- O **environment** de RL depende de dados confiaveis (step 1)
- O **state space** depende de features boas (step 3)
- A **reward function** depende de captura/custo corretos (step 2-4)
- A **avaliacao** depende de infraestrutura de backtesting (step 4)

**O ArbML esta agora no ponto certo para comecar a investigar RL** — mas como P3, nao P0.

## 30. Caminho Pratico para RL no ArbML

### Fase 0: Aproximacao sem RL (R14, ~2h)

Antes de investir em infraestrutura RL, mapear `viability_score` monotonicamente para position size:

```python
# ml_analyzer.py — SEM ML adicional
def suggested_position_size(signal_score, viability_score, max_viability=5.0):
    # Captura parte significativa do valor de RL2 sem complexidade
    raw = min(1.0, max(0.0, viability_score / max_viability))
    # Modular por confianca do sinal
    return round(raw * min(1.0, signal_score / 0.5), 3)
```

**Metrica de sucesso**: Se o sizing heuristico correlaciona com outcomes (sinais com size > 0.7 tem hit rate > sinais com size < 0.3), entao RL pode refinar. Se nao correlaciona, o problema nao e sizing — e feature/modelo.

### Fase 1: Environment Design (R17 prep, ~1 semana)

```python
# Nao implementar ainda — design doc para validacao
class ArbMLTradingEnv(gym.Env):
    """
    State (14-dim):
      - prob (1): inversion_probability do LSTM
      - eta_quantiles (5): Q10/Q25/Q50/Q75/Q90 do ETA head
      - net_capture_policies (3): shallow/median/deep net_capture
      - support (2): support_2h, support_24h
      - context_strength (1): encoded 0/0.3/0.7/1.0
      - regime (1): encoded 0/1/2
      - spread_velocity (1): delta_entry normalizado

    Action: Continuo [-1, +1]
      - 0 = nao operar
      - (0, 1] = sizing da posicao (long spread)
      - [-1, 0) = sizing inverso (short spread, se aplicavel)

    Reward:
      - Se acao > 0 e episodio fecha com hit:
        reward = net_capture * |acao| - custo * |acao|
      - Se acao > 0 e episodio nao fecha (timeout):
        reward = -custo * |acao|
      - Se acao == 0:
        reward = 0 (sem penalidade por inacao)
      - Penalidade por overtrade: -0.01 * |acao| a cada step sem episodio

    Episode: 1 signal ledger entry (signal emitido → outcome resolvido)
    """
```

**Decisao critica**: Usar **A2C** (nao PPO, nao SAC) baseado nos resultados de Yang.

### Fase 2: Treino offline com Signal Ledger (R17, ~3 semanas)

**Pre-requisito**: 200+ sinais resolvidos no signal_ledger (Fase 2.9 do roadmap)

O signal ledger acumula exatamente o que o RL precisa:
- State: todas as features do sinal no momento da emissao
- Action: EXECUTE/STRONG_EXECUTE (converter para sizing continuo)
- Reward: outcome_hit/miss com net_capture real

Treinar A2C offline com historical transitions do ledger. Avaliar contra o `compute_signal_score` heuristico.

### Fase 3: Online A2C (futuro distante)

Substituir `compute_signal_score()` heuristico por `agent.predict(state)` com safety constraints:
- Hard cap de posicao maxima
- Never override WAIT signals do LSTM (RL so decide sizing de sinais que ja passaram o gate)
- Rollback automatico se Sharpe cai abaixo de threshold

## 31. Decisao: RL vs Heuristico — Quando Escalar

| Condicao | Acao |
|----------|------|
| < 200 sinais resolvidos | Usar sizing heuristico (R14) |
| 200+ sinais, heuristico correlaciona com outcomes | Treinar A2C offline, comparar |
| A2C offline supera heuristico por > 10% em hit rate | Deploy A2C com safety constraints |
| A2C offline NAO supera | Manter heuristico, revisitar com mais dados |

---

# PARTE XVI — TABELA MESTRA DE ACOES (ATUALIZADA v1.1)

| ID | Acao | Arquivo Alvo | Impacto | Esforco | Prioridade |
|----|------|-------------|---------|---------|-----------|
| R1 | VSN pre-LSTM (VLSTM) | `ml_model.py` | +0.9 Sharpe | 3 dias | **P0** |
| R2 | Multi-head attention (4 heads) | `ml_model.py` | +6-13% AUC | 4 horas | **P0** |
| R3 | Features: OBI + bidask_width + depth | `feature_contracts.py`, `spread_engine.py` | Top features Okasova | 1 dia | **P0** |
| R4 | OU theta como feature (rolling 30min) | `feature_contracts.py`, `spread_tracker.py` | Regime signal decisivo | 2 dias | **P1** |
| R5 | 3-regime HMM (GMM-3state) | Novo modulo | Adaptive gates | 3 dias | **P1** |
| R6 | CUSUM Tier 1 + BH-FDR | `spread_tracker.py` | Escala 10K pares | 2 dias | **P1** |
| R7 | Signal ledger calibration report | `signal_ledger.py` | Detect miscalibration | 4 horas | **P1** |
| R8 | Episode window 5→10-14 dias | `train_model.py` config | Labeling estavel | 30 min | **P1** |
| R9 | PSI-based retrain trigger | `auto_retrain.py` | Retreino inteligente | 4 horas | **P1** |
| R10 | Platt stability check | `train_model.py` | Prevent bad calibration | 1 hora | **P1** |
| R11 | SHAP feature analysis (LightGBM proxy) | Novo script | Pruning informado | 1 dia | **P2** |
| R12 | Sequence length ablation {10,12,15,20} | `train_model.py` | Validacao | 2 horas | **P2** |
| R13 | Ensemble VLSTM + XGBoost | `ml_analyzer.py` | Robustez | 2 dias | **P2** |
| R14 | Viability→sizing mapping | `ml_analyzer.py` | Approx dynamic scaling | 2 horas | **P2** |
| R15 | Pair embeddings do LSTM encoder | `ml_analyzer.py` | Co-behavior detection | 1 dia | **P2** |
| **R19** | **ETA-1: Multi-Quantile Head** | `ml_model.py`, `train_model.py` | **ETA probabilistico** | **3 horas** | **P1** |
| **R20** | **ETA-2: Censoring Correction** | `train_model.py` | **20x mais dados p/ ETA** | **2 horas** | **P1** |
| **R21** | **ETA-3: Label peak_ts** | `ml_dataset.py`, `feature_contracts.py` | **ETA ate pico (nao close)** | **1-2 horas** | **P1** |
| **R22** | **ETA-4: Dashboard ETA multi-quantile** | `ml_analyzer.py`, `signal_ledger.py` | **"18m / 32m / 61m (±21m)"** | **2 horas** | **P1** |
| **R23** | **CLSTM Context Fusion (19 features)** | `ml_model.py`, `ml_dataset.py`, `train_model.py` | **Sinal contextualizado** | **8-12 horas** | **P2** |
| R16 | GNN GraphSAGE layer | Novo modulo | Cross-exchange context | 2 semanas | **P3** |
| R17 | A2C sizing policy (requer 200+ sinais) | Novo modulo | 3.8x vs heuristico | 3 semanas | **P3** |
| R18 | FASCL pre-training | Novo modulo | Cold-start + priorization | 4 semanas | **P3** |

### Sequenciamento ETA (R19→R22 sao sequenciais):

```
R19 (ETA-1) → R20 (ETA-2) → R21 (ETA-3) → R22 (ETA-4)
     3h            2h           1-2h           2h
     ↓
  Pode comecar em paralelo com R1-R3
```

### Sequenciamento RL (R14→R17):

```
R14 (sizing heuristico, 2h)
     ↓
  Acumular 200+ sinais no ledger
     ↓
R17 (A2C offline, 3 semanas)
     ↓
  Se A2C > heuristico por >10%: deploy
  Se nao: manter heuristico
```

---

---

# PARTE XVII — NOVOS PAPERS: RL AVANCADO PARA FINANCAS (4 PAPERS, JAN-DEZ 2025-2026)

> **v1.1r2**: Adiciona 4 novos papers sobre DRL em financas com analise PhD-level, novas reconstituicoes R24-R26, e implicacoes cruzadas com o ArbML.

---

## 32. Paper 1: BBAPT — Behavioral DRL com Loss Aversion e Overconfidence

**Ref**: Charkhestani & Esfahanipour, Scientific Reports (Nature), 28 Jan 2026. DOI: 10.1038/s41598-026-35902-x

### Framework BBAPT (Behavioral Bias-Based Algorithmic Portfolio Trading)

Arquitetura de 3 agentes comportamentais + regime detector:

```
TimesNet (regime forecasting)
     ↓
  Bull → Overconfident Agent (amplifica posicoes)
  Bear → Loss-Averse Agent (reduz exposicao, lambda=2.25)
  Neutral → Neutral DRL Agent (baseline)
     ↓
  Actor-Critic (PPO/A2C) gera direcao
  Behavioral layer ajusta SIZING (nao direcao)
```

### Prospect Theory na Reward Function

Baseado em Kahneman & Tversky (1979):
```
Gains:  U(x) = (x - x_0)^alpha          # alpha = 0.88
Losses: U(x) = -lambda * (x_0 - x)^alpha  # lambda = 2.25
```

Na pratica, BBAPT simplifica para:
```python
def behavioral_reward(r_base, regime, lambda_loss=2.25):
    if regime == "bear" and r_base < theta_loss:
        return lambda_loss * r_base  # Amplifica dor de perdas
    elif regime == "bull":
        return r_base * overconfidence_amp  # Amplifica ganhos
    else:
        return r_base  # Neutro
```

### Overconfidence: NAO e Exploracao — e Position Sizing

Descoberta crucial: overconfidence NAO modifica epsilon, NAO altera reward, NAO muda acoes disponiveis. Apenas **amplifica a magnitude dos pesos** do portfolio. Isso mantem a dinamica de aprendizado limpa enquanto impoe heterogeneidade comportamental.

Efeito: agente overconfident produz **maiores retornos cumulativos** em bull markets (2020-2021 crypto), mas tambem **maiores drawdowns**.

### Resultados

| Agente | Sharpe | Volatilidade | Max Drawdown | Melhor em |
|--------|--------|-------------|-------------|-----------|
| Loss-Averse | **Mais alto** | **Mais baixa** | **Menor** | Bear/volatil |
| Overconfident | Medio | Mais alta | Maior | Bull |
| Neutro (DRL puro) | Medio | Media | Medio | Sideways |
| **BBAPT integrado** | **Melhor risk-adjusted** | Controlada | Controlado | **Full cycle** |
| Markowitz | Inferior | — | — | Baseline |
| Equal-weight | Inferior | — | — | Baseline |

**Dataset**: Crypto (TRX, DOT, MATIC, ETC, etc.) Jan 2018 - Jun 2024 + DJIA Jan 2008 - Jun 2024. Walk-forward sem look-ahead.

### Implicacao para ArbML: RECONSTITUICAO R24

**Loss aversion para spreads e um gap de pesquisa inexplorado.** Nenhum paper de pairs trading ou arbitragem incorpora prospect theory. A logica e compelling:

- **Loss aversion em spreads**: Quando o spread esta se alargando contra a posicao (unrealized loss crescendo), o agente loss-averse **reduz tamanho da posicao** automaticamente, evitando a armadilha de "esperar mean reversion" quando cointegration quebrou
- **Overconfidence containment**: Suprimir overconfidence durante alta volatilidade de spread previne over-sizing em spreads que parecem mean-reverting mas estao em structural break
- **Regime detection para spreads**: TimesNet no spread dynamics (nao em preco absoluto) identifica oportunidades high-confidence vs risco de break

```python
# R24: Behavioral reward shaping para ArbML
def spread_behavioral_reward(net_capture, regime, unrealized_pnl):
    """Reward function inspirada em BBAPT para spread trading."""
    if regime == "high_vol" and unrealized_pnl < -cost_estimate:
        # Loss aversion: amplifica penalidade de spreads que alargam contra
        return 2.25 * net_capture  # lambda = 2.25 (K&T empirical)
    elif regime == "low_vol" and net_capture > 0:
        # Regime estavel com capture positivo: nao amplificar (evitar overconfidence)
        return net_capture
    else:
        return net_capture
```

---

## 33. Paper 2: Risk-Aware DRL — O Colapso do PPO

**Ref**: Lwele et al., arXiv:2511.11481, Nov 2025.

### O Colapso Documentado

| Metrica | Antes do Treino | Apos Treino |
|---------|----------------|-------------|
| Retorno Anualizado | 51.1% | **2.10%** |
| Volatilidade | 34.9% | 16.32% |
| Sharpe Ratio | 1.41 | **0.13** |

PPO convergiu para **alocacoes ultra-conservadoras** (quase cash) destruindo retornos enquanto estabilizava volatilidade. O agente aprendeu que "nao fazer nada" minimiza penalidades de risco.

### Por que PPO Colapsou — 3 Failure Modes Compounding

**Failure Mode 1 — Premature entropy collapse under clipping**:
PPO limita updates a ratio r_t(theta) ∈ [1-epsilon, 1+epsilon] (epsilon=0.2). Em financas com rewards esparsos e ruidosos, a estimativa do gradient tem alta variancia. PPO sistematicamente subestima acoes lucrativas e superestima acoes seguras (low-return, low-variance). Cada update empurra a policy para acoes de menor variancia — efeito **ratchet** que compoe iterativamente ate a entropia colapsar para zero.

**Failure Mode 2 — Reward structure misalignment**:
O reward `R = ln(A . Y - mu*sum|delta_w|)` e log-return per-step, NAO Sharpe-shaped. Quando PPO aprende a minimizar variancia, alocacoes de baixa variancia (cash, stablecoins) recebem rewards estaveis que acumulam mais confiavelmente que alocacoes voláteis mas lucrativas. O agente converge para "maximize expected sum of small stable returns" — racional sob a reward structure, mas Sharpe ~0 em producao.

**Failure Mode 3 — Non-stationarity overfitting**:
Dataset 2010-2024. Se o train/test boundary corta em ~2020-2022, a policy aprendeu alocacoes conservadoras que funcionaram no bear 2018-2019 mas falham no bull 2020-2023. A non-stationarity quebra a politica treinada.

### Cruzamento com Yang & Malik (2024)

PPO tambem falhou catastroficamente em pairs trading (-77.81%). O padrao e consistente:

| Paper | Contexto | PPO Result | Razao |
|-------|----------|------------|-------|
| Yang 2024 | Crypto pairs 1min | **-77.81%** | Over-trades, nao captura fat-tail rewards |
| Lwele 2025 | Equity portfolio monthly | **Sharpe 0.13** | Over-conservative, converge para cash |
| SAC/DDPG paper 2025 | Crypto portfolio daily | N/A (nao testou PPO) | — |

**Conclusao reforçada**: PPO e **inadequado para financas** onde rewards sao fat-tailed e oportunidades sao raras. A2C e a escolha correta para pairs trading (Yang). SAC funciona para portfolios (nao pairs).

### Solucoes Propostas pelo Paper

1. **Hybrid risk-aware reward**: Combinar Sharpe maximization com CVaR penalty (nao MaxDD diretamente)
2. **Walk-forward validation robusta** em vez de single train/test split
3. **Reward shaping progressivo**: Comecar com retorno puro, gradualmente adicionar risk constraints

---

## 34. Paper 3: SAC vs DDPG em Crypto — O Paradoxo SAC

**Ref**: arXiv:2511.20678, Nov 2025.

### Resultados Completos (BTC/ETH/LTC/DOGE, daily, 2016-2024)

| Metrica | DDPG | **SAC** | MPT Baseline |
|---------|------|---------|-------------|
| Portfolio Final | 1.97 | **2.76** | 2.04 |
| Sharpe | 0.013 | **0.067** | 0.027 |
| Sortino | 0.019 | **0.109** | 0.038 |
| Max Drawdown | -0.682 | **-0.409** | -0.720 |
| CVaR 95% | -0.060 | **-0.058** | -0.058 |

SAC supera DDPG por **5x em Sharpe** e reduz MaxDD em **40%**.

### Arquitetura: 3x LSTM + OHLCV

- **State**: Rolling window W=50 dias de OHLCV log-diff + portfolio weights atuais
- **Feature extraction**: 3 LSTM layers sobre os W=50 timesteps → latent representation
- **Action**: Softmax portfolio weights (long-only, sum=1)
- **Reward**: Differential Sharpe Ratio:
```
DSR_t = (B_{t-1} * delta_A_t - 0.5 * A_{t-1} * delta_B_t) / (B_{t-1} - A_{t-1}^2)^(3/2)
# Onde A_t = running mean return, B_t = running mean return^2
```
- **Transaction costs**: `R_net = R - c * sum(|w_t - w_{t-1}|)`

### O Paradoxo: SAC Ganha em Portfolios mas Perde em Pairs Trading

| Contexto | SAC | A2C | Por que? |
|----------|-----|-----|---------|
| **Portfolio** (diversificado, daily) | **Ganha** (Sharpe 0.067) | Nao testado | Entropy regularization explora alocacoes diversas |
| **Pairs** (single spread, 1min) | **-87.12%** (Yang 2024) | **+31.53%** | SAC over-trades em ambiente de acao unica |

**O "Entropy Paradox" (conceito dos agentes de pesquisa)**: SAC maximiza entropia (variabilidade de acoes). A entropia e simultaneamente benefica E prejudicial, dependendo da estrutura da tarefa:

- **Portfolio weights** tem alta entropia otima (muitas combinacoes de pesos sao validas) → SAC's entropy bonus alinha com a tarefa → **SAC ganha**
- **Spread direction** tem baixa entropia otima (ha uma resposta claramente certa: esperar threshold, entrar na direcao do sinal) → SAC's entropy bonus luta contra a estrutura → **SAC destroi valor** (2,798 trades vs 229 do A2C em Yang 2024)

Dados de trade count confirmam: SAC executou **12x mais trades** que A2C, cada um na direcao errada contra o sinal de mean-reversion. Transaction costs de cada trade errado acumulam-se catastroficamente.

**Mecanismos especificos de SAC em crypto (paper 2511.20678)**:
1. **Regime robustness**: Entropy mantem distribuicao nao-degenerada → recupera rapido de mudancas de regime (ex: crash FTX Nov 2022). DDPG deterministico nao tem escape mechanism → MaxDD -0.682 vs SAC -0.409
2. **Diversificacao implicita**: Entropy encoraja alocacoes nao-zero em todos ativos → hedge natural contra eventos idiossincraticos
3. **Resistencia a ruido**: Policy estocastica media sobre padroes ruidosos do OHLCV → regularizacao implicita (como dropout)

**Temperatura alpha auto-adaptativa (SAC)**:
```python
# Dual gradient descent de Haarnoja et al. (2018)
# H_target = -dim(Action) (heuristica padrao)
log_alpha_new = log_alpha - lr_alpha * (H(policy) - H_target)
# H(policy) > H_target → alpha diminui (menos exploracao)
# H(policy) < H_target → alpha aumenta (mais exploracao)
# Cria feedback self-correcting por regime:
#   Alta volatilidade → policy naturalmente estocastica → alpha cai
#   Baixa volatilidade/trend → policy converge → alpha sobe
```

**Implicacao para ArbML**: Como ArbML opera em spreads individuais (nao portfolios diversificados), SAC e contra-indicado para per-pair sizing. A2C permanece a escolha correta. Para um futuro cross-pair allocator (R26), SAC e viavel porque a alocacao de capital entre pares tem alta entropia otima (analogia com portfolio weights).

---

## 35. Paper 4: Meta-Analysis de 167 Papers — O Veredito do Campo

**Ref**: arXiv:2512.10913, Dec 2025 (submitted to Management Science). 167 papers, 2017-2025.

### A Descoberta Central: Implementation Quality >> Algorithm Choice

**NOTA METODOLOGICA CRITICA**: Os scores 0.92/0.85/0.45 abaixo sao de uma **rubrica de julgamento de praticantes** (Table 7 do paper), NAO coeficientes de regressao. A analise OLS do paper nao encontrou NENHUM preditor estatisticamente significativo. A validacao usa um dataset sintetico calibrado a padroes da literatura, nao extracoes diretas dos 167 papers.

**Table 7 (Rubrica de Praticantes)**:

| Fator | Score (rubrica) | Interpretacao |
|-------|----------------|---------------|
| **Implementation quality** | **0.92** (Critical) | Feature engineering, preprocessing, walk-forward validation |
| **Domain expertise** | **0.85** (High) | Microestrutura de mercado, custos, regime awareness |
| Data quality | High | Survivorship bias, imputation, normalization |
| Algorithm choice | **0.45** (Moderate) | DQN vs PG: diferenca minima (p=0.640) |

**Random Forest Feature Importance (Figura 4, analise quantitativa)**:

| Preditor | Importance | Nota |
|----------|-----------|------|
| Complexity score (impl. sophistication) | **0.31** | Fator #1 real |
| Market making domain indicator | 0.28 | — |
| Sample size / data quality | 0.19 | — |
| Algorithm family | **0.08** | Quase irrelevante |

**OLS Regressions (Figura 2 — NENHUM significativo)**:

| Preditor | Slope | P-valor |
|----------|-------|---------|
| Feature dimensions | 0.171 | 0.499 |
| Number of assets | 0.010 | 0.362 |
| Training period | 0.023 | 0.591 |
| Recession inclusion | — | 0.604 |
| Algorithm family | — | **0.640** |

**"Success in finance is mostly the result of implementation quality, data pre-processing, and domain knowledge instead of algorithmic complexity"**

### Taxonomia por Dominio

| Dominio | Papers | Algos Dominantes | RL Premium |
|---------|--------|-----------------|-----------|
| Market Making | — | DDPG, TD3, SAC | **0.488** (maior) |
| **Crypto Trading** | — | PPO, CNN-RL, Transformer-RL | **0.375** |
| Portfolio Optimization | 45 | DDPG, DQN | Moderate |
| Algorithmic Trading | 62 | DQN, PPO, SAC | — |

### Trend: Pure RL → Hybrids

```
2020: Pure RL 85%, Hybrids 15%
2025: Pure RL 58%, Hybrids 42%
```

Hybrids mais bem-sucedidos:
- **LSTM-DDPG**: Moderate-high performance
- **CNN-PPO**: +17.9% improvement (pattern recognition)
- **Attention-DDPG**: +16-18% em crypto
- **Transformer-RL**: Significant gains em sequence modeling

### Failure Modes Documentados

| Failure Mode | Causa | Mitigacao |
|-------------|-------|-----------|
| Overfitting a regimes historicos | Distribution shift | **Alta** (regularization, early stopping) |
| State space explosion | Features sem domain knowledge | **Alta** (feature selection) |
| Sample inefficiency | Custo de exploracao real | **Moderada** (model-based, transfer) |
| Liquidity violations | Market impact subestimado | **Moderada** (volume-aware) |
| Black-box decisions | Falta de explicabilidade | **Baixa** (SHAP, attention) |
| Adversarial attacks | Defensive training inadequado | **Baixa** |

### 3 Gaps Criticos Identificados

1. **Non-stationarity**: Nao ha solucao convergida para mercados que mudam estruturalmente
2. **Exploration cost**: Equilibrar aprendizado com perdas reais e immaturo
3. **Regulatory-Performance tradeoff**: Explicabilidade vs performance nao resolvido

### Alerta Critico: Transaction Costs Omitidos na Maioria dos Papers

O meta-analysis documenta que **a maioria dos 167 papers usa custos zero ou simplificados**. Impacto real:
- FinRL Contests com 0.1% costs: crypto Sharpe cai para **0.15-0.28** (vs 0.375 RL premium reportado)
- SAC/DDPG com costs: Sharpe de **0.067** (vs claims de Sharpe >1 em papers sem costs)
- China futures pairs: retorno mensal cai **40%** (25bp→15bp) quando costs sao incluidos

**O ArbML ja modela costs** via `cost_estimate_pct=0.30` e `label_cost_floor_pct=0.50` — isso e uma vantagem sobre a maioria da literatura. Mas qualquer comparacao com Sharpe ratios de papers academicos deve descontar **pelo menos 50-70%** para custos realistas.

### Implicacao para ArbML: VALIDACAO FORTE

O meta-analysis **valida a estrategia do ArbML** de priorizar:
1. Qualidade de implementacao (pipeline de dados, certificacao, soak test) sobre complexidade algoritmica
2. Domain expertise (feature contracts V1→V2→V3, episode detection) sobre busca de algoritmo perfeito
3. Hybrids (LSTM + attention, nao pure RL) como caminho correto
4. A2C para spread trading individual (nao SAC, nao PPO)

---

## 36. Sintese Cruzada: O que os 4 Papers Mudam no ArbML

### Consenso 2025 por Tarefa (167 papers + 4 novos)

| Tarefa | Algo Otimo | Contra-indicado | Evidencia |
|--------|-----------|----------------|-----------|
| Portfolio rebalancing (N ativos, daily) | **SAC** (entropy=diversificacao) | PPO (over-conservative) | 2511.20678, 2511.11481 |
| Pairs/stat arb (1 spread, mean-reversion) | **A2C** (low entropy) | SAC (-87%), PPO (-77%) | Yang 2024 |
| Cross-exchange arb (discrete: enter/exit/hold) | **DQN/DDQN** | SAC (over-trades) | Meta-167, FinRL |
| Cross-exchange arb (continuous sizing) | **A2C** ou **TD3** | SAC, PPO | Yang 2024, meta-167 |
| Market making | **DDPG/TD3** | — | Meta-167: premium 0.488 |
| Regime switching | **TimesNet + HMM** | — | BBAPT Nature 2026 |

**Principio geral**: Entropia otima da policy determina o algoritmo. Alta entropia (portfolio) → SAC. Baixa entropia (threshold bet) → A2C/DQN.

### Nova Hierarquia de Algoritmos RL para ArbML

| Nivel | Tarefa | Algoritmo | Evidencia |
|-------|--------|-----------|-----------|
| **Per-pair sizing** | Quanto alocar em 1 par | **A2C** | Yang 2024: +31.53%. SAC: -87%. PPO: -77% |
| **Cross-pair allocation** | Capital entre pares | **SAC** (possivel) | Paper crypto portfolio: SAC Sharpe 5x DDPG |
| **Regime switching** | Quando mudar modo | **TimesNet + HMM** | BBAPT: melhor risk-adjusted full cycle |
| **Risk overlay** | Stop-loss adaptativo | **Loss aversion lambda** | BBAPT: menor drawdown com lambda=2.25 |

### Novas Reconstituicoes (R24-R26)

| ID | Acao | Impacto | Esforco | Prioridade |
|----|------|---------|---------|-----------|
| **R24** | **Loss aversion reward shaping** (lambda=2.25 em bear/high-vol) | Menor drawdown, protecao contra structural breaks | 1 dia | **P2** |
| **R25** | **Differential Sharpe Ratio como reward** (substituir raw return) | Risk-adjusted training nativo | 4 horas | **P2** |
| **R26** | **Per-pair A2C + cross-pair SAC** (2-level RL hierarchy) | Sizing otimizado + allocation otimizada | 4 semanas | **P3** |

### R24: Loss Aversion Reward para ArbML (Detalhe)

```python
# Integrar em ml_analyzer.py ou novo modulo rl_reward.py
LAMBDA_LOSS = 2.25  # Kahneman-Tversky empirical

def prospect_theory_reward(net_capture_pct, regime, unrealized_spread_pnl):
    """
    Reward shaping com loss aversion regime-conditional.
    Baseado em BBAPT (Nature 2026) adaptado para spread trading.
    """
    if net_capture_pct >= 0:
        # Ganho: retorno normal (alpha=0.88 optional)
        return net_capture_pct
    else:
        # Perda: amplificar por lambda se em regime adverso
        if regime in ("high_vol", "trending"):
            return LAMBDA_LOSS * net_capture_pct  # 2.25x a dor
        else:
            return 1.5 * net_capture_pct  # Lambda reduzido em regime normal
```

### R25: Differential Sharpe Ratio Reward (Detalhe)

O paper SAC/DDPG crypto usa DSR como reward — superior a log-return puro porque penaliza volatilidade inline:

```python
def differential_sharpe_ratio(return_t, running_mean_A, running_mean_B, eta=0.01):
    """
    Moody & Saffell (2001). Reward que otimiza Sharpe incrementalmente.
    A_t = running mean of returns
    B_t = running mean of returns^2
    """
    delta_A = return_t - running_mean_A
    delta_B = return_t**2 - running_mean_B
    denom = (running_mean_B - running_mean_A**2) ** 1.5
    if abs(denom) < 1e-10:
        return 0.0
    dsr = (running_mean_B * delta_A - 0.5 * running_mean_A * delta_B) / denom
    # Update running stats
    new_A = running_mean_A + eta * delta_A
    new_B = running_mean_B + eta * delta_B
    return dsr, new_A, new_B
```

---

# PARTE XVIII — TABELA MESTRA FINAL (v1.1r2, R1-R26)

| ID | Acao | Arquivo Alvo | Impacto | Esforco | Prioridade |
|----|------|-------------|---------|---------|-----------|
| R1 | VSN pre-LSTM (VLSTM) | `ml_model.py` | +0.9 Sharpe | 3 dias | **P0** |
| R2 | Multi-head attention (4 heads) | `ml_model.py` | +6-13% AUC | 4 horas | **P0** |
| R3 | Features: OBI + bidask_width + depth | `feature_contracts.py`, `spread_engine.py` | Top features | 1 dia | **P0** |
| R4 | OU theta como feature (rolling 30min) | `feature_contracts.py`, `spread_tracker.py` | Regime signal | 2 dias | **P1** |
| R5 | 3-regime HMM (GMM-3state) | Novo modulo | Adaptive gates | 3 dias | **P1** |
| R6 | CUSUM Tier 1 + BH-FDR | `spread_tracker.py` | Escala 10K | 2 dias | **P1** |
| R7 | Signal ledger calibration report | `signal_ledger.py` | Miscalibration | 4 horas | **P1** |
| R8 | Episode window 5→10-14 dias | `train_model.py` config | Labeling | 30 min | **P1** |
| R9 | PSI-based retrain trigger | `auto_retrain.py` | Retreino smart | 4 horas | **P1** |
| R10 | Platt stability check | `train_model.py` | Bad calibration | 1 hora | **P1** |
| R11 | SHAP feature analysis | Novo script | Pruning | 1 dia | **P2** |
| R12 | Sequence length ablation | `train_model.py` | Validacao | 2 horas | **P2** |
| R13 | Ensemble VLSTM + XGBoost | `ml_analyzer.py` | Robustez | 2 dias | **P2** |
| R14 | Viability→sizing mapping | `ml_analyzer.py` | Approx scaling | 2 horas | **P2** |
| R15 | Pair embeddings do LSTM encoder | `ml_analyzer.py` | Co-behavior | 1 dia | **P2** |
| R19 | ETA-1: Multi-Quantile Head | `ml_model.py`, `train_model.py` | ETA probabilistico | 3 horas | **P1** |
| R20 | ETA-2: Censoring Correction | `train_model.py` | 20x mais dados ETA | 2 horas | **P1** |
| R21 | ETA-3: Label peak_ts | `ml_dataset.py` | ETA ate pico | 1-2 horas | **P1** |
| R22 | ETA-4: Dashboard ETA multi-quantile | `ml_analyzer.py` | UX informativo | 2 horas | **P1** |
| R23 | CLSTM Context Fusion (19 features) | `ml_model.py`, `ml_dataset.py` | Sinal context | 8-12 horas | **P2** |
| **R24** | **Loss aversion reward (lambda=2.25)** | `rl_reward.py` (novo) | **Menor drawdown** | **1 dia** | **P2** |
| **R25** | **Differential Sharpe Ratio reward** | `rl_reward.py` (novo) | **Risk-adjusted training** | **4 horas** | **P2** |
| R16 | GNN GraphSAGE layer | Novo modulo | Cross-exchange | 2 semanas | **P3** |
| R17 | A2C sizing policy | Novo modulo | 3.8x vs heuristico | 3 semanas | **P3** |
| R18 | FASCL pre-training | Novo modulo | Cold-start | 4 semanas | **P3** |
| **R26** | **Per-pair A2C + cross-pair SAC** | Novo modulo | **Allocation otima** | **4 semanas** | **P3** |

### Mapa de Dependencias Atualizado

```
IMEDIATO (P0):     R1 + R2 + R3 (podem ser paralelos)
CURTO PRAZO (P1):  R4→R5→R6 (regime pipeline)
                   R7→R9→R10 (calibracao pipeline)
                   R8 (standalone, 30min)
                   R19→R20→R21→R22 (ETA pipeline, sequencial)
MEDIO PRAZO (P2):  R11→R12 (SHAP → ablation)
                   R13 + R14 + R15 (ensemble + sizing + embeddings)
                   R23 (CLSTM, requer R19-R22)
                   R24 + R25 (reward shaping, requer R17 prep)
LONGO PRAZO (P3):  R16 (GNN), R17 (A2C), R18 (FASCL), R26 (2-level RL)
```

---

## 37. Novas Referencias (Papers 32-35)

33. **Nature Scientific Reports 2026** — Charkhestani & Esfahanipour, BBAPT. Loss aversion lambda=2.25 + overconfidence + TimesNet regime switching.
34. **arXiv:2511.11481** — Lwele et al., Risk-Aware DRL. PPO colapsou de Sharpe 1.41→0.13 com risk constraints.
35. **arXiv:2511.20678** — SAC vs DDPG crypto portfolios. SAC Sharpe 5x DDPG, MaxDD 40% menor. 3-LSTM + DSR reward.
36. **arXiv:2512.10913** — Meta-analysis 167 papers. Implementation quality (0.92) >> algorithm choice (0.45). Hybrids 15%→42%.

### Literatura Complementar Identificada

37. **arXiv:2410.02605** — Policy Gradients for Cumulative Prospect Theory in RL. Gradient theorem completo para CPT.
38. **arXiv:2601.08247** — Cognitive Biases in RL for Finance. Lambda > 2.5 degrada performance. Naive loss aversion insuficiente.
39. **Moody & Saffell 2001** — Differential Sharpe Ratio. Reward function original para trading RL.
40. **arXiv:2403.12180** — Ning & Lee, RL StatArb. 22.0% return vs 5.7% tradicional, MaxDD -3.3% vs -37.5%.

---

*— Fim do Documento (v1.1r2 — 15/Mar/2026) —*
*v1.1: +ETA pipeline, +CLSTM, +mapeamento codigo, +RL status*
*v1.1r: PPO→A2C, VSN completo, MAD→std nuance*
*v1.1r2: +4 novos papers (BBAPT/Risk-Aware/SAC-DDPG/Meta-167), +R24-R26, paradoxo SAC explicado, meta-analysis validation*
*v1.1r2 deep: Entropy Paradox formalizado, PPO 3-failure-modes detalhado, meta-analysis scores corrigidos (rubrica nao regressao), consenso 2025 por tarefa, transaction costs alert, temperatura alpha auto-adaptativa*

# ArbML — Post-V3 Roadmap

**Pré-requisito geral:** Baseline v3 treinado (40 features + ETA single-point) com AUC e ETA MAE medidos.

---

## Sequência

```
BASELINE V3 (treinar agora)
  │
  ├─→ Fase ETA-1: Multi-quantile head + Pinball loss
  │     │
  │     └─→ Fase ETA-2: Censoring correction
  │           │
  │           └─→ Fase ETA-3: Label peak_ts (requer dados novos)
  │                 │
  │                 └─→ Fase ETA-4: Dashboard ETA + calibração
  │
  ├─→ Fase 4: CLSTM context fusion (independente do ETA, após baseline)
  │
  ├─→ Fase 2.9: Análise do ledger (requer 200+ sinais)
  │     │
  │     └─→ Fase 4+: Features de ledger no CLSTM
  │
  └─→ Futuro: Weibull head, cost por venue, horizonte adaptativo
```

---

## Fase ETA-1 — Multi-Quantile Head + Pinball Loss

**Quando:** Imediatamente após baseline v3 medido.
**Dados:** Mesmos dados do baseline (sem requisito adicional).
**Tempo de implementação:** ~3h
**Tempo de treino:** ~5-7 min (pipeline optimizado)

### O que muda

Ficheiros:
- `ml_model.py`: ETA head de 1 → 5 neurónios (Q10/Q25/Q50/Q75/Q90). Monotonicity via softplus deltas: base = raw[0], deltas = softplus(raw[1:4]), quantis = cumsum.
- `train_model.py`: Pinball loss substitui SmoothL1. Loss por quantil: L_τ(y, ŷ) = max(τ(y-ŷ), (τ-1)(y-ŷ)). Total = média dos 5 quantis. Peso 0.25 mantido (loss_prob + 0.25 * loss_eta).
- `train_model.py`: _eta_metrics expandido — reportar calibração (coverage por quantil) + pinball score.
- `ml_analyzer.py`: Inferência extrai 5 quantis, _eta_payload inclui Q10/Q25/Q50/Q75/Q90.

### O que NÃO muda

- Features (mesmas 40)
- Loss de probabilidade (FocalLoss)
- Split, embargo, early stopping
- Signal convention

### Gate de aceitação

- AUC: ≥ baseline v3 - 0.02 (ETA loss não deve degradar prob head)
- ETA Q50 coverage: entre 0.40 e 0.60 (50% dos samples reais caem abaixo do Q50 predito)
- Pinball score Q50: < MAE do baseline

### Comparação

```
Baseline v3:  AUC=X.XX, ETA MAE=Y.Y min
ETA-1:        AUC=X.XX, Q50 coverage=0.XX, Pinball=Z.Z
```

---

## Fase ETA-2 — Censoring Correction

**Quando:** Imediatamente após ETA-1 validado.
**Dados:** Mesmos dados. Muda como os y=0 são tratados no loss.
**Tempo de implementação:** ~2h
**Tempo de treino:** ~5-7 min

### O que muda

Ficheiros:
- `train_model.py`: Samples censurados (y_class=0) contribuem para ETA loss. Para censurados, sabemos que T_real > horizon. Loss censurada: penalizar se predição < horizon (sub-estimação). L_censored = max(0, horizon - pred_Q50)² / horizon².
- Agora usa 100% dos samples em vez de ~4.57% (só positivos).

### Gate de aceitação

- ETA MAE nos positivos: deve melhorar (mais dados → melhor generalização)
- Coverage não deve degradar
- AUC: inalterado (ETA loss weight = 0.25, prob head domina)

---

## Fase ETA-3 — Label peak_ts

**Quando:** APÓS ETA-1+2 validados.
**Dados:** REQUER DADOS NOVOS. O campo peak_ts precisa ser gravado pelo TrackerEpisode.
**Tempo de colecta:** 2-4 semanas de dados com peak_ts (mínimo ~1000 episódios com peak_ts != 0).
**Tempo de implementação:** ~2h
**Tempo de treino:** ~5-7 min

### O que muda

Ficheiros:
- `spread_tracker.py`: Adicionar `peak_ts: float` ao TrackerEpisode. Actualizar durante o tracking: quando entry_spread > current_peak, guardar timestamp.
- `ml_dataset.py`: y_eta = peak_ts - current_ts (tempo até pico, não até close). Fallback: se peak_ts == 0 ou indisponível, usar end_ts como antes.
- `train_model.py`: Nenhuma mudança (loss já funciona com qualquer y_eta target).

### Por que não implementar agora

Os episódios na DB actual NÃO têm peak_ts. Se implementares o tracking agora, precisas de esperar 2-4 semanas para ter dados suficientes. Podes adiantar: adicionar a coluna e o tracking já, mas o treino com peak_ts só após colectar dados.

### Gate de aceitação

- Q50 mediano mais curto que com end_ts (pico é mais previsível que close)
- Coverage Q50 melhor ou igual
- AUC: inalterado

---

## Fase ETA-4 — Dashboard ETA + Métricas de Calibração

**Quando:** APÓS ETA-3 (ou ETA-2 se peak_ts for adiado).
**Dados:** Precisa de outputs do multi-quantile head para calcular métricas.
**Tempo de implementação:** ~2h

### O que muda

Ficheiros:
- `ml_analyzer.py`: _eta_payload com Q10/Q25/Q50/Q75/Q90 condicionais no card. eta_uncertainty = Q75 - Q25. Substituir ETA empírico por ETA condicional do modelo.
- `signal_ledger.py`: ALTER TABLE para adicionar eta_q10_at_signal, eta_q25_at_signal, eta_q50_at_signal, eta_q75_at_signal, eta_q90_at_signal, eta_uncertainty_sec.
- `ml_analyzer.py`: compute_signal_score recebe eta_uncertainty_min. uncertainty_penalty = 1.0 / (1.0 + max(eta_uncertainty_min, 0.0) / 30.0).
- `soak_runbook.py`: Gate de calibração ETA no Stage 2 (coverage, pinball).

### Dashboard antes/depois

```
ANTES:   "ETA: 32m"
DEPOIS:  "ETA: 18m / 32m / 61m (±21m)"
         Q10=18m  Q25=25m  Q50=32m  Q75=46m  Q90=61m
```

---

## Fase 4 — CLSTM Context Vector Fusion

**Quando:** Após baseline v3 medido. Independente do ETA redesign.
**Dados:** Mesmos dados do baseline (context vector vem do recurring_context, já existe).
**Tempo de implementação:** 3-4 sessões (~8-12h total)
**Tempo de treino:** ~7-10 min (modelo maior, mais parâmetros)

### O que muda

Ficheiros:
- `ml_model.py`: Nova classe CLSTM. LSTM(40 features) → attention → pooled. Context vector (19 features) → context_proj(Linear). Concatenar pooled + context_proj → heads prob/ETA.
- `ml_dataset.py`: DatasetBundle inclui context_vector por sample.
- `train_model.py`: Passar context_vector ao modelo. config: model_type="lstm" vs "clstm".
- `ml_analyzer.py`: Passar context vector na inferência.

### Context vector (19 features)

Contínuas (8): support_short, support_long, entry_median_2h, exit_median_2h, median_total_spread, episode_density_per_hour, median_episode_duration_sec, entry_exit_coherence_ratio

Binárias (2): entry_coherent, exit_coherent

One-hot (9): context_strength (3), range_status (3), entry_position (3)

### Gate de aceitação

- PR-AUC ≥ baseline v3 (benchmark é v3 sem context, NÃO v2)
- ECE ≤ 0.10 (recalibrar Platt)
- Se CLSTM não bater v3, descartar — o LSTM simples é suficiente

---

## Fase 2.9 — Análise do Ledger

**Quando:** Após 200+ sinais resolvidos no signal_ledger.
**Dados:** 200+ sinais com outcome_status != 'pending'.
**Tempo estimado para acumular:** Depende da frequência de EXECUTE. Se o sistema emite ~5-10 sinais/dia, 200 sinais ≈ 3-6 semanas. Se emite ~20-30/dia, ≈ 1-2 semanas.
**Tempo de implementação:** ~2-3h (análise + dashboard)

### O que fazer

- Hit rate por política (shallow/median/deep)
- Hit rate por par (quais pares o modelo acerta mais)
- Calibração: prob_at_signal vs hit rate real (reliability diagram)
- ETA vs outcome_duration_sec (calibração temporal)
- Signal score vs outcome: scores altos correlacionam com hits?
- False positive analysis: o que os misses têm em comum?

### Output

Relatório + dashboard metrics. Informam se o modelo precisa de retrain com dados novos ou se os thresholds precisam de ajuste.

---

## Fase 4+ — Features de Ledger no CLSTM

**Quando:** Após 200+ sinais + Fase 4 implementada.
**Dados:** Ledger com outcomes resolvidos.
**Tempo de implementação:** ~2h

### O que adiciona

- `shrunk_hit_rate_pair`: (α₀ + hits) / (α₀ + β₀ + n) com prior global do ledger
- Embargo temporal: 1 horizonte entre último sinal e sample de treino
- Adicionar ao context vector do CLSTM (19 → 20-21 features)

---

## Futuro (6+ meses)

| Fase | O que | Quando |
|---|---|---|
| Weibull ETA | Head paramétrica para censoring nativo + distribuição contínua | Após ETA-4 validado, se pinball loss insuficiente |
| Cost por venue | default_cost_estimate_pct por exchange/par em vez de fixo | Quando houver dados de custos reais de operações |
| Execution ledger | Gravar operações reais, não apenas sinais | Quando sistema estiver a operar com capital |
| Threshold por frequência | Ajustar min_total_spread_pct por par baseado na frequência de episódios | Após 2.9 analysis, se pares high-frequency forem sub-óptimos |
| Horizonte adaptativo | prediction_horizon_sec variável por par | Após CLSTM + ETA-4, análise de heterogeneidade temporal |

---

## Resumo: Quando implementar cada fase

```
┌──────────┬──────────────────────────────┬───────────────────────────────────────────┬───────────┐
│   Fase   │       Pré-requisito          │          Dados necessários                │   Impl.   │
├──────────┼──────────────────────────────┼───────────────────────────────────────────┼───────────┤
│ ETA-1    │ Baseline v3 medido           │ Mesmos dados do baseline                  │ ~3h       │
├──────────┼──────────────────────────────┼───────────────────────────────────────────┼───────────┤
│ ETA-2    │ ETA-1 validado               │ Mesmos dados                              │ ~2h       │
├──────────┼──────────────────────────────┼───────────────────────────────────────────┼───────────┤
│ ETA-3    │ ETA-1+2 validados            │ 2-4 semanas de episódios com peak_ts      │ ~2h       │
│          │                              │ (~1000 episódios mínimo)                  │           │
├──────────┼──────────────────────────────┼───────────────────────────────────────────┼───────────┤
│ ETA-4    │ ETA-3 (ou ETA-2 se peak_ts   │ Outputs do multi-quantile head            │ ~2h       │
│          │ adiado)                      │                                           │           │
├──────────┼──────────────────────────────┼───────────────────────────────────────────┼───────────┤
│ Fase 4   │ Baseline v3 medido           │ Mesmos dados (context vem do recurring)   │ ~8-12h    │
│ (CLSTM)  │                              │                                           │           │
├──────────┼──────────────────────────────┼───────────────────────────────────────────┼───────────┤
│ Fase 2.9 │ 200+ sinais resolvidos       │ 1-6 semanas de sinais acumulados          │ ~2-3h     │
├──────────┼──────────────────────────────┼───────────────────────────────────────────┼───────────┤
│ Fase 4+  │ Fase 4 + 200+ sinais         │ Ledger com outcomes                       │ ~2h       │
├──────────┼──────────────────────────────┼───────────────────────────────────────────┼───────────┤
│ Weibull  │ ETA-4 insuficiente           │ Mesmos dados                              │ ~4-6h     │
└──────────┴──────────────────────────────┴───────────────────────────────────────────┴───────────┘
```

**Nota sobre ETA-3 (peak_ts):** Podes adiantar a gravação do peak_ts no TrackerEpisode agora sem custo. Isso inicia a colecta de dados. Quando tiveres ~1000 episódios com peak_ts, implementas a mudança de label. Se adicionares a coluna hoje, os dados estarão prontos em 2-4 semanas.

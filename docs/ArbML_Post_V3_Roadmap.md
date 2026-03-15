# ArbML — Post-V3 Roadmap (actualizado 2026-03-15)

## Estado actual

```
✅ Fase S:     Pipeline optimizado (~25 min, stride=5, scaffold, AMP)
✅ Fase 1:     40 features (v3_exit_aware_40)
✅ Fase 2:     Dashboard 3 políticas (shallow/median/deep)
✅ Fase 2.5:   Signal Ledger (SQLite, cooldown, resolve)
✅ Fase 3:     SignalScore (prob × viability × support × strength × drift × eta)
✅ Hardening:  6 camadas (ingest → treino), 204 tests
✅ Soak plan:  v3.1 pronto (aguarda baseline)

🔄 Baseline v3:  EM PROGRESSO (8h runtime, T+0 passed)
⏳ Soak 24h:     Após baseline
⏳ Fases pós-v3: Após soak
```

## Sequência completa

```
BASELINE V3 (em progresso)
  │
  └─→ Soak 24h (validar infraestrutura autónoma)
        │
        └─→ FASES PÓS-V3:
              │
              ├─→ S2: max_samples_per_pair (antes do 2º retrain)
              │
              ├─→ Fase ETA-1: Multi-quantile head + Pinball loss
              │     │
              │     └─→ Fase ETA-2: Censoring correction
              │           │
              │           └─→ Fase ETA-3: Label peak_ts (dados já existem)
              │                 │
              │                 └─→ Fase ETA-4: Dashboard ETA + calibração
              │
              ├─→ Fase 4: CLSTM context fusion (independente do ETA)
              │
              ├─→ Fase 2.9: Análise do ledger (requer 200+ sinais)
              │     │
              │     └─→ Fase 4+: Features de ledger no CLSTM
              │
              └─→ Futuro: Weibull head, cost por venue, horizonte adaptativo
```

---

## S2 — max_samples_per_pair (ANTES do 2º retrain)

**Quando:** Após soak completo, antes do próximo retrain automático.
**Dados:** Mesmos dados.
**Tempo de implementação:** ~2h
**Tempo de treino:** ~25 min (pipeline optimizado)

### O que muda

- `ml_dataset.py`: Adicionar parâmetro `max_samples_per_pair` ao `build_dataset_bundle`.
  Após windowing, se um par tem > max_samples_per_pair windows, subsample aleatoriamente.
- Default: `max_samples_per_pair = 2000`
- Logging: pair_sample_counts já existe no bundle summary (hardening). Gini e top1_pair_fraction
  já reportados no training report.

### Por que antes do 2º retrain

O hardening adicionou pair_concentration warning (Gini > 0.70 ou top1 > 20%). Se o primeiro
baseline mostrar concentração elevada, S2 é o fix. Stride=5 mitiga parcialmente, mas pares
com 50K records ainda contribuem 10K samples vs pares raros com 40 samples.

---

## Fase ETA-1 — Multi-Quantile Head + Pinball Loss

**Quando:** Após soak completo e baseline v3 validado em produção.
**Dados:** Mesmos dados do baseline (sem requisito adicional).
**Tempo de implementação:** ~3h
**Tempo de treino:** ~25 min (pipeline optimizado)

### O que muda

Ficheiros:
- `ml_model.py`: ETA head de 1 → 5 neurónios (Q10/Q25/Q50/Q75/Q90). Monotonicity via
  softplus deltas: base = raw[0], deltas = softplus(raw[1:4]), quantis = cumsum.
- `train_model.py`: Pinball loss substitui SmoothL1. Loss por quantil:
  L_τ(y, ŷ) = max(τ(y-ŷ), (τ-1)(y-ŷ)). Total = média dos 5 quantis.
  Peso 0.25 mantido (loss_prob + 0.25 * loss_eta).
- `train_model.py`: _eta_metrics expandido — reportar calibração (coverage por quantil)
  + pinball score.
- `ml_analyzer.py`: Inferência extrai 5 quantis, _eta_payload inclui Q10/Q25/Q50/Q75/Q90.

### O que NÃO muda

- Features (mesmas 40)
- Loss de probabilidade (FocalLoss)
- Split, embargo, early stopping
- Signal convention
- Hardening guards (tensor integrity, positive floor, etc.)

### Gate de aceitação

- AUC: ≥ baseline v3 - 0.02 (ETA loss não deve degradar prob head)
- ETA Q50 coverage: entre 0.40 e 0.60
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
**Tempo de treino:** ~25 min

### O que muda

- `train_model.py`: Samples censurados (y_class=0) contribuem para ETA loss.
  Loss censurada: penalizar se predição < horizon (sub-estimação).
  L_censored = max(0, horizon - pred_Q50)² / horizon².
- Usa 100% dos samples em vez de ~4.57% (só positivos).

### Gate de aceitação

- ETA MAE nos positivos: deve melhorar
- Coverage não deve degradar
- AUC: inalterado

---

## Fase ETA-3 — Label peak_ts

**Quando:** APÓS ETA-1+2 validados.
**Dados:** peak_ts JÁ EXISTE na DB (TrackerEpisode grava peak_ts desde a implementação v3).
  NÃO precisa de esperar dados novos — os episódios existentes já têm peak_ts.
**Tempo de implementação:** ~1-2h (scope menor que previsto)
**Tempo de treino:** ~25 min

### O que muda (scope reduzido)

O TrackerEpisode já grava `peak_ts` (timestamp do pico). O que falta:
- `feature_contracts.py`: Adicionar `peak_ts: float = 0.0` ao `NormalizedEpisode`.
  Actualizar `normalize_episode()` para extrair peak_ts.
- `ml_dataset.py`: Na relabel vectorizada, mudar `y_eta = end_ts - current_ts` para
  `y_eta = peak_ts - current_ts` (tempo até pico, não até close).
  Fallback: se peak_ts == 0 ou peak_ts <= start_ts, usar end_ts.

### O que NÃO precisa de ser feito (já existe)

- ~~spread_tracker.py: Adicionar peak_ts ao TrackerEpisode~~ → JÁ EXISTE
- ~~Esperar 2-4 semanas de colecta~~ → peak_ts já está na DB

### Gate de aceitação

- Q50 mediano mais curto que com end_ts
- Coverage Q50 melhor ou igual
- AUC: inalterado

---

## Fase ETA-4 — Dashboard ETA + Métricas de Calibração

**Quando:** APÓS ETA-3 (ou ETA-2 se peak_ts for adiado).
**Dados:** Outputs do multi-quantile head.
**Tempo de implementação:** ~2h

### O que muda

- `ml_analyzer.py`: _eta_payload com Q10/Q25/Q50/Q75/Q90 condicionais no card.
  eta_uncertainty = Q75 - Q25. Substituir ETA empírico por ETA condicional do modelo.
- `signal_ledger.py`: ALTER TABLE para adicionar eta_q10..q90_at_signal, eta_uncertainty_sec.
- `ml_analyzer.py`: compute_signal_score recebe eta_uncertainty_min.
  uncertainty_penalty = 1.0 / (1.0 + max(eta_uncertainty_min, 0.0) / 30.0).
- `soak_runbook.py`: Gate de calibração ETA no Stage 2 (coverage, pinball).

### Dashboard antes/depois

```
ANTES:   "ETA: 32m"
DEPOIS:  "ETA: 18m / 32m / 61m (±21m)"
         Q10=18m  Q25=25m  Q50=32m  Q75=46m  Q90=61m
```

---

## Fase 4 — CLSTM Context Vector Fusion

**Quando:** Após baseline v3 medido e soak completo. Independente do ETA redesign.
**Dados:** Mesmos dados (context vector vem do recurring_context, já existe).
**Tempo de implementação:** 3-4 sessões (~8-12h total)
**Tempo de treino:** ~30 min (modelo maior)

### O que muda

- `ml_model.py`: Nova classe CLSTM. LSTM(40 features) → attention → pooled.
  Context vector (19 features) → context_proj(Linear).
  Concatenar pooled + context_proj → heads prob/ETA.
- `ml_dataset.py`: DatasetBundle inclui context_vector por sample.
- `train_model.py`: Passar context_vector ao modelo. config: model_type="lstm" vs "clstm".
- `ml_analyzer.py`: Passar context vector na inferência.

### Context vector (19 features)

Contínuas (8): support_short, support_long, entry_median_2h, exit_median_2h,
median_total_spread, episode_density_per_hour, median_episode_duration_sec,
entry_exit_coherence_ratio

Binárias (2): entry_coherent, exit_coherent

One-hot (9): context_strength (3), range_status (3), entry_position (3)

### Gate de aceitação

- PR-AUC ≥ baseline v3 (benchmark é v3 sem context, NÃO v2)
- ECE ≤ 0.10 (recalibrar Platt)
- Se CLSTM não bater v3, descartar

---

## Fase 2.9 — Análise do Ledger

**Quando:** Após 200+ sinais resolvidos no signal_ledger.
**Dados:** 200+ sinais com outcome_status != 'pending'.
**Tempo estimado para acumular:** Depende da frequência de EXECUTE.
  ~5-10 sinais/dia → 200 sinais ≈ 3-6 semanas.
  ~20-30/dia → ≈ 1-2 semanas.
**Tempo de implementação:** ~2-3h

### O que fazer

- Hit rate por política (shallow/median/deep)
- Hit rate por par
- Calibração: prob_at_signal vs hit rate real (reliability diagram)
- ETA vs outcome_duration_sec
- Signal score vs outcome
- False positive analysis

---

## Fase 4+ — Features de Ledger no CLSTM

**Quando:** Após 200+ sinais + Fase 4 implementada.
**Dados:** Ledger com outcomes resolvidos.
**Tempo de implementação:** ~2h

### O que adiciona

- `shrunk_hit_rate_pair`: (α₀ + hits) / (α₀ + β₀ + n) com prior global
- Embargo temporal: 1 horizonte entre último sinal e sample de treino
- Adicionar ao context vector do CLSTM (19 → 20-21 features)

---

## Futuro (6+ meses)

| Fase | O que | Quando |
|---|---|---|
| Weibull ETA | Head paramétrica para censoring nativo + distribuição contínua | Após ETA-4, se pinball insuficiente |
| Cost por venue | default_cost_estimate_pct por exchange/par | Quando houver dados de custos reais |
| Execution ledger | Gravar operações reais, não apenas sinais | Quando operar com capital |
| Threshold por frequência | min_total_spread_pct por par | Após 2.9, se pares high-freq sub-óptimos |
| Horizonte adaptativo | prediction_horizon_sec variável por par | Após CLSTM + ETA-4 |

---

## Resumo: Quando implementar cada fase

```
┌──────────┬───────────────────────────────┬──────────────────────────────────────────┬───────────┐
│   Fase   │       Pré-requisito           │          Dados necessários               │   Impl.   │
├──────────┼───────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ S2       │ Soak completo                 │ Mesmos dados                             │ ~2h       │
├──────────┼───────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ ETA-1    │ Soak completo + baseline      │ Mesmos dados                             │ ~3h       │
│          │ validado em produção          │                                          │           │
├──────────┼───────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ ETA-2    │ ETA-1 validado                │ Mesmos dados                             │ ~2h       │
├──────────┼───────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ ETA-3    │ ETA-1+2 validados             │ peak_ts JÁ EXISTE na DB                  │ ~1-2h     │
│          │                               │ (não precisa de colecta adicional)       │           │
├──────────┼───────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ ETA-4    │ ETA-3 (ou ETA-2 se peak_ts    │ Outputs do multi-quantile head           │ ~2h       │
│          │ adiado)                       │                                          │           │
├──────────┼───────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ Fase 4   │ Soak completo + baseline      │ Mesmos dados (context do recurring)      │ ~8-12h    │
│ (CLSTM)  │                               │                                          │           │
├──────────┼───────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ Fase 2.9 │ 200+ sinais resolvidos        │ 1-6 semanas de sinais                    │ ~2-3h     │
├──────────┼───────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ Fase 4+  │ Fase 4 + 200+ sinais          │ Ledger com outcomes                      │ ~2h       │
├──────────┼───────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ Weibull  │ ETA-4 insuficiente            │ Mesmos dados                             │ ~4-6h     │
└──────────┴───────────────────────────────┴──────────────────────────────────────────┴───────────┘
```

## Notas sobre o pipeline actual

- Pipeline total: ~25 min (1GB), medido com dados reais
- Scaffold build: ~383s (piso Python, windowing loop)
- Relabel: ~18-20s (vectorizado)
- GPU training: ~3-5 min (AMP + batch 1024 + stride=5)
- Certificação full: ~18 min (max_certification_duration_sec=1800)
- Todos os tempos de treino nas fases acima usam o pipeline optimizado
- O hardening (tensor integrity, positive floor, normalization guard, degenerate features,
  episode completeness, pair concentration) aplica-se a TODOS os retrains futuros

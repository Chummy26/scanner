# ArbML — Data Quality Verification Plan

## Objectivo

Desbloquear o primeiro treino confiável do baseline v3. Todos os checks são internos,
implementados nos ficheiros existentes (training_certification.py, ml_dataset.py, train_model.py).
Zero dependências externas novas.

## O que já existe (12 gates)

```
Gate 01: SQLite integrity (PRAGMA integrity_check)
Gate 02: Intra-block temporal regularity (intervalo, irregularidade)
Gate 03: Completeness (pares desaparecidos, intermitentes)
Gate 04: Checkpoint stationarity (regime shifts entry/exit)
Gate 05: Intra-soak feature drift (first vs second half)
Gate 06: Episode yield (episódios/hora, duração)
Gate 07: Book health (hourly health verdicts)
Gate 08: Reconnection stress
Gate 09: Runtime audit → SQLite consistency
Gate 10: Dual-mode preflight (multi-config viability)
Gate 11: Runtime audit health
Gate 12: Entry/exit quality (frozen, outliers, bilateral zeros, episode quality)
```

Split validation em build_group_splits:
- Embargo gap (train_end_label_ts < val_start_ts)
- Pair overlap count
- Chronological order assertion

## O que FALTA — organizado em 3 tiers

---

### TIER 1 — BLOCK (treino não arranca se falhar)

Estes checks protegem contra dados corrompidos, leakage, ou insuficiência
que invalidariam qualquer resultado.

#### T1.1 — Tensor integrity post-construction

**Onde:** build_dataset_bundle, APÓS construir X_tensor, y_class, y_eta
**O que:** assert torch.isfinite(X).all(), assert torch.isfinite(y_class).all(),
assert torch.isfinite(y_eta).all()
**Por quê:** Um único NaN propaga-se silenciosamente pelo LSTM e corrompe
todos os gradientes. Não existe gate que verifique isto actualmente.
**Referência:** Standard PyTorch practice; Google ML Test Score (Breck et al., 2017)
recomenda "test that all features are computed and finite"

#### T1.2 — Degenerate v3 feature detection

**Onde:** training_certification.py, novo check após bundle build (entre gate_05 e gate_06)
**O que:** Para cada uma das 40 features, verificar:
- Variância > 1e-10 (feature constante = inútil)
- NaN fraction < 1% (campo não computado)
- Zero fraction < 95% (feature v3 sem dados de episódio)
- Inf fraction = 0%
**Threshold BLOCK:** Qualquer feature com variância = 0 OU NaN > 5% OU Inf > 0
**Threshold WARN:** Zero fraction > 80% para features de exit/cycle (indica episódios sem campos v3)
**Por quê:** As 15 novas features v3 (exit rolling, cycle, relation) dependem de episódios
com exit_spread, peak_entry, duration preenchidos. Se esses campos são todos 0.0
nos episódios da DB, as features serão todas zero — modelo não aprende nada novo.
**Referência:** TFDV (TensorFlow Data Validation) faz schema inference + anomaly detection
para features degeneradas. Schelter et al. (2018, PVLDB "Automating Large-Scale Data Quality")
define completeness e informativeness como propriedades fundamentais.

#### T1.3 — Episode field completeness for v3

**Onde:** training_certification.py, junto com gate_06 ou como extensão
**O que:** Query nos episódios fechados:
- % com `exit_spread_at_close IS NOT NULL`
- % com `peak_entry_spread IS NOT NULL`
- % com `duration_sec >= 0`
- todos os 3 campos devem ser finitos
**Threshold BLOCK:** Se < 50% dos episódios têm os 3 campos
**Threshold WARN:** Se < 80%
**Por quê:** Se os dados foram colectados antes de o TrackerEpisode gravar estes campos,
as features v3 de ciclo (mean_total_spread, reversion_speed, close_rate) e exit
(mean_exit, exit_p10) serão todas zero. O treino v3 seria equivalente ao v2.
**Nota importante:** `peak_entry_spread <= 0` e `exit_spread_at_close = 0.0` são
resultados de mercado legítimos, não incompletude. Episódios sem entrada favorável
ou que fecham exatamente no neutro continuam a ser episódios v3 completos.
**Referência:** Sambasivan et al. (2021, CHI) "Data Cascades" — 92% dos practitioners
reportam problemas compounding por dados incompletos descobertos tarde.

#### T1.4 — Minimum positive samples per split fold

**Onde:** train_model.py, run_training_loop, APÓS build_group_splits
**O que:** assert min(positives_train, positives_val, positives_test) >= 50
**Threshold BLOCK:** < 50 positivos em qualquer fold (AUC não é estimável com confiança)
**Threshold WARN:** < 100 em qualquer fold
**Por quê:** Hanczar et al. (2010, Bioinformatics) demonstram que AUC com < 50 positivos
tem variância > 0.15 — o número reportado não é confiável.
Com 4.57% positive rate e 531K samples (stride=5): ~24K positivos. Split 70/15/15
→ ~3.6K positivos no test. Margem ampla. Mas com datasets menores ou threshold
mais alto, pode falhar.
**Referência:** Hanczar et al. (2010) "Small-sample precision of ROC-related estimates"

#### T1.5 — Normalization leakage guard

**Onde:** train_model.py, run_training_loop, APÓS normalize_features
**O que:**
1. Verificar que feature_mean e feature_std foram computados APENAS no train split:
   assert feature_mean computed from splits["train"].X
2. Após normalizar, verificar que train.X.mean(dim=(0,1)) ≈ 0 (tolerância 0.01)
   e val.X.mean(dim=(0,1)) != 0 exactamente (val não foi usado para fit)
3. Log max(abs(val.X.mean(dim=(0,1)))) — se < 0.001, suspeita de fit global
**Threshold BLOCK:** Se val.X.mean ≈ 0 (< 0.001) para TODAS as features → scaler fit global
**Por quê:** Normalization leakage é a causa #1 de AUC inflado em pipelines de time series.
Se o scaler vê dados futuros, as features ficam artificialmente centradas no test set.
O código actual parece correcto (compute_feature_stats no train), mas não há ASSERT.
**Referência:** scikit-learn Common Pitfalls documentation; Hughes & Zhang (2026,
Frontiers in Computer Science) "Bursting the bubble" mostra AUC 95% → 51% após fix.

#### T1.6 — Cross-split temporal contamination

**Onde:** ml_dataset.py, build_group_splits, APÓS criar splits
**O que:** assert max(train.timestamps) + prediction_horizon_sec <= min(val.timestamps)
       assert max(val.timestamps) + prediction_horizon_sec <= min(test.timestamps)
**Nota:** O embargo_time_sec já é loggeado no split_summary, mas NÃO enforced como assert.
Se o gap for negativo, o treino prossegue silenciosamente.
**Threshold BLOCK:** embargo_gap < 0 (overlap temporal entre splits)
**Threshold WARN:** embargo_gap < prediction_horizon_sec (embargo insuficiente)
**Por quê:** López de Prado (2018, "Advances in Financial Machine Learning") formaliza
o purging rule: qualquer sample de treino cujo label window [t, t+H] intersecta
o test set contamina a avaliação. O código já calcula o gap mas não bloqueia.
**Referência:** López de Prado (2018), purged cross-validation; timeseriescv library

---

### TIER 2 — WARN (treino prossegue, report sinaliza)

Estes checks identificam problemas que degradam qualidade mas não invalidam o treino.

#### T2.1 — Right-censoring fraction

**Onde:** Já tracked em bundle.summary["skipped_windows_right_censored"]
**O que:** Calcular: censored_rate = right_censored / (right_censored + num_samples)
**Threshold WARN:** censored_rate > 40% (sessões curtas dominam, perde-se quase metade)
**Log:** Incluir no training summary report
**Por quê:** Se a maioria das sessões tem < 2x prediction_horizon (< 8h para H=4h),
o right-censoring descarta >50% dos windows, enviesando labels para negativo
nas extremidades e reduzindo o volume efectivo de treino.

#### T2.2 — Pair concentration (Gini)

**Onde:** build_dataset_bundle summary OU training_certification como check adicional
**O que:** Contar samples por par. Calcular Gini coefficient:
  sorted_counts = sorted(pair_sample_counts)
  n = len(sorted_counts)
  gini = (2 * sum(i * c for i, c in enumerate(sorted_counts))) / (n * sum(sorted_counts)) - (n + 1) / n
**Threshold WARN:** Gini > 0.7 (top 10% dos pares dominam >70% dos samples)
**Threshold MONITOR:** Top-1 pair fraction > 20%
**Log:** Top 10 pares por sample count, Gini, top-1 fraction
**Por quê:** Se DOGE|mexc|gate contribui 25% dos samples, o modelo aprende "padrões DOGE",
não padrões gerais de spread. O stride=5 mitiga parcialmente mas não elimina.
S2 (max_samples_per_pair) é o fix definitivo, mas até lá, o warning sinaliza.
**Referência:** Sampling Analysis doc; cross-learning literature (Smyl, 2020, M4)

#### T2.3 — Label temporal stationarity

**Onde:** build_dataset_bundle summary OU training_certification
**O que:** Dividir timestamps em 4-8 bins temporais equidistantes. Calcular positive_rate
por bin. Verificar que o CV (coeficiente de variação) < 1.5.
**Threshold WARN:** CV > 1.5 OU qualquer bin com positive_rate = 0
**Por quê:** Se todos os positivos estão concentrados nas primeiras 2h do dataset
e o test set usa as últimas horas, o modelo parece aprender mas os positivos
já não existem no período de teste. Indica regime shift ou threshold mal calibrado.

#### T2.4 — V3 feature warm-up coverage

**Onde:** Novo check no build_dataset_bundle ou feature_contracts.py
**O que:** Para segmentos com < 480 records (~2h a 15s), as rolling features de 2h e 8h
retornam valores parciais (mean/std calculados com janela incompleta).
Contar: % de samples onde a janela de 8h (28800s) está < 50% preenchida.
**Threshold WARN:** > 30% dos samples com warm-up parcial nas features de 8h
**Por quê:** Features com warm-up parcial não são erradas (retornam 0.0 ou valor parcial),
mas o LSTM pode interpretar cold-start values como padrão real. Se > 30% dos samples
estão em warm-up, as features de 8h são mais ruído que sinal.

#### T2.5 — Feature variance ratio (v3 vs v2)

**Onde:** training_certification.py, extensão do gate_05
**O que:** Para as 15 features v3 novas, verificar que a variance média é pelo menos
10% da variance média das 25 features v2.
**Threshold WARN:** variance(v3_new) / variance(v2_existing) < 0.10
**Por quê:** Se as features novas têm variance próxima de zero comparada às existentes,
o modelo vai ignorá-las após normalização — o treino v3 seria functionally idêntico
ao v2. Isso indicaria que os dados não têm episódios v3-completos suficientes.

---

### TIER 3 — MONITOR (logged no training summary, análise pós-treino)

Estes metrics são informativos e acumulam-se ao longo de retrains.

#### T3.1 — Per-feature summary statistics

**Onde:** training_report dentro de run_training_loop
**O que:** Para cada uma das 40 features: mean, std, min, max, zero_fraction,
nan_fraction, p01, p99 — computados no train split.
**Output:** Dict no training_report["feature_quality_summary"]
**Uso:** Comparar entre retrains. Se uma feature muda drasticamente (p99 duplica),
algo mudou nos dados.

#### T3.2 — Effective sample size

**Onde:** training_report
**O que:** ESS = num_samples / (1 + max_pair_fraction * (num_pairs - 1))
Aproximação prática: num_samples * (1 - gini)
**Output:** training_report["effective_sample_size"]
**Uso:** Se ESS / num_samples < 0.3, o dataset é dominado por poucos pares.

#### T3.3 — Bootstrap AUC confidence interval

**Onde:** train_model.py, APÓS avaliação final no test set
**O que:** 1000 bootstrap resamples do test set. Calcular AUC em cada. Report CI 95%.
**Output:** training_report["auc_bootstrap_ci"] = {"lower": X.XX, "upper": X.XX, "width": X.XX}
**Uso:** Se width > 0.10, o test set é pequeno demais para confiar no AUC.
Se width > 0.15, reportar WARNING (mas não bloquear).
**Referência:** Hanczar et al. (2010); sklearn.utils.resample

#### T3.4 — Split balance report

**Onde:** training_report
**O que:** Para cada split (train/val/test):
- num_samples, num_positive, positive_rate
- num_unique_pairs, top_3_pairs_by_samples
- temporal_span_hours
- embargo_gap_sec para o split seguinte
**Output:** training_report["split_quality"]
**Uso:** Permite auditar se o split é razoável sem re-correr o pipeline.

---

## Onde implementar cada check

```
┌─────────┬─────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│  Check  │                    Ficheiro                     │                      Local                       │
├─────────┼─────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ T1.1    │ ml_dataset.py (build_dataset_bundle)            │ Após construir X_tensor, y_class, y_eta          │
│ T1.2    │ training_certification.py (novo gate_05b)       │ Após bundle build, antes do gate_06              │
│ T1.3    │ training_certification.py (extensão gate_06)    │ Junto com episode yield                          │
│ T1.4    │ train_model.py (run_training_loop)              │ Após build_group_splits, antes do treino         │
│ T1.5    │ train_model.py (run_training_loop)              │ Após compute_feature_stats + normalize_features  │
│ T1.6    │ ml_dataset.py (build_group_splits)              │ No split_summary, enforce como assert            │
│ T2.1    │ train_model.py (training report)                │ Após bundle build, report section                │
│ T2.2    │ train_model.py / ml_dataset.py (summary)        │ Após bundle build, summary section               │
│ T2.3    │ training_certification.py (extensão gate_10)    │ Após label audit na certificação                 │
│ T2.4    │ ml_dataset.py (bundle summary)                  │ Dentro do windowing loop, counter                │
│ T2.5    │ training_certification.py (extensão gate_05)    │ Após feature drift check                         │
│ T3.1    │ train_model.py (training report)                │ Final do run_training_loop                       │
│ T3.2    │ train_model.py (training report)                │ Final do run_training_loop                       │
│ T3.3    │ train_model.py (training report)                │ Após avaliação no test set                       │
│ T3.4    │ train_model.py (training report)                │ Após build_group_splits                          │
└─────────┴─────────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

## Prioridade de implementação

### Batch 1 — Desbloquear treino (antes do baseline v3)

1. T1.1 — Tensor integrity (3 linhas de assert)
2. T1.2 — Degenerate feature detection (~30 linhas)
3. T1.3 — Episode field completeness (~15 linhas SQL)
4. T1.4 — Min positives per fold (3 linhas de assert)
5. T1.6 — Embargo enforcement (2 linhas de assert)

Estas 5 checks são ~50 linhas de código. Protegem contra os failure modes
mais graves. Implementar ANTES de correr o baseline v3.

### Batch 2 — Melhorar confiança (com o baseline v3)

6. T1.5 — Normalization leakage guard (~10 linhas)
7. T2.2 — Pair Gini (~15 linhas)
8. T2.5 — Feature variance ratio (~10 linhas)
9. T3.4 — Split balance report (~20 linhas)

Estas adicionam informação ao training report para diagnosticar se o AUC
do baseline é confiável.

### Batch 3 — Monitoring contínuo (para retrains subsequentes)

10. T2.1 — Right-censoring fraction (já existe, só logar)
11. T2.3 — Label temporal stationarity (~20 linhas)
12. T2.4 — Warm-up coverage (~15 linhas)
13. T3.1 — Per-feature summary (~15 linhas)
14. T3.2 — Effective sample size (~5 linhas)
15. T3.3 — Bootstrap AUC CI (~15 linhas)

---

## Relação com as failure reasons actuais da certificação

```
┌─────────────────────────────────────────┬─────────────────────────────────────────────────────────────────┐
│         Failure reason actual            │                      Resolução                                  │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ audit_sqlite_inconsistency              │ Snapshot fresco + WAL checkpoint resolve. NÃO é check novo.     │
│ tracker_sqlite_count_divergence         │ Snapshot fresco resolve. Gate 09 verifica.                      │
│ runtime_audit_stale                     │ Snapshot < 24h resolve. Gate 11 verifica.                       │
│ runtime_health_failed                   │ Derivado do audit stale. Resolve com snapshot fresco.           │
│ right_censoring_present                 │ NÃO deveria ser FAIL. Censoring é esperado e correcto.          │
│                                         │ Verificar se gate está a bloquear por existência de censoring   │
│                                         │ (errado) vs censoring excessivo >80% (aceitável como WARN).    │
│                                         │ T2.1 formaliza o threshold correcto.                            │
│ intra_block_irregularity                │ Gate 02 com threshold apertado. Verificar se o p50 do intervalo │
│                                         │ está entre 10-40s (OK para 15s target). Se ratio_p90 > 3.0,    │
│                                         │ pode ser bursts de dados normais. Considerar relaxar threshold. │
│ temporal_instability                    │ Gate 04/05 detecta. Para crypto, regime shifts são normais.     │
│                                         │ Se > 3 features drifted: FAIL. Se 1-3: WARNING. Verificar      │
│                                         │ se o snapshot cobre período com alta volatilidade.              │
└─────────────────────────────────────────┴─────────────────────────────────────────────────────────────────┘
```

## Diagnóstico rápido para desbloquear (correr ANTES de implementar checks novos)

```bash
cd backend && python3 -c "
import sys, sqlite3, time, numpy as np; sys.path.insert(0, 'src')

db = 'out/config/tracker_history.sqlite'
conn = sqlite3.connect(db, timeout=30)
now = time.time()

# 1. Freshness
newest = conn.execute('SELECT MAX(ts) FROM tracker_records').fetchone()[0]
age_h = (now - newest) / 3600
print(f'Data age: {age_h:.1f}h (need < 8h for fresh snapshot)')

# 2. Episode v3 fields
total_ep = conn.execute('SELECT COUNT(*) FROM tracker_pair_episodes WHERE is_closed = 1').fetchone()[0]
gate06_complete = conn.execute('''
    SELECT COUNT(*)
    FROM tracker_pair_episodes
    WHERE is_closed = 1
      AND exit_spread_at_close IS NOT NULL
      AND peak_entry_spread IS NOT NULL
      AND duration_sec IS NOT NULL
      AND duration_sec >= 0
      AND exit_spread_at_close = exit_spread_at_close
      AND peak_entry_spread = peak_entry_spread
      AND duration_sec = duration_sec
''').fetchone()[0]
peak_positive = conn.execute('SELECT COUNT(*) FROM tracker_pair_episodes WHERE is_closed = 1 AND peak_entry_spread > 0').fetchone()[0]
peak_non_positive = conn.execute('SELECT COUNT(*) FROM tracker_pair_episodes WHERE is_closed = 1 AND peak_entry_spread <= 0').fetchone()[0]
exit_zero = conn.execute('SELECT COUNT(*) FROM tracker_pair_episodes WHERE is_closed = 1 AND exit_spread_at_close = 0').fetchone()[0]
print(f'\nEpisode v3 completeness:')
print(f'  Total closed: {total_ep:,}')
print(f'  Gate 06 complete:   {gate06_complete:,} ({100*gate06_complete/max(total_ep,1):.0f}%)')
print(f'  Peak positive:      {peak_positive:,} ({100*peak_positive/max(total_ep,1):.0f}%)')
print(f'  Peak non-positive:  {peak_non_positive:,} ({100*peak_non_positive/max(total_ep,1):.0f}%)')
print(f'  Exit exactly zero:  {exit_zero:,} ({100*exit_zero/max(total_ep,1):.0f}%)')
v3_ready = gate06_complete/max(total_ep,1) >= 0.50
print(f'  V3 READY: {\"YES\" if v3_ready else \"NO — need 50%+ episodes with finite exit/peak/duration\"}\n')

# 3. Interval regularity
intervals = conn.execute('''
  SELECT pair_id, ts - LAG(ts) OVER (PARTITION BY pair_id ORDER BY ts) as gap
  FROM tracker_records
  ORDER BY pair_id, ts
''').fetchall()
gaps = [g for _, g in intervals if g is not None and g > 0]
if gaps:
    p50 = np.percentile(gaps, 50)
    p90 = np.percentile(gaps, 90)
    ratio = p90 / max(p50, 0.01)
    print(f'Record intervals: p50={p50:.1f}s p90={p90:.1f}s ratio={ratio:.1f}')
    print(f'Gate 02 expects: p50 in [10, 40]s, ratio < 3.0')
    print(f'Status: {\"PASS\" if 10 <= p50 <= 40 and ratio < 3.0 else \"INVESTIGATE\"}\n')

# 4. Positive rate estimate
above_050 = conn.execute('''
  SELECT COUNT(*) FROM tracker_pair_episodes
  WHERE is_closed = 1 AND (peak_entry_spread + exit_spread_at_close) >= 0.50
''').fetchone()[0]
print(f'Episodes >= 0.50%: {above_050:,} / {total_ep:,} ({100*above_050/max(total_ep,1):.1f}%)')
print(f'Need >= 100 for ~4% positive rate with 531K samples')

conn.close()
"
```

## Referências

- Breck et al. (2017) "ML Test Score: A Rubric for ML Production Readiness" IEEE Big Data
- Schelter et al. (2018) "Automating Large-Scale Data Quality Verification" PVLDB 11(12)
- Sambasivan et al. (2021) "Data Cascades in High-Stakes AI" CHI Best Paper
- Hanczar et al. (2010) "Small-sample precision of ROC-related estimates" Bioinformatics
- López de Prado (2018) "Advances in Financial Machine Learning" Wiley — purged CV, sample weights
- Hughes & Zhang (2026) "Bursting the bubble" Frontiers in Computer Science — window leakage taxonomy
- Northcutt et al. (2021) "Confident Learning" JAIR — label noise estimation
- Mukhoti et al. (2020) "Calibrating DNNs using Focal Loss" NeurIPS
- Passalis et al. (2019) "Deep Adaptive Input Normalization" IEEE TNNLS
- Neri (2021) "Domain Specific Concept Drift Detectors" arXiv:2103.14079
- Google TFDV (2018) — schema inference + anomaly detection
- SQLite WAL documentation (sqlite.org/wal.html) — snapshot isolation

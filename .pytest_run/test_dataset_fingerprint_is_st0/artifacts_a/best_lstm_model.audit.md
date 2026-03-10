# ArbML LSTM Audit

## Findings
- Alto: o bloco temporal mais fraco caiu abaixo do baseline sempre-negativo em PR-AUC.
- Médio: a calibração ficou acima do guardrail de ECE e exige threshold mais conservador.
- Médio: o threshold de STRONG_EXECUTE não atingiu a precisão mínima esperada para sinal forte.

## Dataset
- Samples: 81
- Positive samples: 5
- Negative samples: 76
- Pair count: 9
- Session count: 1
- Block count: 9
- Blocks used: 9
- Skipped short blocks: 0
- Cross-block windows: 0
- Cross-session windows: 0
- Feature abs sum: 3228.5522

## Split Integrity
- Train pairs: 7
- Validation pairs: 2
- Test pairs: 1
- Train blocks: 7
- Validation blocks: 2
- Test blocks: 1
- Unique overlapping pairs: 1
- Pairwise overlap sum: 1
- Unique pair overlap rate: 0.1111
- Embargo samples: 3
- Embargo time sec: 240
- Global temporal order: True
- Purged temporal separation ok: True
- Min cross-split gap sec: 0.0
- Split mode fallback reason: insufficient_sessions
- Train time range: 1700100180 -> 1700160300
- Validation time range: 1700160540 -> 1700170660
- Test time range: 1700180180 -> 1700180660

## Test Metrics
- Accuracy: 0.6667
- Precision: 0.2500
- Recall: 1.0000
- F1: 0.4000
- ROC-AUC: 1.0000
- PR-AUC: 1.0000
- Balanced accuracy: 0.8125
- Specificity: 0.6250
- False positive rate: 0.3750
- False negative rate: 0.0000
- ETA MAE (s): 239.01
- ETA RMSE (s): 239.01
- ETA median AE (s): 239.01
- ETA p90 AE (s): 239.01

## Baselines
- Always-negative recall: 0.0000
- Always-negative PR-AUC: 0.1111
- Percentile-rule recall: 1.0000
- Percentile-rule PR-AUC: 0.2000

## Thresholds
- Execute threshold: 0.45
- Strong threshold: 0.85
- Validation precision @ execute: 0.0000
- Validation recall @ execute: 0.0000
- Validation precision @ strong: 0.0000
- Test precision @ strong: 0.0000
- Test recall @ strong: 0.0000

## Calibration
- Validation ECE: 0.4546
- Test ECE: 0.3478
- Test Brier score: 0.2128
- Test log loss: 0.6185
- High-confidence status: insufficient_samples
- High-confidence weighted gap: 0.0000

## Validation Partition
- Mode: chronological
- Calibration samples: 6
- Selection samples: 6
- Calibration range: 1700160540 -> 1700170300
- Selection range: 1700170360 -> 1700170660
- Calibration class diversity: 1
- Selection class diversity: 1

## Temporal Audit
- Windows: 3
- Min PR-AUC across windows: 0.0000
- Avg PR-AUC across windows: 0.3333
- Min recall across windows: 0.0000

## Feature Monitoring
- Feature count: 25
- Mean abs runtime z-score (test): 0.5390
- Max train variance: 8.5646
- Min train variance: 0.0018

## Label Audit
- Labeling method: episode_take_profit_time_barrier
- Timeout only: True
- Timeout windows without future episode: 76
- Timeout windows with only sub-threshold episodes: 0
- Positive entry bucket lt_0_30: 5
- Positive entry bucket 0_30_to_0_50: 0
- Positive entry bucket 0_50_to_1_00: 0
- Positive entry bucket 1_00_to_2_00: 0
- Positive entry bucket ge_2_00: 0
- Timeout peak bucket 0_80_to_1_00: 0
- Future total spread p50: 1.2500
- Future total spread p90: 1.2660

## Subgroups
- Entry buckets analysed: 4
- Volatility buckets analysed: 4

## Runtime Gates
- Model status: trained
- Artifact version: arbml-lstm-v3-seed42-t1773181599-dd9e49aca
- Execute threshold: 0.45
- Strong threshold: 0.85

## Residual Risks
- A auditoria offline mede janelas cronológicas, mas não substitui monitoramento contínuo de drift em produção.
- A cabeça de ETA é mais sensível a mudança de regime do que a cabeça de classificação.
- O bootstrap do servidor continua caro porque discovery e feeds não fazem parte desta refatoração de ML.

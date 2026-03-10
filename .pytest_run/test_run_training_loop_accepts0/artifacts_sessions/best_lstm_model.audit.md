# ArbML LSTM Audit

## Findings
- Alto: o modelo não superou o baseline sempre-negativo em recall.
- Alto: o modelo colapsou para zero previsões positivas no threshold operacional.
- Médio: o threshold de STRONG_EXECUTE não atingiu a precisão mínima esperada para sinal forte.

## Dataset
- Samples: 168
- Positive samples: 0
- Negative samples: 168
- Pair count: 8
- Session count: 4
- Block count: 8
- Blocks used: 8
- Skipped short blocks: 0
- Cross-block windows: 0
- Cross-session windows: 0
- Feature abs sum: 7043.9321

## Split Integrity
- Train pairs: 4
- Validation pairs: 2
- Test pairs: 2
- Train blocks: 4
- Validation blocks: 2
- Test blocks: 2
- Unique overlapping pairs: 0
- Pairwise overlap sum: 0
- Unique pair overlap rate: 0.0000
- Embargo samples: 0
- Embargo time sec: 240
- Global temporal order: True
- Purged temporal separation ok: True
- Min cross-split gap sec: 98460.0
- Split mode fallback reason: 
- Train time range: 1700900045 -> 1701001345
- Validation time range: 1701100045 -> 1701101345
- Test time range: 1701200045 -> 1701201345

## Test Metrics
- Accuracy: 1.0000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- ROC-AUC: 0.0000
- PR-AUC: 0.0000
- Balanced accuracy: 0.5000
- Specificity: 1.0000
- False positive rate: 0.0000
- False negative rate: 0.0000
- ETA MAE (s): 0.00
- ETA RMSE (s): 0.00
- ETA median AE (s): 0.00
- ETA p90 AE (s): 0.00

## Baselines
- Always-negative recall: 0.0000
- Always-negative PR-AUC: 0.0000
- Percentile-rule recall: 0.0000
- Percentile-rule PR-AUC: 0.0000

## Thresholds
- Execute threshold: 0.30
- Strong threshold: 0.85
- Validation precision @ execute: 0.0000
- Validation recall @ execute: 0.0000
- Validation precision @ strong: 0.0000
- Test precision @ strong: 0.0000
- Test recall @ strong: 0.0000

## Calibration
- Validation ECE: 0.0734
- Test ECE: 0.0736
- Test Brier score: 0.0055
- Test log loss: 0.0765
- High-confidence status: insufficient_samples
- High-confidence weighted gap: 0.0000

## Validation Partition
- Mode: chronological
- Calibration samples: 21
- Selection samples: 21
- Calibration range: 1701100045 -> 1701100345
- Selection range: 1701101045 -> 1701101345
- Calibration class diversity: 1
- Selection class diversity: 1

## Temporal Audit
- Windows: 4
- Min PR-AUC across windows: 0.0000
- Avg PR-AUC across windows: 0.0000
- Min recall across windows: 0.0000

## Feature Monitoring
- Feature count: 25
- Mean abs runtime z-score (test): 0.0055
- Max train variance: 1.5719
- Min train variance: 0.0000

## Label Audit
- Labeling method: episode_take_profit_time_barrier
- Timeout only: True
- Timeout windows without future episode: 168
- Timeout windows with only sub-threshold episodes: 0
- Positive entry bucket lt_0_30: 0
- Positive entry bucket 0_30_to_0_50: 0
- Positive entry bucket 0_50_to_1_00: 0
- Positive entry bucket 1_00_to_2_00: 0
- Positive entry bucket ge_2_00: 0
- Timeout peak bucket 0_80_to_1_00: 0
- Future total spread p50: 0.0000
- Future total spread p90: 0.0000

## Subgroups
- Entry buckets analysed: 4
- Volatility buckets analysed: 4

## Runtime Gates
- Model status: trained
- Artifact version: arbml-lstm-v3-seed42-t1773181600-d0938aa1b
- Execute threshold: 0.30
- Strong threshold: 0.85

## Residual Risks
- A auditoria offline mede janelas cronológicas, mas não substitui monitoramento contínuo de drift em produção.
- A cabeça de ETA é mais sensível a mudança de regime do que a cabeça de classificação.
- O bootstrap do servidor continua caro porque discovery e feeds não fazem parte desta refatoração de ML.

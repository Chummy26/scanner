# ArbML LSTM Audit

## Findings
- Médio: a calibração ficou acima do guardrail de ECE e exige threshold mais conservador.
- Médio: o threshold de STRONG_EXECUTE não atingiu a precisão mínima esperada para sinal forte.

## Dataset
- Samples: 450388
- Positive samples: 216241
- Negative samples: 234147
- Pair count: 4672
- Feature abs sum: 20318514.0000

## Split Integrity
- Train pairs: 4398
- Validation pairs: 4302
- Test pairs: 4523
- Unique overlapping pairs: 4382
- Pairwise overlap sum: 12720
- Unique pair overlap rate: 0.9379
- Embargo samples: 14
- Global temporal order: True
- Train time range: 1772876585 -> 1772883755
- Validation time range: 1772883755 -> 1772884305
- Test time range: 1772884305 -> 1772884840

## Test Metrics
- Accuracy: 0.7337
- Precision: 0.4912
- Recall: 0.7887
- F1: 0.6054
- ROC-AUC: 0.8335
- PR-AUC: 0.6465
- Balanced accuracy: 0.7516
- Specificity: 0.7145
- False positive rate: 0.2855
- False negative rate: 0.2113
- ETA MAE (s): 191.19
- ETA RMSE (s): 268.64
- ETA median AE (s): 127.36
- ETA p90 AE (s): 452.83

## Baselines
- Always-negative recall: 0.0000
- Always-negative PR-AUC: 0.2589
- Percentile-rule recall: 0.0734
- Percentile-rule PR-AUC: 0.2473

## Thresholds
- Execute threshold: 0.45
- Strong threshold: 0.85
- Validation precision @ execute: 0.7237
- Validation recall @ execute: 0.7411
- Validation precision @ strong: 0.9065
- Test precision @ strong: 0.6906
- Test recall @ strong: 0.4266

## Calibration
- Validation ECE: 0.0124
- Test ECE: 0.1687
- Test Brier score: 0.1745
- Test log loss: 0.5461

## Validation Partition
- Mode: chronological
- Calibration samples: 33777
- Selection samples: 33777
- Calibration range: 1772883755 -> 1772884018
- Selection range: 1772884018 -> 1772884305
- Calibration class diversity: 2
- Selection class diversity: 2

## Temporal Audit
- Windows: 4
- Min PR-AUC across windows: 0.5218
- Avg PR-AUC across windows: 0.6586
- Min recall across windows: 0.7401

## Feature Monitoring
- Feature count: 10
- Mean abs runtime z-score (test): 0.0296
- Max train variance: 5.9718
- Min train variance: 0.0113

## Subgroups
- Entry buckets analysed: 4
- Volatility buckets analysed: 4

## Runtime Gates
- Model status: trained
- Artifact version: arbml-lstm-v3-seed42-t1772887600-d0ab7dbc6
- Execute threshold: 0.45
- Strong threshold: 0.85

## Residual Risks
- A auditoria offline mede janelas cronológicas, mas não substitui monitoramento contínuo de drift em produção.
- A cabeça de ETA é mais sensível a mudança de regime do que a cabeça de classificação.
- O bootstrap do servidor continua caro porque discovery e feeds não fazem parte desta refatoração de ML.

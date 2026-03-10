# Runtime Audit

## Overview
- Events: 4
- Alerts: 7
- Samples: 1
- Signals: {'EXECUTE': 1, 'STRONG_EXECUTE': 1}

## Latency
- Inference ms: min=12.400 max=72.500 p99=71.899
- End-to-end ms: min=18.208 max=89.000 p99=88.292

## Alerts
- Alert counts: {'record_frequency_deviation': 1, 'gap_detected': 1, 'invalid_record': 1, 'non_monotonic_timestamp': 1, 'signal_probability_jump': 1, 'inference_latency_high': 1, 'drift_detected': 1}

## Notes
- Este relatório é separado do relatório principal do modelo para evitar poluição do fluxo normal.
- O runner de 2 horas usa este pacote dedicado e não substitui `ml_diagnostic_report.md`.

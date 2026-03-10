# Runtime Audit Package

## Runtime Overview
- Events: 2
- Alerts: 0
- Signals: {'EXECUTE': 1}
- Inference p99: 10.000 ms

## Context Coverage
- Eligible ready pairs: 0
- ready_short: 0
- ready_long_fallback: 0
- insufficient_empirical_context_rate: 0.0000
- Near misses for STRONG_EXECUTE: {}

## Dashboard Validation
- Adjustando occurrences: 0
- Actionable without range: 0
- Strong with eta divergent: 0
- Above-band mismatches: 0
- Range/core mismatches: 0
- Reason/lane/message mismatches: 0

## Collection & Integrity
- Dataset samples: 2
- Cross-block windows: 0
- Shape: [2, 15, 25]

## Offline Training
- Training status: unknown
- Metrics: {'test': {'precision': 0.5, 'recall': 0.4, 'f1': 0.44, 'average_precision': 0.52}}
- Thresholds: {'execute_threshold': 0.45, 'strong_threshold': 0.85}

## Signal Confirmation
- Summary: {'total': 1, 'confirmable_now': 0, 'confirmed': 0, 'not_confirmed': 0, 'awaiting_confirmation': 1}
- By action: {'EXECUTE': {'awaiting_confirmation': 1}}
- By eta alignment: {'unknown': {'awaiting_confirmation': 1}}

## Auxiliary Short-Block Smoke
- Status: completed
- Dataset status: built
- All-negative expected: True

## Legacy Comparison
- Status: not_requested
- Details: {'status': 'not_requested'}

## Notes
- Esta trilha de auditoria é dedicada e isolada do relatório principal do modelo.
- Output dir: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa0\audit`

## Artifacts
- Runtime summary: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa0\audit\summary.json`
- Dataset summary: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa0\audit\dataset_summary.json`
- Training report: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa0\audit\training_report.json`
- Signal confirmations: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa0\audit\signal_confirmations.json`

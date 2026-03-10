# Runtime Audit Package

## Runtime Overview
- Events: 2
- Alerts: 0
- Signals: {'EXECUTE': 1}
- Inference p99: 12.500 ms

## Context Coverage
- Eligible ready pairs: 1
- ready_short: 1
- ready_long_fallback: 0
- insufficient_empirical_context_rate: 0.0000
- Near misses for STRONG_EXECUTE: {'short_ready_strong': 1}

## Dashboard Validation
- Adjustando occurrences: 0
- Actionable without range: 0
- Strong with eta divergent: 0
- Above-band mismatches: 0
- Range/core mismatches: 0
- Reason/lane/message mismatches: 0

## Collection & Integrity
- Dataset samples: 0
- Cross-block windows: 0
- Shape: [0, 15, 25]

## Offline Training
- Training status: skipped
- Metrics: {}
- Thresholds: {}

## Signal Confirmation
- Summary: {'total': 1, 'confirmable_now': 1, 'confirmed': 1, 'not_confirmed': 0, 'awaiting_confirmation': 0}
- By action: {'EXECUTE': {'confirmed': 1}}
- By eta alignment: {'aligned': {'confirmed': 1}}

## Auxiliary Short-Block Smoke
- Status: completed
- Dataset status: built
- All-negative expected: True

## Legacy Comparison
- Status: not_requested
- Details: {'status': 'not_requested'}

## Notes
- Esta trilha de auditoria é dedicada e isolada do relatório principal do modelo.
- Output dir: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa2\audit`

## Artifacts
- Runtime summary: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa2\audit\summary.json`
- Dataset summary: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa2\audit\dataset_summary.json`
- Training report: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa2\audit\training_report.json`
- Signal confirmations: `C:\Users\nicoolas\Pictures\scanner\scanner\.pytest_run\test_finalize_runtime_audit_pa2\audit\signal_confirmations.json`

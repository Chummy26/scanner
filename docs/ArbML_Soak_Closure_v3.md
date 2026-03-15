# ArbML Soak Closure — Plano v3.1

## Princípio: Zero surpresas nas 24h

O soak valida INFRAESTRUTURA, não o modelo. O modelo já foi treinado e validado
pelo baseline 8h. As 24h confirmam que o sistema completo (scanner + snapshots
automáticos + auto-retrain + signal gates) funciona autonomamente.

```
PRÉ-REQUISITO: Baseline v3 completo (modelo deployed + AUC + ETA MAE medidos)

BLOCO A (~10 min): verificar artefactos do baseline + código + pre-soak checks
BLOCO B (se A falhar): corrigir antes de iniciar
BLOCO C (24h+): confirmação live com Stage 1 + Stage 2
BLOCO D (5 min): validação final + v3 acceptance
```

---

## Pré-requisito: Baseline v3 completo

O soak NÃO deve ser iniciado até que:
- `best_lstm_model.pth` existe com `feature_contract_version == "v3_exit_aware_40"`
- `best_lstm_model.meta.json` contém `test_metrics.roc_auc > 0.55`
- `best_lstm_model.report.json` contém métricas de treino válidas
- O baseline foi treinado com o pipeline endurecido (204 tests pass)

O baseline 8h já valida: certificação completa (12 gates), labeling audit,
preflight, split integrity, tensor integrity, normalization leakage, positive
floor por fold, degenerate features, episode completeness v3. O soak não
re-executa nada disto offline — apenas confirma que os artefactos existem.

---

## Bloco 0 — Código já aplicado (verificação apenas)

Todos os fixes abaixo foram implementados nos commits anteriores. O soak
apenas verifica que estão activos:

- ✅ soak_runbook.py usa FEATURE_NAMES (não V2_MULTISCALE_FEATURE_NAMES)
- ✅ ml_analyzer.py tem context_strength check no EXECUTE gate
- ✅ spread_tracker.py passa cost_estimate_pct ao build_feature_rows
- ✅ spread_engine.py rejeita crossed_book, non_positive_book_px, missing_top_of_book
- ✅ spread_tracker.py faz dedup ts, rejection retrograde, episode abandonment
- ✅ runtime_audit.py força wal_checkpoint + integrity_check no snapshot
- ✅ ml_dataset.py bloqueia tensors NaN/Inf, enforce embargo no split
- ✅ training_certification.py gate_05 estendido (degenerate features, v3/v2 ratio)
- ✅ training_certification.py gate_06 estendido (episode v3 completeness)
- ✅ train_model.py positivo floor 50/fold, normalization leakage guard, pair concentration

Verificação rápida:

```bash
cd backend && python3 -c "
import sys; sys.path.insert(0, 'src')
from spread.soak_runbook import *
from spread.feature_contracts import FEATURE_NAMES, DEFAULT_FEATURE_CONTRACT_VERSION
assert DEFAULT_FEATURE_CONTRACT_VERSION == 'v3_exit_aware_40'
assert len(FEATURE_NAMES) == 40
print('OK — v3 code verified')
" && python3 -m pytest tests/ -x -q --tb=line 2>&1 | tail -3
```

---

## Mapa de Gates — Actualizado para v3 + Hardening

### Stage 1 — 20 gates (soak_runbook)

| # | Gate | Estado v3 |
|---|---|---|
| 1-9 | HTTP + performance | Sem mudança |
| 10 | tracker_low_spread_capture | Sem mudança |
| 11 | dashboard_filters | Sem mudança |
| 12 | data_distribution | Sem mudança |
| 13 | stale_pairs | Sem mudança |
| 14 | cross_exchange_rejection | ✅ Verdict-based (não % rate) |
| 15 | feature_history_contract | ✅ Espera 40 features, usa FEATURE_NAMES |
| 16 | hourly_digest | Sem mudança |
| 17-19 | SKIP OK (runtime_audit) | Sem mudança |
| 20 | disconnects | Sem mudança |

### Stage 2 — 18 gates (soak_runbook)

| # | Gate | Estado v3 |
|---|---|---|
| 1-9 | Base gates | Sem mudança |
| 10 | label_threshold_mode | rolling_pair_percentile |
| 11 | positive_rate_band | ✅ [0.01, 0.20] |
| 12 | label_zero_leakage | Sem mudança |
| 13 | trained_model_auc > 0.55 | Sem mudança |
| 14 | platt_scaling_finite | Sem mudança |
| 15 | inference_probabilities | Sem mudança |
| 16 | feature_contract_version | ✅ Espera v3_exit_aware_40 |
| 17 | training_metadata_mode | Sem mudança |
| 18 | signal_gate_anti_spike | Sem mudança |

### Certification gates (training_certification.py) — durante retrain automático

| Gate | Extensão v3 + Hardening |
|---|---|
| 01 | SQLite integrity (sem mudança) |
| 02 | Intra-block temporal regularity (sem mudança) |
| 03 | Completeness (sem mudança) |
| 04 | Checkpoint stationarity (sem mudança) |
| 05 | ✅ **Estendido:** degenerate features (var ≤ 1e-10 → FAIL), NaN > 5% → FAIL, Inf > 0 → FAIL, zero > 80% v3 features → WARN, v3/v2 variance ratio < 0.10 → WARN |
| 06 | ✅ **Estendido:** episode v3 completeness (< 50% → FAIL, < 80% → WARN) |
| 07 | Book health (sem mudança) |
| 08 | Reconnection stress (sem mudança) |
| 09 | Runtime audit consistency (sem mudança) |
| 10 | Dual-mode preflight — right-censoring reclassificado: > 40% WARN, > 80% FAIL (não mais FAIL por presença) |
| 11 | Runtime audit health (sem mudança) |
| 12 | Entry/exit quality (sem mudança) |

### Training guards (train_model.py) — durante retrain automático

| Guard | Comportamento |
|---|---|
| Tensor NaN/Inf | BLOCK — ValueError com feature indices |
| Positive floor | BLOCK se qualquer fold < 50, WARN se < 100 |
| Normalization leakage | BLOCK se suspicious_global_fit_leakage detected |
| Split overlap | BLOCK se label windows overlap entre splits |
| Embargo gap | WARN se gap < prediction_horizon_sec |
| Pair concentration | WARN se Gini > 0.70 ou top1 > 20% |
| Right-censoring | WARN se > 40%, info no report |

### Timeouts

`max_certification_duration_sec = 1800` em todo o pipeline (certificação full
leva ~18 min medidos). NÃO usar 300s — causa certification_timeout antes de
completar o Gate 10.

---

## BLOCO A — Verificação Pré-Soak (~10 min)

O baseline já validou dados e modelo. O Bloco A verifica apenas que os
artefactos estão prontos e o código está correcto para correr 24h.

### A1. Baseline artefactos existem

```bash
cd backend && python3 -c "
import sys, json, math; sys.path.insert(0, 'src')
from pathlib import Path

meta_path = Path('out/config/best_lstm_model.meta.json')
model_path = Path('out/config/best_lstm_model.pth')

assert model_path.exists(), 'Model .pth not found — run baseline first'
assert meta_path.exists(), 'Model .meta.json not found — run baseline first'

meta = json.load(open(meta_path))
contract = meta.get('feature_contract_version')
auc = float((meta.get('test_metrics') or {}).get('roc_auc', 0))
platt_s = float(meta.get('platt_scale', 0))
platt_b = float(meta.get('platt_bias', 0))
feature_count = len(meta.get('feature_names', []))

print(f'Contract:      {contract}')
print(f'Feature count: {feature_count}')
print(f'AUC:           {auc}')
print(f'Platt:         scale={platt_s}, bias={platt_b}')

assert contract == 'v3_exit_aware_40', f'Wrong contract: {contract}'
assert feature_count == 40, f'Wrong feature count: {feature_count}'
assert auc > 0.55, f'AUC too low: {auc}'
assert math.isfinite(platt_s) and math.isfinite(platt_b), 'Platt non-finite'
print('\nBaseline artefacts OK')
"
```

### A2. DB integrity + freshness

```bash
cd backend && python3 -c "
import sys, sqlite3, time; sys.path.insert(0, 'src')
conn = sqlite3.connect('out/config/tracker_history.sqlite', timeout=30)
now = time.time()

integrity = conn.execute('PRAGMA integrity_check').fetchone()[0]
newest = conn.execute('SELECT MAX(ts) FROM tracker_records').fetchone()[0]
total = conn.execute('SELECT COUNT(*) FROM tracker_records').fetchone()[0]
age_h = (now - newest) / 3600

print(f'Integrity: {integrity}')
print(f'Records:   {total:,}')
print(f'Data age:  {age_h:.1f}h')
print(f'Status:    {\"PASS\" if integrity == \"ok\" and age_h < 1.0 else \"CHECK\"} (need < 1h for soak start)')
conn.close()
"
```

### A3. v3 signal smoke test

```bash
cd backend && python3 -c "
import sys, tempfile; sys.path.insert(0, 'src')
from spread.signal_ledger import SignalLedger
from spread.ml_analyzer import compute_exit_policies, compute_signal_score

context = {'exit_core_range_max': -0.5, 'exit_median': -1.2, 'exit_outer_range_min': -2.5}
policies = compute_exit_policies(current_entry=4.0, context=context, cost_estimate_pct=0.30)
score = compute_signal_score(prob=0.72, net_capture_median=1.5, support_2h=5,
    support_24h=12, context_strength='strong', eta_minutes=30.0, drift_status='stable')
assert len(policies) == 3
assert score > 0
print(f'Exit policies: {len(policies)} OK')
print(f'Signal score:  {score:.4f} OK')

db = tempfile.mktemp(suffix='.sqlite')
ledger = SignalLedger(db)
sid = ledger.record_signal(pair_key='TEST|a|b|c|d', signal_action='EXECUTE',
    policy_name='median', horizon_sec=14400, entry_at_signal=4.0,
    exit_at_signal=-1.2, prob_at_signal=0.72, eta_at_signal=1800.0)
assert sid is not None
assert ledger.check_cooldown('TEST|a|b|c|d', 'median', 14400)
import os; os.unlink(db)
print('Signal ledger: OK')
print('\nAll v3 smoke tests PASS')
"
```

### A4. Tests pass

```bash
cd backend && python3 -m pytest tests/ -x -q --tb=short
# Deve: 204+ passed
```

**PASS se:** A1-A4 todos verdes. NÃO re-correr certificação ou treino offline.

---

## BLOCO B — Correcções (se A falhar)

| Falha | Correcção |
|---|---|
| Model .pth missing | Baseline não completou — re-correr baseline 8h |
| Contract != v3_exit_aware_40 | Baseline treinou com contrato errado — investigar |
| AUC < 0.55 | Baseline produziu modelo fraco — investigar dados |
| DB corrompida | Restaurar backup + re-correr baseline |
| Data age > 1h | Scanner não está activo — restart antes do soak |
| Tests fail | Código regressou — investigar commit |

**Regra:** NÃO iniciar Bloco C até Bloco A todo verde.

---

## BLOCO C — Run Final 24h+

### C1. Limpar estado de soak anterior (não modelo!)

```bash
# NÃO apagar best_lstm_model.pth/meta.json — o modelo do baseline fica!
rm -f backend/out/snapshots/snapshot_test_slot_*.sqlite
rm -f backend/out/config/auto_retrain.state.json
# Limpar soak runs anteriores se existirem
rm -rf backend/out/arbml_v2_soak_*
```

### C2. Subir servidor (se não estiver já activo do baseline)

```bash
# Verificar se servidor do baseline ainda está activo
curl -s http://127.0.0.1:8000/api/v1/system/health | python3 -m json.tool
# Se não responder, subir:
python3 backend/src/server.py --host 127.0.0.1 --port 8000 &
```

### C3. Lançar soak runner

```bash
python3 backend/tests/run_arbml_v2_soak.py &
```

### C4. Timeline esperada

```
T+0h:       Servidor activo com modelo v3 do baseline.
            Stage 1 começa.
T+3h:       Stage 1 avalia. Todos os gates devem PASS.
            feature_history_contract espera 40 features.
T+3h:       Stage 2 começa. Primeiro slot UTC → snapshot 1.
            Snapshot usa create_sqlite_snapshot (wal_checkpoint + integrity_check).
            Certificação full do snapshot (~18 min com max_certification_duration_sec=1800).
T+11h:      Segundo slot UTC → snapshot 2.
T+19h:      Terceiro slot UTC → snapshot 3.
            pass_snapshots >= 3 → should_retrain = "first_training"
T+19-19.5h: Retrain completa (~25 min com pipeline optimizado).
            Certificação full (~18 min) + preflight (~3 min) + GPU (~3-5 min).
            Modelo v3 re-deployed.
            signal_gate_anti_spike avalia (30min antes/depois deploy).
T+24h+:     Window SLA avaliado (H1-H4 vs H20-H24).
T+27h:      Stage 2 avaliação final.
```

### C5. Monitoramento (3 checkpoints humanos)

```
T+4h:  Stage 1 PASS? RSS estável? Snapshot 1 certificado?
       feature_history_contract PASS com 40 features?
       Gate 05 (degenerate features) PASS? Gate 06 (episode completeness) PASS?

T+12h: RSS flat? Snapshots 1+2 no manifest? Verdict healthy/degraded?
       Right-censoring fraction no report (deve ser < 40%)?
       Pair concentration warning no report?

T+20h: Retrain triggered? Pipeline completou em < 30 min?
       Modelo v3_exit_aware_40 re-criado? AUC > 0.55?
       signal_anti_spike PASS?
       Exit policies presentes no output? Signal score > 0?
       Normalization leakage guard PASS?
       Positive floor (50/fold) PASS?
```

---

## BLOCO D — Validação Final (~5 min)

```bash
cat backend/out/arbml_v2_soak_*/stage2_result.json | python3 -m json.tool
# Se ok=true → SOAK COMPLETO

# V3 + hardening validation:
cd backend && python3 -c "
import sys, json; sys.path.insert(0, 'src')
meta = json.load(open('out/config/best_lstm_model.meta.json'))
print(f'Feature contract: {meta.get(\"feature_contract_version\")}')
print(f'Feature count:    {meta.get(\"feature_count\", len(meta.get(\"feature_names\", [])))}')
print(f'AUC:              {meta.get(\"test_metrics\", {}).get(\"roc_auc\")}')

# Verify model was re-trained during soak (not just baseline model)
trained_at = meta.get('trained_at_utc', '')
print(f'Trained at:       {trained_at}')

assert meta.get('feature_contract_version') == 'v3_exit_aware_40', 'Wrong contract!'
assert len(meta.get('feature_names', [])) == 40, 'Wrong feature count!'
print('\nv3 soak model verified')
"
```

---

## Cenários de FAIL

| Cenário | Impacto | Acção |
|---|---|---|
| feature_count = 25 (not 40) | feature_history_contract FAIL | Código regressou — verificar imports |
| contract = v2_multiscale_25 | feature_contract_version FAIL | DEFAULT_FEATURE_CONTRACT não é v3 |
| positive_rate fora [0.01, 0.20] | labeling gate FAIL | Threshold ou dados mudaram |
| AUC < 0.55 no retrain | trained_model_auc FAIL | Dados novos degradaram — investigar |
| Exit policies all zero | v3 functionality broken | normalize_episode não extrai fields |
| Signal score = 0 | Ranking inútil | viability ou prob = 0 |
| Retrain > 30 min | Pipeline slower than expected | Verificar max_certification_duration_sec=1800 |
| Gate 05 FAIL (degenerate features) | Features NaN/zero no snapshot | Episódios v3 não acumularam |
| Gate 06 FAIL (episode completeness) | < 50% episódios com campos v3 | Dados pré-v3 dominam snapshot |
| Tensor NaN/Inf FAIL | Feature computation bug | Investigar quais features no error msg |
| Positive floor FAIL (< 50/fold) | Dados insuficientes no snapshot | Threshold alto ou snapshot pequeno |
| Normalization leakage | Scaler fit em dados errados | Bug no pipeline — investigar |
| Split overlap | Embargo insuficiente | Dados concentrados — split impossível |
| Right-censoring > 80% | Sessões muito curtas | Verificar gap_threshold_sec |
| context_strength weak → EXECUTE | Gate bug | context_strength check missing |
| RSS > 2x warmup | Memory leak | Investigar após soak |
| signal_anti_spike FAIL | Spike de sinais pós-deploy | Modelo emitindo demasiados EXECUTE |
| certification_timeout | max_certification_duration_sec muito baixo | Deve ser 1800, não 300 |

---

## Sampling: estado actual

```
✅ Stride=5:                Reduz overlap 93% → 67%
✅ Right-censoring:         Thresholded (> 40% WARN, > 80% FAIL), não binary
✅ Embargo no split:        Enforced — ValueError se overlap
✅ FocalLoss:               Equilibra classes
✅ Tensor integrity:        NaN/Inf bloqueados
✅ Degenerate features:     Gate 05 estendido
✅ Episode completeness:    Gate 06 estendido
✅ Positive floor:          50/fold minimum
✅ Normalization guard:     Global fit leakage detectado
✅ Pair concentration:      Gini + top1 reportados (WARN)

❌ max_samples_per_pair (S2): NÃO implementado
   → Pares dominantes podem contribuir 80%+ dos samples
   → MITIGAÇÃO: stride=5 + pair concentration warning
   → PRIORIDADE: Implementar antes do 2º retrain

❌ Sampling diagnostics (S5): PARCIALMENTE implementado
   → pair_sample_counts no bundle summary ✅
   → Gini + top1_pair_fraction no training report ✅
   → Per-feature summary statistics: pendente
```

---

## Checklist Completa

```
PRÉ-REQUISITO:
□ Baseline v3 completo: model .pth + .meta.json com AUC > 0.55
□ 204+ tests passam

BLOCO A (10 min):
□ A1: Baseline artefactos existem (contract=v3_exit_aware_40, AUC > 0.55, Platt finite)
□ A2: DB integrity OK + data age < 1h
□ A3: Signal ledger + exit policies smoke test
□ A4: All tests pass (204+)

BLOCO C (24h):
□ C1: Estado de soak limpo (modelo do baseline preservado)
□ C2: Servidor activo com modelo v3
□ C3: Soak runner lançado
□ C5-T+4h:  Stage 1 PASS, 40 features, snapshot 1 certificado
            Gate 05 + Gate 06 estendidos PASS
□ C5-T+12h: RSS flat, snapshots 1+2 no manifest
□ C5-T+20h: Retrain v3 completou (< 30 min, max_cert=1800s)
            Gate 05/06/10 estendidos PASS no retrain
            Positive floor 50/fold PASS
            Normalization leakage guard PASS
            signal_anti_spike PASS

BLOCO D (5 min):
□ Stage 2 ok=true
□ Modelo re-trained durante soak (trained_at > soak_start)
□ Contract = v3_exit_aware_40, features = 40
□ → SOAK v3 COMPLETO
```

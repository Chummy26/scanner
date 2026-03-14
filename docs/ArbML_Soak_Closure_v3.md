# ArbML Soak Closure — Plano v3

## Princípio: Zero surpresas nas 24h

Validar OFFLINE tudo que pode falhar antes de iniciar as 24h live.
As 24h confirmam estabilidade temporal, não descobrem bugs.

```
BLOCO A (agora, ~1h): validar offline + corrigir código
BLOCO B (se A falhar): corrigir antes de subir servidor
BLOCO C (24h): confirmação live
BLOCO D (5 min): validação final
```

---

## BLOCO 0 — Correcções de Código (ANTES de tudo)

### 0.1 soak_runbook.py — V2 → V3 references (6 linhas)

O soak_runbook.py tem 6 referências a V2_MULTISCALE_FEATURE_NAMES que vão
causar FAIL imediato com o feature contract v3.

```python
# Linha 27 — import
# ANTES:
from .feature_contracts import DEFAULT_FEATURE_CONTRACT_VERSION, V2_MULTISCALE_FEATURE_NAMES
# DEPOIS:
from .feature_contracts import DEFAULT_FEATURE_CONTRACT_VERSION, FEATURE_NAMES

# Linhas 544, 574 — feature harness
# ANTES:
rows = tracker.get_feature_history(*pair, limit=int(limit), feature_names=list(V2_MULTISCALE_FEATURE_NAMES))
# DEPOIS:
rows = tracker.get_feature_history(*pair, limit=int(limit), feature_names=list(FEATURE_NAMES))

# Linha 549 — feature enumeration
# ANTES:
for index, name in enumerate(V2_MULTISCALE_FEATURE_NAMES)
# DEPOIS:
for index, name in enumerate(FEATURE_NAMES)

# Linha 868 — bundle feature_contract check
# ANTES:
"feature_contract_version": DEFAULT_FEATURE_CONTRACT_VERSION if len(bundle.feature_names) == len(V2_MULTISCALE_FEATURE_NAMES) else "",
# DEPOIS:
"feature_contract_version": DEFAULT_FEATURE_CONTRACT_VERSION if len(bundle.feature_names) == len(FEATURE_NAMES) else "",

# Linha 1172 — feature_history_contract gate
# ANTES:
and int(feature_harness.get("feature_count", 0)) == len(V2_MULTISCALE_FEATURE_NAMES)
# DEPOIS:
and int(feature_harness.get("feature_count", 0)) == len(FEATURE_NAMES)

# Linha 1176 — expected message
# ANTES:
expected="25 features, cache invalidates, multiscale variance > 0",
# DEPOIS:
expected=f"{len(FEATURE_NAMES)} features, cache invalidates, multiscale variance > 0",
```

### 0.2 ml_analyzer.py — EXECUTE gate missing context_strength check

Na cadeia de elif antes de EXECUTE, adicionar:

```python
elif str(context["context_strength"]) not in ("normal", "strong"):
    signal_reason_code = "context_strength_weak"
    signal_reason = "Contexto fraco — recorrência insuficiente para execução."
```

### 0.3 spread_tracker.py — cost_estimate_pct threading

Linha ~2053, adicionar parâmetro:

```python
rows = build_feature_rows(
    recent_records,
    feature_names=list(feature_names or []),
    episodes=episodes,
    cost_estimate_pct=float(getattr(self.config, "default_cost_estimate_pct", 0.30)),
)
```

### 0.4 Verificar fix aplicado

```bash
cd backend && python3 -c "
import sys; sys.path.insert(0, 'src')
from spread.soak_runbook import *  # deve importar sem erro
from spread.feature_contracts import FEATURE_NAMES, DEFAULT_FEATURE_CONTRACT_VERSION
print(f'Contract: {DEFAULT_FEATURE_CONTRACT_VERSION}')
print(f'Features: {len(FEATURE_NAMES)}')
assert len(FEATURE_NAMES) == 40, f'Expected 40, got {len(FEATURE_NAMES)}'
assert DEFAULT_FEATURE_CONTRACT_VERSION == 'v3_exit_aware_40'
print('OK — imports clean')
"
```

---

## Mapa de Gates — Actualizado para v3

### Stage 1 — 20 gates

| # | Gate | O que mudou no v3 |
|---|---|---|
| 1-9 | HTTP + performance | Sem mudança |
| 10 | tracker_low_spread_capture | Sem mudança |
| 11 | dashboard_filters | Sem mudança |
| 12 | data_distribution | Sem mudança |
| 13 | stale_pairs | Sem mudança |
| 14 | cross_exchange_rejection | ✅ Já verdict-based |
| 15 | feature_history_contract | ⚠️ MUDAR: 25 → 40 features (fix 0.1) |
| 16 | hourly_digest | Sem mudança |
| 17-19 | SKIP OK (runtime_audit) | Sem mudança |
| 20 | disconnects | Sem mudança |

### Stage 2 — 18 gates

| # | Gate | O que mudou no v3 |
|---|---|---|
| 1-9 | Base gates | Sem mudança |
| 10 | label_threshold_mode | Sem mudança (rolling_pair_percentile) |
| 11 | positive_rate_band | ✅ Já [0.01, 0.20] no código |
| 12 | label_zero_leakage | Sem mudança |
| 13 | trained_model_auc > 0.55 | Sem mudança (threshold mantido) |
| 14 | platt_scaling_finite | Sem mudança |
| 15 | inference_probabilities | Sem mudança |
| 16 | feature_contract_version | ⚠️ Espera v3_exit_aware_40 agora |
| 17 | training_metadata_mode | Sem mudança |
| 18 | signal_gate_anti_spike | Sem mudança |

---

## BLOCO A — Validação Offline (~30 min com pipeline optimizado)

### A1. Certificação de snapshot (~2 min)

```bash
cd backend && python3 -c "
import sys; sys.path.insert(0, 'src')
from spread.runtime_audit import create_sqlite_snapshot
from spread.train_model import certify_data_for_training
from spread.auto_retrain import certification_verdict_view
from pathlib import Path

source = Path('out/config/tracker_history.sqlite')

for slot in ['test_slot_a', 'test_slot_b', 'test_slot_c']:
    target = Path(f'out/snapshots/snapshot_{slot}.sqlite')
    if target.exists():
        target.unlink()
    create_sqlite_snapshot(source, target)
    cert = certify_data_for_training(state_file=target, certification_mode='quick')
    verdict = certification_verdict_view(cert)
    print(f'{slot}: {cert[\"verdict\"]} (view: {verdict})')
    failures = cert.get('failure_reasons', [])
    warnings = cert.get('warnings', [])
    if failures:
        print(f'  FAILURES: {failures}')
    if warnings:
        print(f'  WARNINGS: {warnings[:5]}')
"
```

**PASS se:** verdict = CERTIFIED ou CERTIFIED_WITH_WARNINGS para os 3.

### A2. Labeling audit (~3 min)

```bash
cd backend && python3 -c "
import sys; sys.path.insert(0, 'src')
from spread.soak_runbook import audit_snapshot_labeling
from pathlib import Path

result = audit_snapshot_labeling(Path('out/snapshots/snapshot_test_slot_a.sqlite'))
print(f'threshold_mode: {result.get(\"label_threshold_mode\")}')
print(f'positive_rate:  {result.get(\"positive_rate\")}')
print(f'zero_leakage:   {result.get(\"zero_leakage_ok\")}')
print(f'total_samples:  {result.get(\"total_samples\")}')
print(f'contract:       {result.get(\"feature_contract_version\")}')

ok_mode = result.get('label_threshold_mode') == 'rolling_pair_percentile'
ok_rate = 0.01 <= float(result.get('positive_rate', 0)) <= 0.20
ok_leak = bool(result.get('zero_leakage_ok'))
ok_contract = result.get('feature_contract_version') == 'v3_exit_aware_40'
print(f'\n--- Gate check ---')
print(f'label_threshold_mode:     {\"PASS\" if ok_mode else \"FAIL\"}')
print(f'positive_rate_band:       {\"PASS\" if ok_rate else \"FAIL\"} ({result.get(\"positive_rate\")})')
print(f'label_zero_leakage:       {\"PASS\" if ok_leak else \"FAIL\"}')
print(f'feature_contract_version: {\"PASS\" if ok_contract else \"FAIL\"} ({result.get(\"feature_contract_version\")})')
"
```

**PASS se:** mode=rolling_pair_percentile, rate ∈ [0.01, 0.20], leakage=True, contract=v3_exit_aware_40

### A3. Treino offline (~5-10 min com pipeline optimizado)

```bash
cd backend && python3 -c "
import sys, json, math; sys.path.insert(0, 'src')
from spread.train_model import run_clean_training_cycle
from pathlib import Path

result = run_clean_training_cycle(
    base_path=Path('out'),
    artifact_dir=Path('/tmp/arbml_test_model'),
)
print(f'status:           {result.get(\"status\")}')
print(f'auc:              {result.get(\"auc\")}')
print(f'platt_scale:      {result.get(\"platt_scale\")}')
print(f'platt_bias:       {result.get(\"platt_bias\")}')
print(f'feature_contract: {result.get(\"feature_contract_version\")}')
print(f'label_mode:       {result.get(\"label_threshold_mode\")}')
print(f'samples:          {result.get(\"dataset_samples\")}')
print(f'feature_count:    {result.get(\"feature_count\", \"?\")}')

ok_auc = float(result.get('auc', 0)) > 0.55
ok_platt = math.isfinite(float(result.get('platt_scale', 0))) and math.isfinite(float(result.get('platt_bias', 0)))
ok_contract = result.get('feature_contract_version') == 'v3_exit_aware_40'
ok_mode = result.get('label_threshold_mode') == 'rolling_pair_percentile'
print(f'\n--- Gate check ---')
print(f'trained_model_auc > 0.55:   {\"PASS\" if ok_auc else \"FAIL\"} ({result.get(\"auc\")})')
print(f'platt_scaling_finite:       {\"PASS\" if ok_platt else \"FAIL\"}')
print(f'feature_contract_version:   {\"PASS\" if ok_contract else \"FAIL\"} ({result.get(\"feature_contract_version\")})')
print(f'training_metadata_mode:     {\"PASS\" if ok_mode else \"FAIL\"} ({result.get(\"label_threshold_mode\")})')

# ESTE É O BASELINE V3 — guardar números!
print(f'\n=== BASELINE V3 ===')
print(f'AUC:              {result.get(\"auc\")}')
print(f'ETA MAE:          {result.get(\"eta_mae_seconds\", \"?\")}s')
print(f'Positive rate:    {result.get(\"positive_rate\", \"?\")}')
print(f'Samples:          {result.get(\"dataset_samples\")}')
print(f'Features:         40 (v3_exit_aware_40)')
print(f'Pipeline time:    check wall clock')
"
```

**PASS se:** AUC > 0.55, Platt finite, contract=v3_exit_aware_40, mode=rolling_pair_percentile
**GUARDAR:** Este é o BASELINE V3. Anotar AUC e ETA MAE para referência futura.

### A4. DB integrity + v3 config check (~1 min)

```bash
cd backend && python3 -c "
import sys, time, sqlite3; sys.path.insert(0, 'src')
from spread.models import SpreadConfig

# DB check
db = 'out/config/tracker_history.sqlite'
conn = sqlite3.connect(db, timeout=30)
now = time.time()

oldest = conn.execute('SELECT MIN(ts) FROM tracker_records').fetchone()[0]
newest = conn.execute('SELECT MAX(ts) FROM tracker_records').fetchone()[0]
total = conn.execute('SELECT COUNT(*) FROM tracker_records').fetchone()[0]
episodes = conn.execute('SELECT COUNT(*) FROM tracker_pair_episodes').fetchone()[0]
pairs = conn.execute('SELECT COUNT(DISTINCT pair_id) FROM tracker_records').fetchone()[0]

age_h = (now - oldest) / 3600
span_h = (newest - oldest) / 3600
print(f'records:    {total:,}')
print(f'episodes:   {episodes:,}')
print(f'pairs:      {pairs:,}')
print(f'data span:  {span_h:.1f}h')
print(f'oldest age: {age_h:.1f}h')
print(f'sqlite_retains_gt_8h: {\"PASS\" if age_h > 8.0 else \"FAIL\"}')
conn.close()

# Config invariant check
config = SpreadConfig()
print(f'\n--- Config v3 ---')
print(f'min_total_spread_pct:     {config.min_total_spread_pct}')
print(f'min_net_capture_pct:      {config.min_net_capture_pct}')
print(f'default_cost_estimate_pct: {config.default_cost_estimate_pct}')
print(f'label_cost_floor_pct:     {config.label_cost_floor_pct}')
invariant = config.min_total_spread_pct >= config.min_net_capture_pct + config.default_cost_estimate_pct
print(f'invariant (total >= net + cost): {\"PASS\" if invariant else \"FAIL\"} ({config.min_total_spread_pct} >= {config.min_net_capture_pct} + {config.default_cost_estimate_pct})')
"
```

### A5. Signal Ledger + Exit Policies smoke test (~1 min)

```bash
cd backend && python3 -c "
import sys, tempfile; sys.path.insert(0, 'src')
from spread.signal_ledger import SignalLedger
from spread.ml_analyzer import compute_exit_policies, compute_signal_score, ExitPolicyResult

# Ledger: create, insert, check cooldown, resolve
db = tempfile.mktemp(suffix='.sqlite')
ledger = SignalLedger(db)

# Mock context for exit policies
context = {
    'exit_core_range_max': -0.5,
    'exit_median': -1.2,
    'exit_outer_range_min': -2.5,
}
policies = compute_exit_policies(current_entry=4.0, context=context, cost_estimate_pct=0.30)
print(f'Exit policies: {len(policies)}')
for p in policies:
    print(f'  {p.name}: exit={p.exit_target}, capture={p.capture_gross}, net={p.net_capture}, min_entry={p.min_entry_required}')

# Verify formulas: capture = entry + exit (NO abs)
for p in policies:
    expected_capture = 4.0 + p.exit_target
    expected_net = expected_capture - 0.30
    expected_min = max(0.0, 0.30 - p.exit_target)
    assert abs(p.capture_gross - round(expected_capture, 4)) < 0.001, f'{p.name}: capture mismatch'
    assert abs(p.net_capture - round(expected_net, 4)) < 0.001, f'{p.name}: net mismatch'
    assert abs(p.min_entry_required - round(expected_min, 4)) < 0.001, f'{p.name}: min_entry mismatch'

# Signal score
score = compute_signal_score(
    prob=0.72, net_capture_median=1.5, support_2h=5, support_24h=12,
    context_strength='strong', eta_minutes=30.0, drift_status='stable',
)
print(f'Signal score: {score:.4f}')
assert score > 0.0, 'Score should be positive'

# Ledger cooldown
signal_id = ledger.record_signal(
    pair_key='TEST|a|b|c|d', signal_action='EXECUTE', policy_name='median',
    horizon_sec=14400, entry_at_signal=4.0, exit_at_signal=-1.2,
    prob_at_signal=0.72, eta_at_signal=1800.0,
)
assert signal_id is not None, 'Should insert'
assert ledger.check_cooldown('TEST|a|b|c|d', 'median', 14400), 'Cooldown should be active'

pending = ledger.pending_signals()
assert len(pending) >= 1, 'Should have pending signal'

ledger.resolve(signal_id, outcome_status='hit', outcome_exit=-1.1, outcome_total=2.9,
               outcome_duration_sec=1200.0, best_total_in_horizon=3.2,
               label_hit_shallow=1, label_hit_median=1, label_hit_deep=0)

import os; os.unlink(db)
print('\nAll v3 smoke tests PASS')
"
```

---

## BLOCO B — Correcções (se A falhar)

| Falha | Correcção | Ficheiro |
|---|---|---|
| Cert FAIL | Ajustar gate thresholds | training_certification.py |
| positive_rate fora [0.01, 0.20] | Ajustar label_cost_floor / percentile | models.py |
| AUC < 0.55 | Investigar data quality, features | train_model.py |
| Platt non-finite | Bug no treino — overfit ou NaN | train_model.py |
| contract != v3_exit_aware_40 | Feature contract não carregado | feature_contracts.py |
| feature_count != 40 | soak_runbook.py não actualizado | Fix 0.1 |
| Rejection unhealthy | 2+ exchanges OPEN | Investigar conectividade |
| Config invariant FAIL | Thresholds inconsistentes | models.py |
| Exit policy formula wrong | capture != entry + exit | ml_analyzer.py |
| Signal ledger FAIL | Schema ou cooldown bug | signal_ledger.py |

**Regra:** NÃO iniciar Bloco C até Bloco A todo verde.

---

## BLOCO C — Run Final 24h

### C1. Limpar estado anterior

```bash
kill <server_pid> <stage2_pid>
sleep 5
sqlite3 backend/out/config/tracker_history.sqlite "PRAGMA wal_checkpoint(TRUNCATE)"
rm -f backend/out/snapshots/snapshot_test_slot_*.sqlite
rm -f backend/out/config/best_lstm_model.pth
rm -f backend/out/config/best_lstm_model.meta.json
rm -f backend/out/config/auto_retrain.state.json
```

### C2. Subir servidor com código limpo

```bash
git log --oneline -5  # confirmar fix 0.1-0.3 aplicados
python3 -m pytest backend/tests/ -x -q
# Subir servidor
```

### C3. Timeline esperada (mais rápida com pipeline optimizado)

```
T+0h:     Servidor sobe. Stage 1 começa.
T+3h:     Stage 1 avalia. Todos os gates devem PASS.
T+3h:     Stage 2 começa. Primeiro slot UTC → snapshot 1.
T+11h:    Segundo slot UTC → snapshot 2.
T+19h:    Terceiro slot UTC → snapshot 3.
           pass_snapshots >= 3 → should_retrain = "first_training"
T+19-20h: Retrain completa (~5-10 min com optimizações, não 30+ min).
           Modelo deployed com v3_exit_aware_40 (40 features).
           signal_gate_anti_spike avalia.
T+24h+:   Window SLA avaliado.
T+27h:    Stage 2 avaliação final.
```

### C4. Monitoramento (3 checkpoints)

```
T+4h:  Stage 1 PASS? RSS estável? Snapshot 1 certificado?
       feature_history_contract PASS com 40 features?
T+12h: RSS flat? Snapshots 1+2 no manifest? Verdict healthy/degraded?
T+20h: Retrain triggered? Modelo v3_exit_aware_40 criado?
       AUC > 0.55? signal_anti_spike PASS?
       Exit policies presentes no output? Signal score > 0?
```

---

## BLOCO D — Validação Final (~5 min)

```bash
cat backend/out/arbml_v2_soak_*/stage2_result.json | python3 -m json.tool
# Se ok=true → SOAK COMPLETO

# Extra v3 validation:
cd backend && python3 -c "
import sys, json; sys.path.insert(0, 'src')
meta = json.load(open('out/config/best_lstm_model.meta.json'))
print(f'Feature contract: {meta.get(\"feature_contract_version\")}')
print(f'Feature count:    {meta.get(\"feature_count\", len(meta.get(\"feature_names\", [])))}')
print(f'AUC:              {meta.get(\"test_metrics\", {}).get(\"roc_auc\")}')
assert meta.get('feature_contract_version') == 'v3_exit_aware_40', 'Wrong contract!'
print('v3 model deployed OK')
"
```

---

## Cenários de FAIL

| Cenário | Impacto | Acção |
|---|---|---|
| feature_count = 25 (not 40) | feature_history_contract FAIL | Fix 0.1 não aplicado |
| contract = v2_multiscale_25 | feature_contract_version FAIL | DEFAULT_FEATURE_CONTRACT not v3 |
| positive_rate < 0.01 | labeling gate FAIL | Threshold muito alto para dados actuais |
| positive_rate > 0.20 | labeling gate FAIL | Threshold muito baixo |
| AUC < 0.55 | trained_model_auc FAIL | Investigar: features NaN? exit features vazias? |
| Exit policies all zero | v3 functionality broken | normalize_episode não extrai fields |
| Signal score = 0 | Ranking inútil | viability ou prob = 0, investigar |
| Retrain > 15 min | Pipeline optimizations not active | stride/scaffold/AMP não carregados |
| context_strength weak → EXECUTE | Gate bug | Fix 0.2 não aplicado |

---

## Dados: Qualidade e Requisitos

### Estado actual dos dados

Verificar antes do soak:

```bash
cd backend && python3 -c "
import sys, sqlite3, time; sys.path.insert(0, 'src')
conn = sqlite3.connect('out/config/tracker_history.sqlite', timeout=30)
now = time.time()

# Volume
total = conn.execute('SELECT COUNT(*) FROM tracker_records').fetchone()[0]
episodes = conn.execute('SELECT COUNT(*) FROM tracker_pair_episodes').fetchone()[0]
closed = conn.execute('SELECT COUNT(*) FROM tracker_pair_episodes WHERE is_closed = 1').fetchone()[0]
pairs = conn.execute('SELECT COUNT(DISTINCT pair_id) FROM tracker_records').fetchone()[0]

# Time span
oldest = conn.execute('SELECT MIN(ts) FROM tracker_records').fetchone()[0]
newest = conn.execute('SELECT MAX(ts) FROM tracker_records').fetchone()[0]
span_h = (newest - oldest) / 3600

# Episode quality for v3 features
ep_with_exit = conn.execute(
    'SELECT COUNT(*) FROM tracker_pair_episodes WHERE exit_spread_at_close IS NOT NULL AND exit_spread_at_close != 0'
).fetchone()[0]
ep_with_peak = conn.execute(
    'SELECT COUNT(*) FROM tracker_pair_episodes WHERE peak_entry_spread IS NOT NULL AND peak_entry_spread > 0'
).fetchone()[0]
ep_with_duration = conn.execute(
    'SELECT COUNT(*) FROM tracker_pair_episodes WHERE duration_sec IS NOT NULL AND duration_sec > 0'
).fetchone()[0]

# Threshold check
above_050 = conn.execute(
    'SELECT COUNT(*) FROM tracker_pair_episodes WHERE is_closed = 1 AND (peak_entry_spread + exit_spread_at_close) >= 0.50'
).fetchone()[0]

print(f'=== Data Volume ===')
print(f'Records:    {total:,}')
print(f'Pairs:      {pairs:,}')
print(f'Episodes:   {episodes:,} (closed: {closed:,})')
print(f'Span:       {span_h:.1f}h ({span_h/24:.1f} days)')
print()
print(f'=== Episode Quality (v3 features need these) ===')
print(f'With exit_spread:       {ep_with_exit:,} / {episodes:,} ({100*ep_with_exit/max(episodes,1):.1f}%)')
print(f'With peak_entry_spread: {ep_with_peak:,} / {episodes:,} ({100*ep_with_peak/max(episodes,1):.1f}%)')
print(f'With duration_sec:      {ep_with_duration:,} / {episodes:,} ({100*ep_with_duration/max(episodes,1):.1f}%)')
print(f'Total >= 0.50%:         {above_050:,} / {closed:,} ({100*above_050/max(closed,1):.1f}%)')
print()

# Minimum requirements for meaningful v3 training
print(f'=== Minimum Requirements ===')
ok_records = total >= 500_000
ok_episodes = closed >= 5_000
ok_above_050 = above_050 >= 100
ok_span = span_h >= 24
ok_ep_exit = ep_with_exit / max(episodes, 1) >= 0.80
print(f'Records >= 500K:         {\"PASS\" if ok_records else \"FAIL\"} ({total:,})')
print(f'Closed episodes >= 5K:   {\"PASS\" if ok_episodes else \"FAIL\"} ({closed:,})')
print(f'Episodes >= 0.50%:       {\"PASS\" if ok_above_050 else \"FAIL\"} ({above_050:,})')
print(f'Data span >= 24h:        {\"PASS\" if ok_span else \"FAIL\"} ({span_h:.1f}h)')
print(f'Episodes with exit >= 80%: {\"PASS\" if ok_ep_exit else \"FAIL\"} ({100*ep_with_exit/max(episodes,1):.1f}%)')

conn.close()
"
```

### Requisitos mínimos para treino v3

| Métrica | Mínimo | Motivo |
|---|---|---|
| Records | 500K+ | Features multi-escala precisam de volume |
| Closed episodes | 5K+ | Rolling episode stats precisam de amostra |
| Episodes >= 0.50% | 100+ | Positivos para treino (~4-5% rate) |
| Data span | 24h+ | Rolling 8h window precisa de contexto |
| Episodes with exit_spread | 80%+ | v3 exit features dependem deste campo |
| Episodes with peak_entry | 80%+ | Cycle features dependem deste campo |
| Episodes with duration | 80%+ | Reversion speed depende deste campo |

### Se episódios não têm exit/peak/duration

Se < 80% dos episódios têm estes campos, significa que os dados foram
colectados ANTES de o TrackerEpisode gravar estes campos. Opções:

1. **Esperar dados novos:** 24-48h de colecta com código v3 gera episódios completos
2. **Fallback:** v3 features retornam 0.0 para episódios incompletos — modelo treina
   mas features de ciclo são menos informativas
3. **Limpar e recomeçar:** Se dados muito antigos, truncar DB e colectar de novo

### Sampling: o que está implementado vs pendente

```
✅ Stride=5:           Reduz overlap 93% → 67%, samples mais independentes
✅ Right-censoring:    Janelas sem horizonte completo descartadas
✅ Embargo no split:   Train/val/test separados cronologicamente
✅ FocalLoss:          Equilibra classes (y=0 vs y=1)

❌ max_samples_per_pair (S2): NÃO implementado
   → Pares dominantes podem contribuir 80%+ dos samples
   → MITIGAÇÃO: stride=5 reduz o efeito (menos samples por par)
   → PRIORIDADE: Baixa para soak, implementar antes do 2º retrain

❌ Sampling diagnostics (S5): NÃO implementado
   → Sem métricas de concentração de pares no summary
   → PRIORIDADE: Baixa, monitorar manualmente após treino
```

---

## Checklist Completa

```
□ 0.1: soak_runbook.py V2→V3 (6 linhas)
□ 0.2: ml_analyzer.py context_strength gate
□ 0.3: spread_tracker.py cost_estimate_pct threading
□ 0.4: Import smoke test
□ A1: Certificação offline (3 snapshots CERTIFIED)
□ A2: Labeling audit (mode, rate ∈ [0.01,0.20], leakage, contract=v3_exit_aware_40)
□ A3: Treino offline (AUC > 0.55, Platt finite, contract=v3_exit_aware_40)
      → GUARDAR baseline v3: AUC=___, ETA MAE=___
□ A4: DB integrity (8h+ retention, config invariant)
□ A5: Signal Ledger + Exit Policies smoke test
□ B:  Correcções se necessário
□ C1: Estado limpo
□ C2: Código com todos os fixes + testes passam
□ C3: Runner lançado
□ C4-T+4h:  Stage 1 PASS, 40 features, snapshot 1 certificado
□ C4-T+12h: RSS flat, snapshots 1+2 no manifest
□ C4-T+20h: Retrain v3 ok (~5-10 min), signal_anti_spike ok
□ D:  Stage 2 ok=true → SOAK v3 COMPLETO
```

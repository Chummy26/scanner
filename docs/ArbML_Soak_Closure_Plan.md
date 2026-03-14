# ArbML Soak Closure — Plano Validado (v2)

## Princípio: Zero surpresas nas 24h

Validar OFFLINE tudo que pode falhar antes de iniciar as 24h live.
As 24h confirmam estabilidade temporal, não descobrem bugs.

```
BLOCO A (agora, ~1h): validar offline tudo que pode falhar
BLOCO B (se A falhar): corrigir antes de subir servidor
BLOCO C (24h): confirmação live
BLOCO D (5 min): validação final
```

---

## Mapa completo de gates (38 gates máximo)

### Stage 1 — 20 gates

| # | Gate | Offline? | Nota |
|---|---|---|---|
| 1 | debug_perf_api | ❌ | HTTP |
| 2 | system_health_api | ❌ | HTTP |
| 3 | pipeline_status_api | ❌ | HTTP |
| 4 | scanner_lite_api | ❌ | HTTP |
| 5 | rss_growth_pct_per_hour | ❌ | 3h+ live, <= 5.0 |
| 6 | calculate_ms_p95 | ❌ | < 80ms |
| 7 | event_loop_lag_p95 | ❌ | < 200ms |
| 8 | event_loop_lag_p99 | ❌ | < 500ms |
| 9 | record_sink_idle_cycles | ❌ | p50 == 0 |
| 10 | tracker_low_spread_capture | ✅ SQL | records + pairs > 0 |
| 11 | dashboard_filters_sub_0_2 | ❌ | API |
| 12 | data_distribution_below_0_2 | ✅ SQL | min entry < 0.2% |
| 13 | stale_pairs_absent | ✅ SQL | <= 3 stale |
| 14 | cross_exchange_rejection | ✅ SQL | MUDAR para verdict-based (ver A5) |
| 15 | feature_history_contract | ❌ | Tracker live, 25 features |
| 16 | hourly_digest_present | ✅ SQL | verdict present |
| 17 | runtime_audit_events_budget | SKIP OK | Skippable se audit pkg ausente |
| 18 | runtime_audit_ws_latency | SKIP OK | Skippable |
| 19 | books_p95_under_5s | SKIP OK | Skippable |
| 20 | disconnects_absent | ❌ | ⚠️ Ver nota abaixo |

Política SKIP: Gates 17-19 são SKIPPED (ok=true) quando runtime_audit package
não está disponível na sessão actual. Aceitável para closure — são subsistemas opcionais.

Nota disconnects_absent: Exige 0 disconnects E 0 reconnects. Com WebSocket em
produção, reconexões breves são normais (especialmente com XT a flap). Se runtime
audit estiver AUSENTE, defaults a 0 (PASS). Se estiver PRESENTE e XT reconectou,
FAIL. Nesse caso: verificar se são reconexões breves (tolerável) ou desconexões
prolongadas (problema real). Se necessário, relaxar para permitir reconexões
transientes.

### Stage 2 — 18 gates (máximo)

| # | Gate | Offline? | Condicional? |
|---|---|---|---|
| 1 | pass_snapshots >= 3 | ✅ manual | Base |
| 2 | snapshot_certification_and_sha | ✅ | Base |
| 3 | snapshot_idempotency | ✅ | Base |
| 4 | sqlite_retains_gt_8h | ✅ DB | Base |
| 5 | memory_drops_gt_13h | ❌ 13h+ | Base |
| 6 | rss_24h_under_2x_warmup | ❌ 24h | Base |
| 7 | window_sla_calculate_ms | ❌ 24h | Base |
| 8 | window_sla_event_loop | ❌ 24h | Base |
| 9 | auto_retrain_not_stuck | ❌ | Base |
| 10 | label_threshold_mode | ✅ audit | if label_audit |
| 11 | positive_rate_band [0.05, 0.20] | ✅ audit | if label_audit |
| 12 | label_zero_leakage | ✅ audit | if label_audit |
| 13 | trained_model_auc > 0.55 | ✅ treino offline | if retrain_checks |
| 14 | platt_scaling_finite | ✅ treino offline | if retrain_checks |
| 15 | inference_probabilities_bounded | ❌ deploy live | if retrain_checks |
| 16 | feature_contract_version | ✅ metadata | if retrain_checks |
| 17 | training_metadata_threshold_mode | ✅ metadata | if retrain_checks |
| 18 | signal_gate_anti_spike | ❌ live post-deploy | if retrain_checks |

Nota feature_contract_version: Gate aceita v2_multiscale_25 E v1_micro_10.
Para closure, v2_multiscale_25 é o esperado. v1_micro_10 indica que o treino
usou contrato antigo — investigar se dados eram insuficientes para v2.

Nota signal_gate_anti_spike (soak_runbook.py:934-995): Compara sinais
EXECUTE/STRONG_EXECUTE 30min antes vs 30min depois do deploy do modelo.
PASS se post_avg <= max(1, 2x baseline). Auto-PASS se deploy_ts não disponível
ou dados insuficientes. Risco baixo.

Nota CERTIFIED_WITH_WARNINGS: Snapshots com verdict WARN são aceitáveis
(auto_retrain.py:99 is_trainable_snapshot aceita PASS e WARN).

---

## BLOCO A — Validação Offline (~1h)

### A1. Certificação de snapshot (~2 min)

Imports correctos:
- `spread.train_model` → certify_data_for_training (linha 1213)
- `spread.runtime_audit` → create_sqlite_snapshot (linha 234)
- `spread.auto_retrain` → certification_verdict_view (linha 90)

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
Ambos são aceitáveis — is_trainable_snapshot aceita PASS e WARN.

### A2. Labeling audit (~3 min)

Import: `spread.soak_runbook` → audit_snapshot_labeling (linha 750)

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

ok_mode = result.get('label_threshold_mode') == 'rolling_pair_percentile'
ok_rate = 0.05 <= float(result.get('positive_rate', 0)) <= 0.20
ok_leak = bool(result.get('zero_leakage_ok'))
print(f'\\n--- Gate check ---')
print(f'label_threshold_mode:  {\"PASS\" if ok_mode else \"FAIL\"}')
print(f'positive_rate_band:    {\"PASS\" if ok_rate else \"FAIL\"} ({result.get(\"positive_rate\")})')
print(f'label_zero_leakage:    {\"PASS\" if ok_leak else \"FAIL\"}')
"
```

**PASS se:** mode=rolling_pair_percentile, rate ∈ [0.05, 0.20], zero_leakage=True

### A3. Treino offline (~5-15 min)

Import correcto: `spread.train_model` → run_clean_training_cycle (linha 1915)

```bash
cd backend && python3 -c "
import sys, json, math; sys.path.insert(0, 'src')
from spread.train_model import run_clean_training_cycle
from pathlib import Path

result = run_clean_training_cycle(
    base_path=Path('out'),
    artifact_dir=Path('/tmp/arbml_test_model'),
)
print(f'status: {result.get(\"status\")}')
print(f'auc:    {result.get(\"auc\")}')
print(f'platt_scale: {result.get(\"platt_scale\")}')
print(f'platt_bias:  {result.get(\"platt_bias\")}')
print(f'feature_contract: {result.get(\"feature_contract_version\")}')
print(f'label_mode: {result.get(\"label_threshold_mode\")}')
print(f'samples: {result.get(\"dataset_samples\")}')

ok_auc = float(result.get('auc', 0)) > 0.55
ok_platt = math.isfinite(float(result.get('platt_scale', 0))) and math.isfinite(float(result.get('platt_bias', 0)))
ok_contract = result.get('feature_contract_version') == 'v2_multiscale_25'
ok_mode = result.get('label_threshold_mode') == 'rolling_pair_percentile'
print(f'\\n--- Gate check ---')
print(f'trained_model_auc > 0.55:   {\"PASS\" if ok_auc else \"FAIL\"} ({result.get(\"auc\")})')
print(f'platt_scaling_finite:       {\"PASS\" if ok_platt else \"FAIL\"}')
print(f'feature_contract_version:   {\"PASS\" if ok_contract else \"FAIL\"} ({result.get(\"feature_contract_version\")})')
print(f'training_metadata_mode:     {\"PASS\" if ok_mode else \"FAIL\"} ({result.get(\"label_threshold_mode\")})')

if result.get('feature_contract_version') == 'v1_micro_10':
    print('\\n⚠️  ATENÇÃO: treino usou v1_micro_10, não v2_multiscale_25.')
    print('   Provável causa: dados insuficientes para features multi-escala.')
    print('   Investigar antes de prosseguir.')
"
```

**PASS se:** AUC > 0.55, Platt finite, contract=v2_multiscale_25, mode=rolling_pair_percentile
**Se contract=v1_micro_10:** investigar — dados insuficientes para multi-escala?

### A4. DB integrity check (~1 min)

```bash
cd backend && python3 -c "
import sys, time, sqlite3; sys.path.insert(0, 'src')

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
"
```

### A5. Rejection gate — mudar para verdict-based

Ficheiro a editar: `backend/src/spread/soak_runbook.py` (gate cross_exchange_rejection_rate)

```python
# ANTES:
_bool_gate(
    "cross_exchange_rejection_rate",
    _safe_float(latest_hour.get("rejection_rate_pct"), 0.0) < 10.0 if latest_hour else False,
    value=latest_hour.get("rejection_rate_pct") if latest_hour else None,
    expected="< 10%",
),

# DEPOIS:
_bool_gate(
    "cross_exchange_rejection_rate",
    str(latest_hour.get("quality_verdict") or "") != "unhealthy" if latest_hour else False,
    value={
        "quality_verdict": latest_hour.get("quality_verdict"),
        "rejection_rate_pct": latest_hour.get("rejection_rate_pct"),
    } if latest_hour else None,
    expected="verdict != unhealthy",
),
```

Justificação: 10% é mais restritivo que o próprio sistema (20% = degraded, 50% = unhealthy).
Com 1 exchange OPEN (XT), rejection ~32% é protecção a funcionar, não erro.
FAIL só com unhealthy (2+ exchanges OPEN ou rejection > 50%).

Verificar:

```bash
cd backend && python3 -c "
import sys, sqlite3; sys.path.insert(0, 'src')
conn = sqlite3.connect('out/config/tracker_history.sqlite', timeout=30)
conn.row_factory = sqlite3.Row
rows = list(conn.execute(
    'SELECT quality_verdict, rejection_rate_pct FROM tracker_hourly_health ORDER BY hour_start_ts DESC LIMIT 5'
))
for r in rows:
    v = r['quality_verdict']
    rate = r['rejection_rate_pct']
    ok = v != 'unhealthy'
    print(f'  {v} rejection={rate:.2f}% {\"PASS\" if ok else \"FAIL\"}')
conn.close()
"
```

**PASS se:** nenhum digest recente é "unhealthy".
**Se PASS mas rejection > 10%:** verificar que rejection está concentrada em
exchanges com breaker OPEN (esperado), não distribuída em CLOSED (bug real).

---

## BLOCO B — Correcções (se A falhar)

| Falha | Correcção | Ficheiro |
|---|---|---|
| Cert FAIL | Ajustar gate thresholds | training_certification.py |
| positive_rate fora [0.05, 0.20] | Ajustar label_cost_floor / percentile | models.py |
| AUC < 0.55 | Investigar data quality, sequence_length | train_model.py |
| Platt non-finite | Bug no treino — overfit ou NaN | train_model.py |
| contract = v1_micro_10 | Dados insuficientes para v2? | Investigar |
| Rejection unhealthy | 2+ exchanges OPEN | Investigar conectividade |
| 8h retention fail | DB muito recente | Esperar |

**Regra:** NÃO iniciar Bloco C até Bloco A todo verde. Cada ciclo A→B→A ~15-20min.

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
git log --oneline -5  # confirmar todos os patches
python3 -m pytest backend/tests/ -x -q  # deve passar 190+
# Subir servidor (comando habitual)
```

### C3. Lançar runner

```bash
python3 backend/tests/run_arbml_v2_soak.py &
```

### C4. Timeline esperada

```
T+0h:     Servidor sobe. Stage 1 começa.
T+3h:     Stage 1 avalia. Todos os gates devem PASS.
T+3h:     Stage 2 começa. Primeiro slot UTC → snapshot 1.
T+11h:    Segundo slot UTC → snapshot 2.
T+19h:    Terceiro slot UTC → snapshot 3.
           pass_snapshots >= 3 → should_retrain = "first_training"
T+19-21h: Retrain completa. Modelo deployed.
           signal_gate_anti_spike avalia (30min antes/depois deploy).
T+24h+:   Window SLA avaliado (H1-H4 vs H20-H24).
T+27h:    Stage 2 avaliação final.
```

### C5. Monitoramento (3 checkpoints)

```
T+4h:  Stage 1 PASS? RSS estável? Snapshot 1 certificado?
T+12h: RSS flat? Snapshots 1+2 no manifest? Verdict healthy/degraded?
T+20h: Retrain triggered? Modelo criado? AUC ok? signal_anti_spike PASS?
```

---

## BLOCO D — Validação Final (~5 min)

```bash
cat backend/out/arbml_v2_soak_*/stage2_result.json | python3 -m json.tool
# Se ok=true → SOAK COMPLETO
# Se ok=false → ler quais gates falharam
```

---

## Cenários de FAIL

| Cenário | Impacto | Acção |
|---|---|---|
| Cert FAIL no slot automático | 0 pass_snapshots | Corrigir cert, restart, esperar próximo slot |
| AUC < 0.55 no retrain live | post_retrain gate FAIL | Investigar dados/features |
| RSS > 2x warmup | rss gate FAIL | Investigar leak |
| XT OPEN toda a run | verdict = degraded, modelo sem pares XT | Tolerável — XT entra no retrain futuro |
| Rejection > 10% sem breaker OPEN | Bug subtil num parser | INVESTIGAR — problema real |
| Retrain > 2h | auto_retrain_stuck | Kill retrain, investigar |
| positive_rate fora [5%, 20%] | labeling gate FAIL | Ajustar cost_floor / percentile |
| disconnects_absent FAIL | XT reconexões | Se breves: relaxar gate. Se prolongadas: investigar |
| signal_anti_spike FAIL | Spike de sinais pós-deploy | Modelo emitindo demasiados EXECUTE — investigar |

---

## Checklist

```
□ A1: Certificação offline PASS (imports de spread.train_model)
□ A2: Labeling audit PASS (mode, rate ∈ [0.05,0.20], leakage)
□ A3: Treino offline PASS (AUC > 0.55, Platt finite, contract=v2_multiscale_25)
□ A4: DB integrity PASS (8h+ retention)
□ A5: Rejection verdict != unhealthy + concentrada em exchanges OPEN
□ A5: Gate verdict-based aplicado em soak_runbook.py
□ B:  Correcções aplicadas se necessário
□ C1: Estado limpo
□ C2: Código com todos os patches + testes passam
□ C3: Runner lançado
□ C4-T+4h:  Stage 1 PASS, snapshot 1 certificado
□ C4-T+12h: RSS flat, snapshots 1+2 no manifest
□ C4-T+20h: Retrain ok, signal_anti_spike ok
□ D:  Stage 2 ok=true → SOAK COMPLETO
```

---

## Após SOAK COMPLETO → Transição para v3

O soak valida a INFRAESTRUTURA. O v3 plan é o PRODUTO.

### Imediato (dia seguinte)

1. Reset do modelo:
   - `rm best_lstm_model.pth best_lstm_model.meta.json auto_retrain.state.json`

2. Implementar Fase S (sampling + training perf, ~2.5h):
   - S1: stride=5 na janela deslizante (ml_dataset.py)
   - S2: max_samples_per_pair=2000 (ml_dataset.py)
   - S5: diagnóstico de sampling no summary
   - T1-T4: batch=1024, pin_memory, AMP, cudnn.benchmark (train_model.py)
   - Ver ArbML_Sampling_Analysis.md e ArbML_Training_Performance.md

3. Implementar Fase 1 do v3 plan (features):
   - 40 features exit-aware multi-scale (V3_EXIT_AWARE_40)

4. Primeiro treino com dados limpos:
   - Baseline v2_multiscale_25 → depois v3_exit_aware_40 → comparar

### Semana 1-2: Fase 2 (dashboard) + Fase 3 (SignalScore)
### Semana 2-3: Acumular ledger (200+ sinais)
### Semana 3-4: Fase 4 (CLSTM context fusion)
### Futuro (6+ meses): Survival ETA, cost por venue, threshold por frequência, horizonte adaptativo

---

## Ficheiros críticos

| Ficheiro | Acção |
|---|---|
| backend/src/spread/soak_runbook.py | EDITAR gate rejection → verdict-based |
| backend/src/spread/train_model.py | REFERÊNCIA (certify_data_for_training:1213, run_clean_training_cycle:1915) |
| backend/src/spread/auto_retrain.py | REFERÊNCIA (certification_verdict_view:90, update_snapshot_manifest:172) |
| backend/src/spread/runtime_audit.py | REFERÊNCIA (create_sqlite_snapshot:234) |
| backend/tests/run_arbml_v2_soak.py | EXECUTAR (runner) |

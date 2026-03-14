# ArbML v3 — Plano Final Consolidado (v1.2)

## Contexto

Este documento consolida todas as iterações de análise num plano único e implementável:

1. Feature set v3 exit-aware (40 features)
2. Dashboard com sinais operacionais (3 políticas de saída)
3. Signal Ledger (feedback loop)
4. SignalScore composto para ranking
5. CLSTM — Context Vector Fusion no modelo

Cada item tem dependências explícitas, estimativa de esforço e critérios de aceitação.

---

## Convenção de sinal (CRÍTICO — aplica-se a todo o documento)

O exit_spread é NEGATIVO quando o spread inverteu. O codebase usa adição direta:

```python
# spread_tracker.py:93
def total_spread(self) -> float:
    return float(self.peak_entry_spread + self.exit_spread_at_close)
# Exemplo: 4.2 + (-1.8) = 2.4%
```

Regras:

- **capture = entry + exit** (SEM módulo — exit já carrega sinal correto)
- **min_entry = max(0, cost - exit)** (breakeven: quando exit é negativo, entry compensa; quando positivo, exit ajuda)
- **Denominadores de ratios** usam piso eps=0.01 para evitar divisão por zero

---

## Diagnóstico: o que existe e o que falta

### O que já funciona

| Componente | Estado |
|---|---|
| Captura persistente 15s, SQLite WAL, 12h RAM / 8d disco | ✅ Produção |
| Episódios econômicos (peak_entry + exit_at_close) | ✅ Produção |
| Recurring context: bandas P10-P90 entry/exit, coerência short/long | ✅ Produção |
| LSTM + TemporalAttention + dual head (prob + multi-quantile ETA) | ✅ Produção |
| Platt scaling, FocalLoss, threshold selection cronológica | ✅ Produção |
| Adaptive labeling (rolling pair percentile, 3×3 sweep) | ✅ Produção |
| Drift detection (features > 3σ do treino) | ✅ Produção |
| Snapshots imutáveis com SHA-256 | ✅ Produção |
| Soak v2 (Stage 1 PASS, Stage 2 em curso) | ✅ Em execução |

### O que está errado ou ausente

| Problema | Impacto |
|---|---|
| 15 features multi-escala são TODAS sobre entry_spread | Modelo cego para exit behavior em 30m/2h/8h |
| Zero features de profitabilidade ou timing de episódios | Modelo não sabe se operação é lucrativa |
| Zero features cross entry↔exit | Modelo não calcula viabilidade |
| Contexto episódico (support, coerência, ranges) NÃO entra no modelo | Usado apenas como gate pós-modelo |
| Dashboard mostra ranges independentes sem spread total | Operador precisa fazer contas de cabeça |
| Sinal é modelo-cêntrico (WAIT/EXECUTE) ao invés de operacional | Não responde "quanto vou lucrar?" |
| Sem ranking composto dos sinais | Ordenação por probabilidade pura é insuficiente |
| Sem cost_estimate no config | Lucro líquido não é calculável |
| Sem feedback loop (modelo não sabe se sinais acertaram) | Sem calibração online, sem aprendizado de erros |

---

## Fase 1 — Feature Set v3: Exit-Aware Multi-Scale

**Pré-requisito:** Nenhum (dados já existem nos records e episódios)
**Timing:** ANTES do primeiro treino v3
**Estimativa:** ~150 linhas em feature_contracts.py + ~30 linhas de teste

### 1.1 Features mantidas (25 existentes)

**Micro — 10 features (~75s window):**

- entry_spread, exit_spread
- delta_entry, delta_exit
- delta2_entry, delta2_exit
- rolling_std_entry, rolling_std_exit
- zscore_entry, zscore_exit

**Entry multi-escala — 15 features:**

- 30m: mean_entry_30m, std_entry_30m, max_entry_30m, episode_count_30m, position_in_range_30m
- 2h: mean_entry_2h, std_entry_2h, max_entry_2h, episode_count_2h, trend_slope_2h
- 8h: mean_entry_8h, std_entry_8h, max_entry_8h, episode_count_8h, zscore_vs_8h

### 1.2 Features novas (15 adicionais)

**Exit multi-escala — 6 features:**

- mean_exit_2h: média do exit_spread nas últimas 2h
- exit_p10_2h: percentil 10 do exit nas últimas 2h (robusto a outliers, consistente com tracker)
- std_exit_2h: volatilidade do exit nas últimas 2h
- mean_exit_8h: média do exit_spread nas últimas 8h
- exit_p10_8h: percentil 10 do exit nas últimas 8h
- zscore_exit_vs_8h: (exit_atual − mean_exit_8h) / max(std_exit_8h, 0.01)

**Ciclo completo / profitabilidade — 6 features:**

- mean_total_spread_2h: média do spread total (entry + exit, sem módulo) dos episódios fechados nas últimas 2h
- median_episode_duration_2h: mediana do tempo de fechamento dos episódios em 2h (segundos)
- episode_close_rate_2h: taxa de fechamento em coorte matura (ver nota)
- mean_total_spread_8h: idem para 8h
- median_episode_duration_8h: idem para 8h
- mean_reversion_speed_8h: slope médio de convergência (Δspread / Δt) por hora em 8h

Nota sobre episode_close_rate: usa coorte matura para evitar viés de censoring. Apenas episódios iniciados em [t - W - H, t - H] são contados, onde W é a janela e H é o horizonte de maturação (default: 2h). Todos os episódios da coorte tiveram tempo suficiente para fechar.

Nota sobre filtro de episódios: as features de ciclo completo usam TODOS os episódios fechados (sem filtro de min_total_spread_pct). O modelo decide o que importa. O recurring context (que gera os targets do card na Fase 2) continua filtrado por min_total_spread_pct. Ver seção 2.3.1.

**Relação entry↔exit — 3 features:**

- current_spread_total: entry_atual + exit_atual (SEM módulo)
- entry_exit_ratio_2h: mean_entry_2h / max(abs(mean_exit_2h), 0.01)
- viability_score: (entry_atual + mean_exit_2h - cost) / max(std_entry_2h, 0.01) (net capture normalizada)

Nota sobre viability_score: usa entry + exit - cost (net capture real), não entry - |exit|. Quando exit é positivo (spread inverteu), exit AJUDA. Quando negativo, o efeito é o mesmo (entry + (-1.5) - cost = entry - 1.5 - cost).

### 1.3 Total: 40 features — Contrato V3_EXIT_AWARE_40

### 1.4 Implementação

1. `feature_contracts.py`: V3_EXIT_AWARE_40, `_RollingExitWindow`, `_RollingEpisodeStats` (coorte matura), expandir `_build_multiscale_feature_dicts`
2. `ml_model.py`: Sem alteração — input_size parametrizado
3. `train_model.py`: Sem alteração — feature_names do contract
4. Testes: feature names, viability negativo, eps floor, coorte matura, episódios fechados only

### 1.5 Critérios de aceitação

- 40 features por record com > 2h de histórico; 0.0 para warm-up
- Denominadores nunca produzem inf/nan (eps floor)
- Cycle features (mean_total_spread, close_rate, etc.) use ALL episodes, not filtered
- Snapshots existentes reprocessáveis com v3
- Invariante: `min_total_spread_pct >= min_net_capture_pct + default_cost_estimate_pct` validado no startup

---

## Fase 2 — Dashboard: Sinais Operacionais com 3 Políticas

**Pré-requisito:** Nenhum (usa dados empíricos do tracker)
**Timing:** PARALELO ao soak
**Estimativa:** ~60 linhas render_prediction + ~20 linhas to_scanner_lite_dict

### 2.1 Três políticas de saída

Nomenclatura descritiva neutra (sem implicação de frequência condicional):

```
shallow  = exit_core_range_max    (P75 — saída rasa)
median   = mediana dos exits      (P50 — saída mediana)
deep     = exit_outer_range_min   (P10 — saída profunda)
```

A frequência condicional de atingir cada nível depende do estado atual, não apenas do quantil. A coluna Prob no card (e, futuramente, probabilidades por política) informa frequência.

### 2.2 Campos calculados

```
capture_gross(k) = entry_atual + exit_target(k)                         ← SEM módulo
net_capture(k)   = capture_gross(k) - default_cost_estimate_pct
min_entry(k)     = max(0, default_cost_estimate_pct - exit_target(k))   ← breakeven correto
```

Nota sobre min_entry: quando exit é NEGATIVO (spread não inverteu), entry precisa compensar: `cost - (-1.50) = 1.80%`. Quando exit é POSITIVO (spread já inverteu a favor), exit ajuda — min_entry cai ou chega a zero.

```
exit = -1.50%: max(0, 0.30 + 1.50) = 1.80%  (precisa entry alto)
exit = +0.50%: max(0, 0.30 - 0.50) = 0.00%  (exit já cobre custos)
exit =  0.00%: max(0, 0.30 - 0.00) = 0.30%  (só custos)
```

Exemplo com exit negativo (entry=4.20%, exit_median=-1.80%, cost=0.30%):

```
capture_gross = 4.20 + (-1.80) = 2.40pp
net_capture   = 2.40 - 0.30 = 2.10pp
min_entry     = max(0, 0.30 - (-1.80)) = 2.10%
```

Exemplo com exit positivo (entry=1.50%, exit_median=+0.90%, cost=0.30%):

```
capture_gross = 1.50 + 0.90 = 2.40pp
net_capture   = 2.40 - 0.30 = 2.10pp
min_entry     = max(0, 0.30 - 0.90) = 0.00%  (qualquer entry positivo já é lucrativo)
```

### 2.3 Config

```python
default_cost_estimate_pct: float = 0.30
min_net_capture_pct: float = 0.20
```

### 2.3.1 Relação com min_total_spread_pct (CRÍTICO)

`min_total_spread_pct` é o ÚNICO threshold de qualidade económica. Cobre fees, slippage e risco num só número. `label_cost_floor_pct` segue automaticamente (usa min_total como fallback).

```
min_total_spread_pct = 0.50%  (default, SpreadConfig)
  = fees (~0.10-0.15% por lado)
  + slippage (~0.05%)
  + margem de segurança (~0.10-0.15%)
```

Afeta TODO o sistema em cascata:

```
1. Coleta:    tracker_min_spread_pct = 0.05% (records com entry < 0.05% descartados)
2. Episódios: detector baseline + MAD (spreads ~0.01-0.10% raramente ativam)
3. Labels:    cost_floor = max(0.50%, P70 do par) → y=1 SÓ com total >= threshold
4. Adaptive:  P70 sobe automaticamente para pares com histórico forte
5. Contexto:  episódios < 0.50% excluídos de ranges/support/strength
6. Prob:      execute_threshold calibrado bloqueia sinais fracos
```

Dados reais do DB live (73K episódios fechados):

```
>= 0.20%: 1,944 episódios (2.66%)
>= 0.30%: 1,098 episódios (1.50%)
>= 0.50%:   454 episódios (0.62%)  ← threshold actual
>= 1.00%:    96 episódios (0.13%)  ← threshold anterior (insuficiente para treino)
```

Com 0.50%: 454 episódios qualificados → centenas de positivos no dataset → modelo treina. Com 1.00%: 96 episódios → ~0 positivos no dataset → modelo não treina.

Hierarquia:

```
PRIMÁRIO:    min_total_spread_pct = 0.50% bruto
             → Define o que o modelo aprende (labels)
             → Define o que o contexto mostra (ranges, support)
             → Define o que o operador vê (exit targets)

SECUNDÁRIO:  execute_threshold / strong_threshold
             → Probabilidade calibrada mínima para publicar

SAFETY NET:  min_net_capture_pct = 0.20% líquido
             → Protege contra edge case: capture marginal
             → Equivalente bruto: 0.20% + 0.30% cost = 0.50%
```

**Invariante de consistência:**

`min_total_spread_pct` DEVE ser >= `min_net_capture_pct + default_cost_estimate_pct`.

```
Valores actuais: 0.50% >= 0.20% + 0.30% = 0.50%  ✅ (exacto)
```

**Limitação conhecida: spreads moderados de alta frequência**

Um par com total_spread = 0.40% que cicla 15 vezes em 2h produz ~6% de edge acumulado bruto — potencialmente bom. Mas com `min_total_spread_pct = 0.50%`, esse par é invisível. A solução futura é threshold ajustado por frequência.

**Features v3 vs filtro:**

Features do modelo (Fase 1) veem TODOS os episódios — o LSTM decide o que importa. O recurring context (targets do card) continua filtrado por min_total_spread_pct. Isso dá ao modelo mais informação sem contaminar os targets do operador.

### 2.4 Payload expandido

```python
"exit_shallow", "exit_median", "exit_deep",
"capture_gross_shallow", "capture_gross_median", "capture_gross_deep",
"net_capture_shallow", "net_capture_median", "net_capture_deep",
"min_entry_shallow", "min_entry_median", "min_entry_deep",
"expected_net_median": prob_median * net_capture_median,  # ranking, NÃO gate

# PROB e ETA são explicitamente da política MEDIAN
"prob_median": prob_calibrada,                # probabilidade de atingir exit_median em horizonte
"eta_q10_min", "eta_q25_min", "eta_q50_min", "eta_q75_min", "eta_q90_min",  # quantis condicionais do modelo
"eta_uncertainty_min": (eta_q75 - eta_q25),   # dispersão do intervalo central
```

Nota: `prob_median` refere-se à política median (P50). Os quantis de ETA (`eta_q10..q90`) são condicionais às features actuais — NÃO são médias históricas. São produzidos pelo multi-quantile head do modelo. Futuramente, probabilidades por política (prob_shallow, prob_deep) poderão ser adicionadas quando o modelo suportar multi-target.

### 2.5 Card do dashboard

```
┌─────────────────────────────────────────────────────────┐
│  DOGE  │  MEXC spot → GATE futures  │  EXECUTE (MEDIAN) │
├─────────────────────────────────────────────────────────┤
│  ENTRADA ATUAL:      4.20%                               │
│  ┌──────────────┬──────────┬──────────┬──────────┐      │
│  │  Política    │  Saída   │ Cap.Líq  │ Mín.Entr │      │
│  ├──────────────┼──────────┼──────────┼──────────┤      │
│  │ Shallow P75  │  -1.20%  │  +2.70pp │  1.50%   │      │
│  │ Median  P50  │  -1.80%  │  +2.10pp │  2.10%   │      │
│  │ Deep    P10  │  -2.40%  │  +1.50pp │  2.70%   │      │
│  └──────────────┴──────────┴──────────┴──────────┘      │
│  PROB (median): 74%    ETA (median): 18 / 32 / 61 min   │
│  ├──10m────|═══════32m═══════|────75m──┤  confiança: ±22m│
│  SUPORTE: 5 ep (2h) · 28 ep (24h) · CONTEXTO: strong    │
└─────────────────────────────────────────────────────────┘
```

PROB e ETA referem-se explicitamente à política median. O operador sabe que 74% é a probabilidade de atingir a saída mediana, não a shallow ou deep.

ETA mostra distribuição condicional do modelo (Q10/Q25/Q50/Q75/Q90), NÃO médias históricas. Os quantis consideram o estado actual das features (spread, velocidade, hora, par). Confiança = Q75-Q25: se < 10min → alta, se > 45min → baixa.

### 2.5.1 ETA Redesign — Multi-Quantile Condicional

**Problema do ETA actual:** O modelo produz 1 ponto único (SmoothL1 em log-space).
Os Q25/Q50/Q75 no dashboard vêm de estatísticas empíricas históricas, desconectadas
do modelo. O operador não sabe a incerteza. 95% dos dados (y=0) são ignorados para
treino de ETA. A loss é simétrica (sub e sobre-estimação pesam igual).

**Redesign (4 mudanças):**

**1) Multi-quantile head:** ETA head passa de 1 neurónio para 5 (Q10/Q25/Q50/Q75/Q90).
Monotonicity por construção: modelo prediz Q10 + 4 deltas positivos via softplus.

```python
# Inferência: 5 quantis condicionais às features actuais
eta_quantiles = model.eta_out(pooled)  # shape: (batch, 5)
# Q10=eta[0], Q25=Q10+softplus(eta[1]), Q50=Q25+softplus(eta[2]), ...
```

**2) Pinball loss com censoring:** Substitui SmoothL1. Usa TODOS os samples:
- Positivos (y=1): pinball loss standard para cada quantil
- Censurados (y=0): penaliza sub-estimação quando T > horizonte (sabemos que o evento NÃO aconteceu em 4h, logo quantis abaixo de 4h são penalizados)

**3) Label peak_ts:** O target muda de "tempo até episódio FECHAR" para "tempo até spread ATINGIR pico". Mais útil para o operador (quando entrar/sair) e mais previsível (pico acontece antes do close).

**4) Métricas de calibração:** Coverage (% de valores reais abaixo do quantil predito), Pinball Score, Winkler Score (penaliza intervalos largos E violações de cobertura).

Ver `ArbML_ETA_Redesign.md` para detalhes completos, código, e referências.

### 2.6 Regras de publicação

**EXECUTE:**

```
p_calibrada >= execute_threshold                         # SECUNDÁRIO: prob mínima
AND net_capture_median >= min_net_capture_pct             # SAFETY NET: edge mínimo (0.20%)
AND context_strength in ("normal", "strong")              # PRIMÁRIO: requer episódios qualificados
AND range_status != "insufficient_empirical_context"      # PRIMÁRIO: requer ranges empíricos
AND drift_status != "drifted"
```

Nota: na prática, `net_capture_median >= 0.20%` é quase sempre satisfeito quando o contexto está "normal"/"strong", porque os episódios que constroem os ranges já passaram pelo filtro de `min_total_spread_pct = 0.50%`. O gate existe como proteção defensiva para edge cases (entry muito baixo + exit raso).

**STRONG_EXECUTE** (todas de EXECUTE mais):

```
p_calibrada >= strong_threshold
AND net_capture_median >= 2 * min_net_capture_pct
AND net_capture_shallow >= min_net_capture_pct
AND context_strength == "strong"
AND support_2h >= 3 AND support_24h >= 5
AND eta_alignment_status != "divergent"
```

expected_net = prob × net_capture é campo de ranking/diagnóstico, NÃO gate.

---

## Fase 2.5 — Signal Ledger

**Pré-requisito:** Fase 2
**Timing:** Junto com Fase 2
**Estimativa:** ~80 linhas

### Princípios

- Tabela única, separação lógica rígida: snapshot (imutável) vs outcome (pending → resolvido)
- Cooldown: máximo 1 sinal ativo por `pair_key × direction × horizon × policy_name`
- Multi-policy labels desde dia 1
- Schema pronto para scoring rules

### Schema

```sql
CREATE TABLE signal_ledger (
    signal_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at             REAL DEFAULT (strftime('%s','now')),

    -- SNAPSHOT (imutável)
    ts_emitted             REAL NOT NULL,
    pair_key               TEXT NOT NULL,
    signal_action          TEXT NOT NULL,
    policy_name            TEXT NOT NULL,       -- shallow / median / deep
    horizon_sec            INTEGER NOT NULL,
    entry_at_signal        REAL NOT NULL,
    exit_at_signal         REAL NOT NULL,
    target_exit_shallow    REAL,
    target_exit_median     REAL,
    target_exit_deep       REAL,
    target_net_shallow     REAL,
    target_net_median      REAL,
    target_net_deep        REAL,
    target_gross_shallow   REAL,             -- bruto para auditoria/reconciliação
    target_gross_median    REAL,
    target_gross_deep      REAL,
    prob_at_signal         REAL,
    eta_q10_at_signal      REAL,             -- quantis condicionais do modelo (multi-quantile head)
    eta_q25_at_signal      REAL,
    eta_q50_at_signal      REAL,
    eta_q75_at_signal      REAL,
    eta_q90_at_signal      REAL,
    eta_uncertainty_sec    REAL,             -- Q75-Q25 (dispersão do intervalo central)
    support_2h             INTEGER,
    support_24h            INTEGER,
    context_strength       TEXT,
    range_status           TEXT,
    drift_status           TEXT,
    viability_score        REAL,
    signal_score           REAL,
    expected_net_median    REAL,
    model_version          TEXT,
    feature_contract       TEXT,
    execute_threshold_used REAL,
    default_cost_pct_used  REAL,
    min_net_capture_used   REAL,

    -- OUTCOME (preenchido na resolução)
    outcome_status                 TEXT DEFAULT 'pending', -- pending / hit / miss_timeout
    outcome_ts                     REAL,
    outcome_exit                   REAL,
    outcome_total                  REAL,
    outcome_duration_sec           REAL,
    best_total_spread_in_horizon   REAL,
    worst_total_spread_in_horizon  REAL,
    first_hit_ts                   REAL,
    target_reached_ts              REAL,      -- quando o spread atingiu o target publicado
    episode_closed_ts              REAL,      -- quando o tracker detectou fechamento (pode diferir)
    label_hit_shallow              INTEGER,
    label_hit_median               INTEGER,
    label_hit_deep                 INTEGER,

    UNIQUE(pair_key, ts_emitted)
);

CREATE INDEX idx_ledger_pending ON signal_ledger(pair_key, outcome_status)
    WHERE outcome_status = 'pending';
```

### Cooldown

Máximo 1 sinal ativo por `pair_key × direction × horizon × policy_name`. Após resolução, cooldown de horizon_sec / 2 antes de novo sinal para a mesma chave.

Nota de implementação: a unicidade operacional do cooldown é garantida pela **lógica de emissão** (runtime), não pelo índice SQL. O `UNIQUE(pair_key, ts_emitted)` no schema previne duplicatas exatas de timestamp, mas não substitui o cooldown. O implementador deve verificar sinais pending e cooldown ANTES de inserir.

`policy_name` = a política usada para decidir/publicar o sinal principal (ex.: `"median"` em `EXECUTE (MEDIAN)`). Cada row do ledger representa uma decisão de publicação, não o card inteiro. Os campos `target_*_shallow/median/deep` e `label_hit_*` permitem avaliar as 3 políticas a partir de um único row.

### Resolução

Estados terminais:
- **hit**: o mercado atingiu a condição operacional publicada dentro do horizonte
- **miss_timeout**: horizonte expirou sem atingir

Definição de hit: baseada em **target reached** (o spread atingiu o nível publicado), NÃO em "episódio fechou no tracker". O operador quer saber "o mercado chegou onde eu publiquei?", não "o detector de episódios marcou fechamento?". O spread pode atingir o target sem que o tracker detecte episódio formal.

label_hit por política — comparação **líquida** consistente:

```python
best_net_in_horizon = best_total_spread_in_horizon - default_cost_pct_used
label_hit_shallow = 1 if best_net_in_horizon >= target_net_shallow else 0
label_hit_median  = 1 if best_net_in_horizon >= target_net_median  else 0
label_hit_deep    = 1 if best_net_in_horizon >= target_net_deep    else 0
```

Nota: `best_total_spread_in_horizon` é bruto (entry + exit observado). A comparação com targets líquidos exige subtrair custo do bruto, ou comparar bruto com targets brutos. O schema armazena ambos para auditoria.

### Análise (Fase 2.9 — após 200+ sinais)

Brier score, reliability diagrams, cobertura ETA (coverage: % de outcomes dentro de cada quantil predito — Q50 deve ter ~50% coverage), Pinball Score, Winkler Score (penaliza intervalos largos + violações), hit rate por bucket, net_capture prevista vs realizada. Schema já suporta; análise espera volume.

---

## Fase 3 — SignalScore para Ranking

```python
def signal_score(prob, net_capture_median, support_2h, support_24h,
                 context_strength, eta_q50_minutes, eta_uncertainty_minutes, drift_status):
    viability = max(0.0, net_capture_median)
    support_factor = min(1.0, (support_2h / 3.0 + support_24h / 10.0) / 2.0)
    strength_mult = {"strong": 1.0, "normal": 0.7, "weak": 0.3}.get(context_strength, 0.1)
    drift_penalty = 0.5 if drift_status == "drifted" else 1.0
    eta_discount = 1.0 / (1.0 + max(eta_q50_minutes, 1.0) / 60.0)
    # Penalizar incerteza alta (Q75-Q25 grande = menos confiável)
    uncertainty_penalty = 1.0 / (1.0 + max(eta_uncertainty_minutes, 0.0) / 30.0)
    return prob * viability * support_factor * strength_mult * drift_penalty * eta_discount * uncertainty_penalty
```

Nota: `eta_q50_minutes` vem do multi-quantile head (condicional às features actuais).
`eta_uncertainty_minutes` = Q75-Q25 (dispersão). Sinais com ETA incerto são penalizados no ranking.

Natureza: **score de priorização visual** para o dashboard. NÃO é proxy de retorno esperado nem métrica de pesquisa. A avaliação rigorosa do sistema usa proper scoring rules (Brier, pinball) sobre o ledger. Ranking heurístico de UX e avaliação probabilística rigorosa são coisas distintas.

Nota de implementação: o score é calculado **pré-gate** (antes do filtro de publicação) e armazenado no ledger para análise. Sobre sinais publicados, `drift_penalty` é redundante (drift bloqueia EXECUTE) e `strength_mult` para "weak" não entra (gate exige normal/strong). Esses termos existem para uso interno de ranking pré-gate e para cenários futuros onde o score possa ser exposto em sinais WAIT.

---

## Fase 4 — Context Vector Fusion (CLSTM)

**Pré-requisito:** Fase 1 + baseline v3
**Timing:** DEPOIS do primeiro treino v3

### Arquitetura

LSTM + TemporalAttention para sequência (40 features) concatenado com context_proj para vetor episódico (~18-19 features) antes das heads prob/ETA. Categoricals como one-hot.

### Context vector (~19 features)

Contínuas (8): support_short, support_long, entry_median_2h, exit_median_2h, median_total_spread, episode_density_per_hour, median_episode_duration_sec, entry_exit_coherence_ratio

Binárias (2): entry_coherent, exit_coherent

One-hot (9): context_strength (3), range_status (3), entry_position (3)

### Critérios de aceitação

Gate: PR-AUC ≥ baseline **v3 sem context fusion** + ECE ≤ 0.10 (recalibrar Platt após treino).
O benchmark é o melhor modelo v3 (40 features, sem context vector), não o v2. Comparar com v2 seria bar baixo demais para aprovar fusão de contexto.
Monitoramento: Brier, ETA Pinball Score, ETA Coverage (Q50 target: 0.50), Winkler Score, Precision@top-20.

### Features de ledger (Fase 4+ — após 200+ sinais)

shrunk_hit_rate_pair = (α₀ + hits) / (α₀ + β₀ + n) com prior global. Embargo temporal de 1 horizonte.

---

## O que NÃO fazer

| Descartado | Motivo |
|---|---|
| DeepHit / Dynamic-DeepHit | Sem censoring nativo necessário, dados insuficientes para competing risks |
| TFT | Sem known future inputs |
| Multi-horizonte 12 heads | Fragmenta positivos → overfitting |
| 3 tabelas separadas para ledger | Over-engineering para v1 |
| expected_net como gate | Mistura grandezas com incertezas diferentes |
| Weibull head para ETA agora | Complexidade de censoring nativo; quantile regression resolve 90% do problema com menos risco |
| SmoothL1 para ETA (actual) | Ponto único sem incerteza, simétrica, ignora censoring — substituída por Pinball loss multi-quantile |

---

## Sequência de Implementação

```
Fase S: Sampling + Pipeline perf      ← ANTES de qualquer treino v3, ~4-6h
  S1: stride=5 na janela deslizante (reduz overlap 93% → 67%)
  S2: max_samples_per_pair=2000 (elimina dominância de pares activos)
  S5: diagnóstico de sampling no summary do dataset
  T1: batch_size=1024 (GPU 8x menos kernel launches)
  T2: num_workers=4 + persistent_workers (async data loading)
  T3: AMP autocast + GradScaler (FP16 Tensor Cores, 2x compute)
  T4: cudnn.benchmark + zero_grad(set_to_none)
  SC1: Scaffold + Relabel — separar build_dataset_bundle em:
       build_window_scaffold() → posições + features + metadata (1× por config)
       relabel_scaffold() → aplicar threshold/percentile (rápido, N×)
       Cache: dict[(seq_len, horizon, merge), WindowScaffold]
       Impacto: 71 windowing loops → 6 scaffolds + 63 relabels
  V1: Vectorização de build_feature_rows (numpy/pandas)
      → scaffold de ~60s → ~5s por config
  Resultado combinado:
    GPU: 30min → 3-5min
    Data prep: 2h+ → 4-8min
    Pipeline total 1GB: 2.5h+ → 8-10min
    Pipeline total 10GB: inviável → 16-20min
    Pipeline total 50GB: inviável → ~40min (RTX 3060)
Fase 1: Features v3 (40)            ← ANTES do treino, 2-3 sessões
  → Treinar com ETA antigo (single point, SmoothL1)
  → Resultado = BASELINE V3 (AUC + ETA MAE de referência)
Fase 2: Dashboard 3 políticas       ← PARALELO, 2-3 sessões
Fase 2.5: Signal Ledger             ← com Fase 2, 1-2 sessões
Fase 3: SignalScore                  ← com Fase 2, 1 sessão
Fase ETA-1: Multi-quantile head     ← APÓS baseline v3, ~3h
  ETA head 1→5 neurónios (Q10/Q25/Q50/Q75/Q90)
  Pinball loss (substitui SmoothL1)
  Monotonicity por softplus deltas
  Comparar com baseline: AUC deve manter, ETA calibração deve melhorar
Fase ETA-2: Censoring correction    ← COM ETA-1, ~2h
  Samples censurados (y=0) contribuem para loss de ETA
  Usa 100% dos dados em vez de 4.57%
Fase ETA-3: Label peak_ts           ← APÓS ETA-1+2 validados, ~2h
  Adicionar peak_ts ao TrackerEpisode
  Target: tempo até pico (não até close)
Fase ETA-4: Dashboard + calibração  ← COM ETA-3, ~2h
  Q10/Q25/Q50/Q75/Q90 condicionais no card
  Coverage, Pinball Score, Winkler Score como métricas
Fase 2.9: Análise do ledger         ← após 200+ sinais
Fase 4: CLSTM context fusion        ← após baseline v3, 3-4 sessões
Fase 4+: Features de ledger         ← após 200+ sinais + Fase 4

Futuro (6+ meses): Weibull ETA head (censoring nativo), cost por venue, execution_ledger, threshold ajustado por frequência, horizonte adaptativo por par
```

---

## Referências

- Stübinger et al. (2017). Pairs trading with mean-reverting jump-diffusion. https://ideas.repec.org/p/zbw/iwqwdp/102017.html
- Leung & Li (2016). Optimal Mean Reversion Trading. World Scientific.
- arXiv:2309.16008 (2023). Optimal Entry and Exit with Signature. https://arxiv.org/pdf/2309.16008
- Elliott, Van Der Hoek & Malcolm (2005). Stochastic spread model.
- Chen et al. (2016). Quantile forecasting GARCH for pair trading. https://doi.org/10.1016/j.iref.2016.09.005
- Palomar (HKUST). Pairs Trading lectures. https://palomar.home.ece.ust.hk/MAFS5310_lectures/slides_pairs_trading.pdf
- Ghosh et al. (2016). Contextual LSTM. https://arxiv.org/abs/1602.06291
- Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
- Ovadia et al. (2019). Can You Trust Your Model's Uncertainty? NeurIPS.
- Gneiting & Raftery (2007). Strictly Proper Scoring Rules. JASA.
- Dawid (1984). The Prequential Approach. JRSS.
- Efron & Morris (1977). Stein's Paradox. Scientific American.

# Métricas Completas para Monitoramento do LSTM — ArbML

---

## 1. Métricas de Treinamento do Modelo

### 1.1 Loss (Função de Perda)

| Métrica | O que mede | Quando usar |
|---------|-----------|-------------|
| **MSE (Mean Squared Error)** | Erro quadrático médio entre previsão e valor real | Previsão de valores contínuos de spread |
| **MAE (Mean Absolute Error)** | Erro absoluto médio, menos sensível a outliers que MSE | Quando spreads têm picos extremos e você não quer que distorçam o treino |
| **RMSE (Root Mean Squared Error)** | Raiz do MSE, na mesma unidade do spread | Interpretação direta em pontos percentuais de spread |
| **Huber Loss** | Combinação de MSE e MAE com threshold configurável | Quando há outliers frequentes nos dados de spread |
| **Binary Cross-Entropy** | Perda para classificação binária | Se o modelo classifica "entrar" vs "não entrar" |
| **Categorical Cross-Entropy** | Perda para classificação multiclasse | Se o modelo classifica "fraco / intermediário / forte" |

### 1.2 Curvas de Aprendizado

| Métrica | O que monitorar |
|---------|----------------|
| **Training Loss por época** | Deve diminuir de forma consistente |
| **Validation Loss por época** | Deve acompanhar o training loss sem divergir |
| **Gap entre Train e Validation Loss** | Gap crescente = overfitting |
| **Early Stopping patience** | Número de épocas sem melhoria antes de parar |
| **Learning Rate ao longo do treino** | Se usa scheduler, monitorar a taxa atual |
| **Gradient Norm** | Norma dos gradientes por época — detecta exploding/vanishing gradients |

---

## 2. Métricas de Regressão (Previsão de Valor de Spread)

Se o LSTM prevê o valor numérico do spread futuro:

| Métrica | Fórmula conceitual | Interpretação no ArbML |
|---------|-------------------|----------------------|
| **MAE** | média(\|real - previsto\|) | Erro médio em pontos percentuais de spread |
| **RMSE** | √(média((real - previsto)²)) | Penaliza mais erros grandes de previsão |
| **MAPE (Mean Absolute Percentage Error)** | média(\|real - previsto\| / \|real\|) × 100 | Erro percentual relativo ao spread real |
| **R² (Coeficiente de Determinação)** | 1 - (SS_res / SS_tot) | Quanto da variação do spread o modelo explica (0 a 1) |
| **Directional Accuracy** | % de vezes que o modelo acertou a direção do movimento | Crucial — mesmo errando o valor, acertar a direção já tem valor |
| **Max Error** | max(\|real - previsto\|) | Pior caso de erro — importante para gestão de risco |
| **Erro por faixa de spread** | MAE segmentado por faixa (0-2%, 2-4%, 4-6%, 6%+) | Saber se o modelo erra mais em spreads baixos ou altos |

---

## 3. Métricas de Classificação (Fraco / Intermediário / Forte)

Se o LSTM classifica o contexto atual do spread:

### 3.1 Métricas Básicas

| Métrica | O que mede |
|---------|-----------|
| **Accuracy** | % de acertos geral — cuidado com classes desbalanceadas |
| **Precision por classe** | De tudo que o modelo chamou de "forte", quanto realmente era forte |
| **Recall por classe** | De todos os momentos realmente fortes, quantos o modelo identificou |
| **F1-Score por classe** | Média harmônica entre precision e recall |
| **F1-Score Macro** | Média do F1 de todas as classes (trata classes igualmente) |
| **F1-Score Weighted** | Média ponderada pelo número de amostras por classe |

### 3.2 Métricas Avançadas de Classificação

| Métrica | Relevância para o ArbML |
|---------|------------------------|
| **Confusion Matrix** | Ver exatamente onde o modelo confunde (ex: classifica "forte" como "intermediário") |
| **Cohen's Kappa** | Concordância além do acaso — mais robusto que accuracy pura |
| **ROC-AUC por classe** | Capacidade de separação entre classes (one-vs-rest) |
| **PR-AUC (Precision-Recall AUC)** | Melhor que ROC-AUC quando classes são desbalanceadas |
| **Log Loss** | Qualidade das probabilidades emitidas pelo modelo |
| **Calibration Curve** | Se o modelo diz 80% de confiança, acerta ~80% das vezes? |
| **Top-K Accuracy** | Se a classe correta está entre as K mais prováveis |

### 3.3 Métricas Críticas para o Negócio

| Métrica | Por que importa |
|---------|----------------|
| **False Positive Rate para "forte"** | Quantas vezes o modelo diz "forte" e está errado — gera entradas ruins |
| **False Negative Rate para "forte"** | Quantas vezes era "forte" e o modelo não detectou — oportunidade perdida |
| **Precision de "forte"** | A mais importante: quando o modelo sinaliza "forte", precisa acertar |

---

## 4. Métricas Específicas de Séries Temporais

| Métrica | O que avalia |
|---------|-------------|
| **MASE (Mean Absolute Scaled Error)** | Compara o modelo com um baseline ingênuo (previsão = último valor) |
| **Autocorrelação dos resíduos** | Resíduos devem ser aleatórios — padrão nos resíduos = modelo incompleto |
| **Ljung-Box Test (p-value)** | Teste estatístico para autocorrelação nos resíduos |
| **SMAPE (Symmetric MAPE)** | MAPE simétrico, evita distorção quando valores reais são próximos de zero |
| **Tracking Signal** | Detecta viés sistemático do modelo ao longo do tempo |
| **Walk-Forward Validation Score** | Performance em validação que respeita a ordem temporal |
| **Erro por horizonte de previsão** | Erro separado por 5min, 15min, 30min, 1h, 2h à frente |

---

## 5. Métricas de Qualidade dos Dados (Input)

### 5.1 Integridade dos Dados

| Métrica | O que monitorar |
|---------|----------------|
| **% de dados faltantes por feature** | Gaps nos dados de spread das corretoras |
| **% de timestamps fora de ordem** | Dados chegando fora de sequência |
| **Frequência real vs esperada** | Se espera dado a cada 1s, está realmente chegando a cada 1s? |
| **Latência de ingestão** | Tempo entre o evento na corretora e chegada no pipeline |
| **Contagem de duplicatas** | Registros duplicados distorcem o treinamento |
| **Taxa de dados corrompidos** | Valores impossíveis (spreads de 500%, preços negativos) |

### 5.2 Distribuição dos Dados

| Métrica | O que monitorar |
|---------|----------------|
| **Média e desvio padrão do spread por moeda** | Mudança indica regime diferente |
| **Skewness (assimetria)** | Distribuição dos spreads é simétrica ou enviesada? |
| **Kurtosis (curtose)** | Frequência de spreads extremos |
| **Percentis (P5, P25, P50, P75, P95)** | Faixas de comportamento típico |
| **Contagem de outliers (>3σ)** | Quantos pontos estão fora do padrão |

---

## 6. Métricas de Features (Engenharia de Variáveis)

| Métrica | O que avalia |
|---------|-------------|
| **Feature Importance (permutation)** | Quais features mais impactam a previsão |
| **Correlação entre features** | Features muito correlacionadas são redundantes |
| **Variância de cada feature** | Feature com variância zero é inútil |
| **Missing rate por feature** | Features com muitos dados faltantes degradam o modelo |
| **Distribuição de cada feature ao longo do tempo** | Detectar data drift nas variáveis de entrada |
| **VIF (Variance Inflation Factor)** | Multicolinearidade entre features |

---

## 7. Métricas de Arquitetura e Hiperparâmetros do LSTM

| Parâmetro / Métrica | O que monitorar |
|---------------------|----------------|
| **Número de hidden units** | Performance vs custo computacional |
| **Número de camadas LSTM** | Mais camadas nem sempre = melhor |
| **Dropout rate** | Impacto no gap train/validation |
| **Sequence length (janela)** | Qual tamanho de janela temporal dá melhor resultado |
| **Batch size** | Estabilidade do treinamento |
| **Learning rate** | Convergência vs velocidade |
| **Número de épocas até convergência** | Eficiência do treinamento |
| **Tempo de treinamento por época** | Viabilidade operacional |
| **Tamanho do modelo (parâmetros)** | Capacidade vs risco de overfitting |
| **Cell state statistics** | Valores médios e variância dos cell states — detecta saturação |

---

## 8. Métricas de Produção e Operação (Deploy)

### 8.1 Performance em Tempo Real

| Métrica | O que monitorar |
|---------|----------------|
| **Latência de inferência (ms)** | Tempo para o modelo gerar uma previsão |
| **P99 de latência** | Latência no pior 1% dos casos |
| **Throughput (previsões/segundo)** | Quantas moedas o modelo consegue analisar por segundo |
| **Uso de memória (RAM/GPU)** | Consumo de recursos |
| **Uso de CPU/GPU (%)** | Carga computacional |
| **Uptime do serviço** | Disponibilidade do modelo em produção |
| **Taxa de erro de inferência** | % de previsões que falharam por erro técnico |

### 8.2 Data Drift e Model Drift

| Métrica | O que monitorar |
|---------|----------------|
| **PSI (Population Stability Index)** | Mudança na distribuição dos inputs vs dados de treino |
| **KS Test (Kolmogorov-Smirnov)** | Teste estatístico para drift na distribuição |
| **CSI (Characteristic Stability Index)** | Drift por feature individual |
| **Performance decay** | Queda da métrica principal ao longo de dias/semanas |
| **Comparação janela fixa vs janela móvel** | O modelo treinado há 7 dias ainda performa igual? |
| **Concept Drift detection** | A relação entre features e target mudou? |
| **Frequência de retraining necessário** | De quanto em quanto tempo o modelo precisa ser retreinado |

---

## 9. Métricas de Negócio Específicas do ArbML

### 9.1 Qualidade da Leitura de Spread

| Métrica | O que mede |
|---------|-----------|
| **Acurácia de classificação de faixa** | Quando o modelo diz "spread na faixa de 6%", está certo? |
| **Erro médio na estimativa de faixa recorrente** | Diferença entre a faixa recorrente estimada e a observada |
| **Concordância com leitura manual** | % de vezes que modelo e operador concordam |
| **Tempo economizado por análise** | Redução de tempo vs análise manual |

### 9.2 Qualidade das Sinalizações

| Métrica | O que mede |
|---------|-----------|
| **Taxa de sinalização "forte" que resultou em spread favorável** | Das vezes que sinalizou "forte", quantas realmente evoluíram bem |
| **Taxa de oportunidades perdidas** | Momentos fortes que o modelo não sinalizou |
| **Falsos alarmes por hora** | Quantas vezes por hora o modelo sinaliza sem fundamento |
| **Tempo médio entre sinalização e melhor momento de entrada** | Quão antecipado o modelo é |
| **Distribuição de confiança nas sinalizações** | O modelo está calibrado ou sempre diz "80% de certeza"? |

### 9.3 Métricas de Recorrência (Core do Projeto)

| Métrica | O que mede |
|---------|-----------|
| **Precisão na detecção de padrão recorrente** | O modelo identifica corretamente quando uma faixa se repete |
| **Estabilidade da faixa de entrada detectada** | A faixa estimada muda a cada minuto ou é consistente? |
| **Estabilidade da faixa de saída detectada** | Mesma lógica para o lado de saída |
| **Coerência entre janelas temporais** | A leitura de 2h contradiz a de 24h? Com que frequência? |
| **Velocidade de adaptação a novo regime** | Quando a moeda muda de comportamento, quão rápido o modelo percebe |

### 9.4 Métricas de Execução Prática

| Métrica | O que mede |
|---------|-----------|
| **Spread estimado vs spread executável real** | Diferença entre o que o modelo vê e o que é possível executar |
| **Impacto da liquidez na previsão** | O modelo funciona pior em moedas com baixa liquidez? |
| **Performance por par de corretoras** | O modelo é melhor em certos pares do que outros? |
| **Performance por moeda** | Algumas moedas são mais previsíveis que outras? |
| **Performance por horário** | O modelo funciona melhor em certos horários do mercado? |
| **Performance por volatilidade** | Em momentos de alta volatilidade, o modelo degrada? |

---

## 10. Métricas de Comparação com Baselines

| Baseline | Comparar com |
|----------|-------------|
| **Naive (último valor)** | O LSTM supera a previsão de "spread vai continuar igual"? |
| **Média móvel simples** | O LSTM supera uma média móvel das últimas N observações? |
| **Regra fixa de percentil** | O LSTM supera uma regra de "entrar quando spread > P75"? |
| **Random Forest / XGBoost** | O LSTM supera modelos não-sequenciais nos mesmos dados? |
| **Modelo ARIMA / GARCH** | O LSTM supera modelos estatísticos clássicos de séries temporais? |
| **Ensemble (LSTM + outros)** | Combinar LSTM com outros modelos melhora o resultado? |

---

## 11. Dashboard de Monitoramento — Métricas Prioritárias

Para operação diária, as métricas mais críticas para ter em um painel:

### Tier 1 — Olhar Sempre
- Loss de validação (tendência)
- Precision de "contexto forte"
- Latência de inferência
- PSI (drift dos dados)
- Taxa de falsos alarmes
- Taxa de sinalização forte que se confirmou

### Tier 2 — Olhar Diariamente
- Directional Accuracy
- MAE por faixa de spread
- Gap train/validation loss
- Performance por moeda (top 5 e bottom 5)
- Distribuição de confiança das previsões

### Tier 3 — Olhar Semanalmente
- Comparação com baselines
- Feature importance (mudou?)
- Autocorrelação dos resíduos
- Performance por horário e volatilidade
- Necessidade de retraining

---

## 12. Alertas Automáticos Recomendados

| Condição | Ação |
|----------|------|
| Validation loss subiu >15% em relação à melhor época | Investigar overfitting ou data drift |
| PSI > 0.2 em qualquer feature principal | Verificar se distribuição dos dados mudou |
| Latência P99 > 500ms | Otimizar modelo ou infraestrutura |
| Precision de "forte" caiu abaixo de 60% | Considerar retraining |
| Mais de 5 falsos alarmes por hora | Ajustar thresholds de classificação |
| Taxa de dados faltantes > 5% | Investigar pipeline de ingestão |
| Gradient norm > 100 durante treino | Aplicar gradient clipping |
| Accuracy em produção caiu >10% vs validação | Concept drift — retreinar com dados recentes |

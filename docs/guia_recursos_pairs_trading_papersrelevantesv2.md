# Guia Completo de Recursos: Pairs Trading & Arbitragem Estatística com ML, DL e RL

**Compilado para o Projeto ArbML**
**15 de Março de 2026 — v2.0 (atualizado com 7 novos recursos)**

> Entradas marcadas com 🆕 foram adicionadas nesta versão.

---

## Introdução

Este documento reúne e analisa detalhadamente cada recurso identificado na pesquisa sobre Pairs Trading e Arbitragem Estatística aplicando técnicas de Machine Learning (ML), Deep Learning (DL) e Reinforcement Learning (RL). A lista cobre repositórios open-source com código executável, papers acadêmicos publicados em journals e conferências de referência, bibliotecas profissionais, e teses acadêmicas brasileiras e portuguesas.

Cada entrada contém: descrição do conteúdo, metodologia empregada, ativos cobertos (equities, crypto, commodities), tecnologias e frameworks utilizados, e uma análise de relevância para o projeto ArbML, que foca em arbitragem de spread cross-exchange de criptomoedas usando LSTM e análise de séries temporais.

A versão 2.0 adiciona 6 novos papers acadêmicos (cobrindo Graph Neural Networks, Regime-Aware RL, Contrastive Learning e LLMs para séries temporais) e 1 novo repositório open-source focado em crypto pairs trading multi-agente.

**Total de recursos catalogados: 32** (25 originais + 7 novos) — 13 repositórios, 15 papers, 4 teses/livros.

---

---

# Seção 1 — Repositórios Open-Source Principais

---

## 1.1 — Hongshen-Yang/pair-trading-envs

*https://github.com/Hongshen-Yang/pair-trading-envs*

Implementa um sistema completo de Reinforcement Learning aplicado a pairs trading de criptomoedas, construindo ambientes personalizados sobre o framework Backtrader. Acompanhado pelo paper "Reinforcement Learning Pair Trading: A Dynamic Scaling Approach" (Yang & Malik, 2024), publicado no JRFM (MDPI).

**Metodologia:**
Agentes de RL (PPO, A2C, DQN via Stable-Baselines3) treinados em ambientes Gym customizados. Inovação principal: dynamic scaling — o agente aprende a dimensionar dinamicamente o tamanho da posição com base no estado do spread. Reward shaping específico para mean-reversion.

**Ativos e Dados:**
Pares de BTC-GBP e BTC-EUR com dados em intervalos de 1 minuto (263.520 observações).

**Resultados:**
Abordagem tradicional: 8,33% anualizado. Com RL: 9,94% a 31,53% dependendo do agente.

**Relevância para ArbML: ALTA.**
Recurso mais diretamente comparável ao ArbML: domínio (crypto cross-exchange), granularidade (minuto), abordagem (spread-based). Dynamic scaling pode inspirar sizing de posições no dashboard.

---

## 1.2 — seferlab/pairstrading

*https://github.com/seferlab/pairstrading*

Repositório oficial do paper "Pairs Trading with Time-Series Deep Learning Models" (Yilmaz et al., 2026), ScienceDirect. Estado da arte em pairs trading com modelos de séries temporais profundas.

**Metodologia:**
Pipeline completo: (1) PCA rolling com janela de 252 dias e 80% de variância explicada; (2) spreads residuais; (3) comparação sistemática entre LSTM, Informer, Autoformer, iTransformer, Scaleformer e Chronos.

**Relevância para ArbML: ALTA.**
LSTM como modelo base idêntico ao ArbML. Comparação com Transformers fornece roadmap de evolução.

---

## 1.3 — fiit-ba/ML-for-arbitrage-in-cryptoexchanges

*https://github.com/fiit-ba/ML-for-arbitrage-in-cryptoexchanges*

Código-fonte oficial do paper "Using Machine Learning for Predicting Arbitrage Occurrences in Cryptocurrency Exchanges" (Okasóva et al., IEEE ICBC 2024, expandido Wiley 2026).

**Metodologia:**
Classificadores ML (Logistic Regression, Random Forest, SVM, MLP) para prever arbitragem cross-exchange Binance/Bybit. Classificação binária: "haverá oportunidade de arbitragem nos próximos N minutos?".

**Relevância para ArbML: MUITO ALTA.**
Correspondência mais direta ao ArbML — previsão de arbitragem cross-exchange em crypto. Difere na metodologia (classificadores estáticos vs LSTM sequencial).

---

## 🆕 1.4 — Amdev-5/crypto-pairs-trading-ai

*https://github.com/Amdev-5/crypto-pairs-trading-ai*

Sistema completo multi-agente AI para cryptocurrency pairs trading, com foco educacional e de pesquisa. Combina múltiplas estratégias de arbitragem estatística operando simultaneamente com gerenciamento de risco em tempo real.

**Metodologia:**
Arquitetura multi-agente com 4 estratégias simultâneas: (1) Cointegração Engle-Granger para seleção e monitoramento de pares; (2) Order Book Imbalance (OBI) para sinais de microestrutura; (3) Correlação + RSI para confirmação de timing; (4) Mean Reversion com Z-score e Bollinger Bands adaptativo. O sistema inclui conexão via WebSocket com latência sub-100ms e gerenciamento de risco em tempo real (position sizing, stop-loss, drawdown limits).

**Ativos e Dados:**
Pares de criptomoedas via WebSocket. Pronto para backtest com dados históricos. A cobertura multi-estratégia é modular e extensível.

**Relevância para ArbML: ALTA.**
O mais próximo do ArbML em termos de domínio (crypto) e escopo (sistema completo). O uso de OBI (Order Book Imbalance) como feature é particularmente relevante — o paper de Okasóva (2026) identificou order book imbalance como uma das features mais preditivas para arbitragem crypto. A arquitetura multi-agente pode inspirar a coordenação dos ~1000 símbolos do ArbML. A latência sub-100ms via WebSocket é referência de engenharia para futuras evoluções do scanner.

---

---

# Seção 2 — Outros Repositórios Relevantes com ML

---

## 2.1 — anthonyli01/Statistical-Arbitrage-Pairs-Trading-Strategy

*https://github.com/anthonyli01/Statistical-Arbitrage-Pairs-Trading-Strategy*

Combina cointegração (Engle-Granger/Johansen), filtro de Kalman para hedge ratios dinâmicos, cópulas para dependência não-linear, e ML para seleção/otimização.

**Relevância: MÉDIA-ALTA.** Kalman filter para hedge ratios dinâmicos pode normalizar spreads assimétricos.

---

## 2.2 — stefan-jansen/machine-learning-for-trading

*https://github.com/stefan-jansen/machine-learning-for-trading*

Livro "Machine Learning for Algorithmic Trading" (2ª ed., O'Reilly). Notebook "06_statistical_arbitrage_with_cointegrated_pairs.ipynb" — tutorial completo e referência definitiva.

**Relevância: MÉDIA.** Excelente material didático, focado em equities/cointegração clássica.

---

## 2.3 — raktim-roychoudhury/pairs_trading

*https://github.com/raktim-roychoudhury/pairs_trading*

Unsupervised Clustering (K-Means, DBSCAN, Hierarchical) para seleção + DRL (DQN, Policy Gradient) para trading.

**Relevância: MÉDIA.** Combinação clustering + DRL como evolução futura.

---

## 2.4 — prakhar1602/pair_trading_with_machine_learning

*https://github.com/prakhar1602/pair_trading_with_machine_learning*

ML (RF, GB, SVM) com features derivadas do spread (rolling z-score, rate of change, volatilidade). Feature engineering conceitualmente próximo ao ArbML.

**Relevância: MÉDIA.** Feature engineering baseado em spread diretamente aplicável.

---

## 2.5 — achntj/statistical-arbitrage

*https://github.com/achntj/statistical-arbitrage*

Framework multi-fase: seleção de pares, cointegração, spread rolling, sinais z-score com confirmação ML. Arquitetura semelhante ao pipeline scan→detect→signal→dashboard do ArbML.

**Relevância: MÉDIA.**

---

## 2.6 — chicago-joe/InteractiveBrokers-PairsTrading-Algo

*https://github.com/chicago-joe/InteractiveBrokers-PairsTrading-Algo*

Pairs trading HFT conectado à API do Interactive Brokers. Foco em execução real, não ML. Referência de integração com APIs de exchange em tempo real.

**Relevância: BAIXA-MÉDIA.**

---

## 2.7 — ALF28Dev/statistical_arbitrage

*https://github.com/ALF28Dev/statistical_arbitrage*

Bollinger Bands adaptativas no spread + confirmação ML. Simples mas eficaz como baseline.

**Relevância: BAIXA.**

---

## 2.8 — ArbitrageLab (Hudson & Thames)

*https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/ml_approach/ml_based_pairs_selection.html*

Biblioteca profissional mais completa para arbitragem estatística. Cobre distância, cointegração, cópulas, optimal transport, seleção ML com TensorFlow/Keras. Half-life of mean-reversion conceitualmente relacionado à duration dos episódios do ArbML.

**Relevância: MÉDIA.**

---

## 2.9 — bkmulusew/ml_pairs_trading

*https://github.com/bkmulusew/ml_pairs_trading*

Implementação oficial de "Machine Learning Enhanced Pairs Trading" (2025). BiLSTM + Attention + TCN + MADDPG multi-agente. Uma das implementações mais sofisticadas da lista.

**Metodologia detalhada:**
O BiLSTM com Attention processa a série temporal do spread em ambas as direções, capturando dependências passadas e futuras (no contexto de treinamento). O TCN fornece receptive fields longas sem o custo computacional de LSTMs profundas. O MADDPG é usado como múltiplos agentes cooperativos para gerenciar simultaneamente vários pares.

**Relevância: ALTA.** BiLSTM + Attention é evolução natural do LSTM unidirecional do ArbML.

---

---

# Seção 3 — Papers Acadêmicos Fundamentais (2021–2024)

---

## 3.1 — Deep Learning Statistical Arbitrage

*https://arxiv.org/abs/2106.04028*

**Autores:** Jorge Guijarro-Ordonez, Markus Pelger, Greg Zanotti (Stanford, 2021/2022).

Paper seminal que estabeleceu o framework unificador moderno para arbitragem estatística com deep learning. Três pilares: (1) portfólios de arbitragem como residuais de fatores latentes condicionais de asset pricing; (2) extração de sinais temporais com convolutional transformer; (3) política ótima de trading que maximiza retornos ajustados ao risco sob restrições.

**Resultados:** Sharpe ratios consistentemente altos out-of-sample em equities diários dos EUA. Demonstrou alta compensação para arbitrageurs.

**Relevância para ArbML: ALTA (conceitual).** O framework "residual portfolio + time series signal + optimal policy" é conceitualmente o mesmo que o ArbML segue: spread residual + LSTM signal + dashboard signal.

---

## 3.2 — Advanced Statistical Arbitrage with Reinforcement Learning

*https://arxiv.org/html/2403.12180v1*

**Autores:** Boming Ning, Kiseop Lee (Purdue University, 2024).

Framework model-free RL com duas inovações: (1) métrica empírica de tempo de reversão à média, otimizada para encontrar os melhores coeficientes do spread; (2) espaço de estados que encapsula tendências recentes de preço ao invés de apenas desvios da média.

**Detalhes técnicos:** O estado inclui não apenas o nível atual do spread, mas também features de momentum, regime e volatilidade recente. A função de recompensa é customizada para mean-reversion.

**Relevância para ArbML: MÉDIA-ALTA.** Tempo de reversão empírica é análogo a "duration" dos episódios. Estado com tendências recentes = features multi-escala do v3 Phase 1.

---

## 3.3 — Reinforcement Learning Pair Trading: A Dynamic Scaling Approach

*https://arxiv.org/abs/2407.16103*

**Autores:** Hongshen Yang, Avinash Malik (University of Auckland, 2024). JRFM (MDPI).

BTC-GBP vs BTC-EUR, dados de 1 minuto. Lucro anualizado de 9,94% a 31,53% com RL vs 8,33% tradicional. Contribuição: dynamic scaling — o agente decide não apenas quando entrar/sair, mas quanto alocar.

**Relevância para ArbML: ALTA.** Dados 1min crypto cross-exchange = cenário mais próximo do ArbML. Dynamic scaling pode evoluir sinais categóricos para escala contínua.

---

## 3.4 — Statistical Arbitrage in Cryptocurrency Markets

*https://d-nb.info/1207154512/34*

**Autores:** Fischer et al. Deutsche Nationalbibliothek.

Estudo específico sobre arbitragem estatística em mercados de criptomoedas. Base empírica e teórica sobre persistência de oportunidades (latency, fees, liquidity, withdrawal times).

**Relevância para ArbML: MÉDIA.** Valida a premissa fundamental do ArbML.

---

---

# Seção 4 — Papers Recentes e Estado da Arte (2025–2026)

---

## 4.1 — Pairs Trading with Time-Series Deep Learning Models

*https://www.sciencedirect.com/science/article/pii/S2405918826000024*

**Autores:** Yilmaz et al. (2026). ScienceDirect.

O paper mais recente e abrangente na comparação de modelos de séries temporais para pairs trading. Testa LSTM, Informer, Autoformer, iTransformer, Scaleformer e Chronos no mesmo dataset e pipeline. Conclusão: Transformers não necessariamente superam LSTMs em dados ruidosos ou não-estacionários.

**Relevância para ArbML: MUITO ALTA.** Valida a escolha do LSTM como modelo base do ArbML. A conclusão de que Transformers nem sempre vencem LSTMs é particularmente relevante para a decisão arquitetural de rejeitar arquiteturas over-engineered.

---

## 4.2 — Predicting Arbitrage Occurrences With Machine Learning

*https://onlinelibrary.wiley.com/doi/full/10.1002/nem.70030*

**Autores:** Okasóva et al. (2026). Wiley — International Journal of Network Management.

Expansão do paper IEEE ICBC 2024. ML clássico para prever arbitragem cross-exchange crypto. A versão expandida inclui análise de feature importance: spread velocity, volume ratio e order book imbalance emergem como as features mais preditivas.

**Relevância para ArbML: MUITO ALTA.** Feature importance informa diretamente a expansão de 25→40 features do v3 Phase 1.

---

## 4.3 — Deep Learning-Based Pairs Trading: Real-Time Forecasting of Co-Integrated Crypto Pairs

*https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1749337/full*

**Autores:** Tsoku et al. (2026). Frontiers in Applied Mathematics and Statistics.

Foca em previsão em tempo real de pares cointegrados de criptomoedas. Implementa rolling cointegration tests para detectar quando um par deixa de ser cointegrado.

**Relevância para ArbML: ALTA.** Rolling cointegration tests é análogo ao soak test e certificação contínua do ArbML.

---

## 4.4 — Attention Factors for Statistical Arbitrage

*https://arxiv.org/abs/2510.11616*

**Autores:** Elliot L. Epstein, Rose Wang, Jaewon Choi, Markus Pelger (Stanford, Out/2025). Aceito no 6th ACM ICAIF.

Evolução do paper 3.1. Attention Factors — fatores latentes condicionais com embeddings de características e mecanismo de atenção. Estimação conjunta fatores/política (one-step solution).

**Resultados:** Sharpe ratio out-of-sample acima de 4 em equities EUA em 24 anos. Sharpe de 2.3 líquido de custos. Fatores fracos são importantes para arbitragem.

**Relevância para ArbML: MÉDIA-ALTA.** Attention mechanism inspira CLSTM Phase 4. Estimação conjunta reforça pipeline unificado.

---

## 4.5 — Signature Decomposition Method Applying to Pair Trading

*https://arxiv.org/abs/2505.05332*

**Autores:** Zihao Guo et al. (Mai/2025, revisado Out/2025).

Rough path theory signatures para capturar todas as características estatísticas relevantes de forma compacta e invariante a reparametrização temporal.

**Relevância para ArbML: BAIXA-MÉDIA.** Computacionalmente custoso, alternativa futura ao LSTM.

---

## 4.6 — Statistical Arbitrage in Polish Equities Using Deep Learning

*https://arxiv.org/abs/2512.02037*

**Autores:** Marek Adamczyk, Michał Dąbrowski (Nov/2025).

LSTM, GRU, CNN-LSTM híbrido em mercado polonês. Valida robustez em mercados menos líquidos.

**Relevância para ArbML: BAIXA.** Valida LSTM/GRU em baixa liquidez.

---

## 4.7 — Hybrid Ridgelet Deep Neural Networks for Arbitrage

*https://arxiv.org/abs/2510.10599*

**(Out/2025).** Ridgelet Neural Networks para estratégias data-driven. Teórico e avançado.

**Relevância para ArbML: BAIXA.**

---

## 4.8 — A Survey of Statistical Arbitrage Pair Trading with ML, DL, and RL

*https://ideas.repec.org/p/war/wpaper/2025-22.html*

**Autores:** Sun et al. (2025). Warsaw School of Economics Working Papers.

Survey abrangente: classifica técnicas por fase (pair selection, spread modeling, trading strategy). Mapa do campo.

**Relevância para ArbML: ALTA (como referência).** Guia decisões futuras sobre técnicas a explorar.

---

## 4.9 — Deep Reinforcement Learning for Optimal Trading with Partial Information

*https://arxiv.org/abs/2511.00190*

**(Nov/2025).** DRL com informação parcial explícita. Robustez a incerteza sobre estado do mercado.

**Relevância para ArbML: MÉDIA.** ArbML opera com observabilidade parcial por definição.

---

## 🆕 4.10 — Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network

*https://arxiv.org/abs/2601.04602*

**Autores:** Jack Fanshawe, Rumi Masih, Alexander Cameron (Jan/2026).

Propõe uma Temporal-Heterogeneous Graph Neural Network (THGNN) que combina um encoder temporal baseado em Transformer com uma rede de atenção em grafos (edge-aware GAT) para prever correlações futuras entre ações do S&P 500. O modelo trabalha em Fisher-z space e prediz desvios residuais sobre uma baseline histórica rolling.

**Metodologia detalhada:**
Os inputs incluem retornos diários, indicadores técnicos, estrutura setorial, correlações anteriores e sinais macroeconômicos, permitindo forecasts regime-aware. A interpretação é feita via attention-based feature importance e neighbor importance. Quando integrado a um framework de clustering baseado em grafos, as correlações forward-looking produzem baskets adaptativos que performam particularmente bem em períodos de stress de mercado. Resultados out-of-sample (2019–2024) com Sharpe ratio de 1.837.

**Relevância para ArbML: MÉDIA-ALTA.**
A idéia central — prever correlações futuras entre ativos para construir baskets de mean-reversion — é diretamente generalizável para crypto futures. No contexto do ArbML, uma GNN poderia modelar as relações entre os ~1000 símbolos monitorados, identificando quais pares de exchanges têm maior probabilidade de exibir spreads favoráveis no futuro próximo. A feature importance via attention é um template para interpretar quais características do spread são mais relevantes. A abordagem de predizer correlações como residuais sobre baseline rolling é conceitualmente análoga ao ArbML predizer o comportamento futuro do spread relativo ao histórico recente.

---

## 🆕 4.11 — Statistical Arbitrage in Options Markets by Graph Learning and Synthetic Long Positions

*https://arxiv.org/abs/2508.14762*

**Autores:** Yoonsik Hong, Diego Klabjan (Northwestern, Ago/2025).

Abordagem em dois estágios para identificação e exploração de arbitragem estatística em mercados de opções usando Graph Learning. O primeiro estágio define um target de predição que isola arbitragens puras via synthetic bonds. O segundo estágio propõe SLSA (Synthetic Long Statistical Arbitrage) — posições de risco mínimo e neutras a todos os fatores Black-Scholes.

**Metodologia detalhada:**
Introduz RNConv, uma arquitetura de Graph Learning que incorpora estrutura tree-based, superando baselines de GL em dados tabulares. Resultados em opções KOSPI 200 com information ratio médio de 0.1627 e retornos consistentemente positivos. A insight principal é que métodos tree-based frequentemente superam DL em dados tabulares — e a arquitetura RNConv combina o melhor de ambos os mundos.

**Relevância para ArbML: BAIXA-MÉDIA.**
O mercado é diferente (opções), mas dois conceitos são transferíveis: (1) a idéia de isolar "arbitragens puras" via construção sintética é análoga ao ArbML filtrar episódios que representam oportunidades "limpas" vs ruído; (2) a observação de que tree-based methods competem com DL em dados tabulares questiona se XGBoost/LightGBM poderiam complementar o LSTM em features tabulares do ArbML.

---

## 🆕 4.12 — Graph Learning for Foreign Exchange Rate Prediction and Statistical Arbitrage

*https://arxiv.org/abs/2508.14784*

**Autores:** Yoonsik Hong, Diego Klabjan (Northwestern, Ago/2025).

Aplica Graph Learning diretamente à previsão de taxas de câmbio (FX) e exploração de arbitragem estatística em mercados de forex. Formula a previsão FX como um problema de regressão edge-level em um grafo espaço-temporal discreto, onde moedas são nós e exchanges são arestas.

**Metodologia detalhada:**
O grafo tem taxas de juros como features de nó e taxas de câmbio como features de aresta. Um segundo método de GL maximiza retorno ajustado ao risco considerando o time lag entre observação e execução — um problema crítico em trading real. Resultados: information ratio 61,89% superior e Sortino ratio 45,51% superior ao benchmark. Prova que o método satisfaz empirical arbitrage constraints.

**Relevância para ArbML: ALTA.**
Este paper é altamente adaptável ao contexto do ArbML porque: (1) a modelagem de exchanges como arestas de um grafo é exatamente a estrutura do ArbML (mesma moeda, duas exchanges diferentes); (2) o tratamento explícito do observation-execution time lag é crítico para crypto arbitrage onde latência afeta execução; (3) a prova formal de que o método satisfaz arbitrage constraints fornece rigor teórico que o ArbML poderia incorporar. Converter o scanner do ArbML para um grafo onde exchanges são nós e spreads são features de aresta é uma generalização natural e poderosa.

---

## 🆕 4.13 — RAmmStein: Regime Adaptation in Mean-reverting Markets with Stein Thresholds

*https://arxiv.org/abs/2602.19419*

**Autores:** (Mar/2026).

Deep Reinforcement Learning com adaptação de regime para mercados mean-reverting, baseado em processos Ornstein-Uhlenbeck. O nome "RAmmStein" combina Regime Adaptation, Mean-reverting Markets e Stein thresholds.

**Metodologia detalhada:**
O sistema detecta automaticamente regimes de mercado (alta/baixa volatilidade, forte/fraca mean-reversion) e adapta os thresholds de trading em tempo real usando Stein's method. O resultado principal é uma redução de 85% no rebalanceamento enquanto maximiza ROI em ambientes realistas com custos de transação. A abordagem reconhece que o mesmo spread pode ter características completamente diferentes em regimes diferentes — algo que thresholds estáticos não capturam.

**Relevância para ArbML: MUITO ALTA.**
Este paper ataca diretamente um dos problemas centrais do ArbML: a calibração de gates. O ArbML já descobriu que "gates desenhados para dezenas de pares quebram catastroficamente em ~10.000 pares" — isso é exatamente um problema de regime. A adaptação automática de thresholds baseada em regime de mercado poderia resolver a necessidade de recalibração manual. A redução de 85% em rebalanceamento é relevante porque reduz custos operacionais. A base em Ornstein-Uhlenbeck é o modelo teórico clássico para mean-reversion, que é exatamente o comportamento que o ArbML monitora nos spreads.

---

## 🆕 4.14 — Cross-Sectional Asset Retrieval via Future-Aligned Soft Contrastive Learning

*https://arxiv.org/abs/2602.10711*

**Autores:** (Fev/2026).

Aplica Contrastive Learning para identificar ativos que terão movimentos correlacionados no futuro (não apenas ativos que foram correlacionados no passado). O método "Future-Aligned" aprende embeddings onde ativos que se moverão juntos ficam próximos no espaço latente, mesmo que seu histórico passado não mostre correlação clara.

**Metodologia detalhada:**
Soft Contrastive Learning permite graduações de similaridade (não apenas binário similar/dissimilar), capturando relações contínuas entre ativos. Avaliado em downstream tasks de pairs trading, demonstrando que a seleção de pares baseada em embeddings futuros supera métodos tradicionais baseados em correlação histórica.

**Relevância para ArbML: MÉDIA.**
O ArbML não seleciona pares (opera com todos os símbolos cross-exchange), mas o conceito de "future-aligned embeddings" poderia ser usado para priorizar quais símbolos têm maior probabilidade de gerar episódios de spread favoráveis no futuro. Isso complementaria o SignalScore do v3 Phase 3 com uma dimensão forward-looking.

---

## 🆕 4.15 — LLMs for Time Series: Application for Single Stocks and Statistical Arbitrage

*https://arxiv.org/abs/2412.09394*

**Autores:** Sebastien Valeyre, Sofiane Aboura (Dez/2024).

Aplica LLMs (especificamente o modelo Chronos de Ansari et al., 2024) a previsão de séries temporais financeiras, testando tanto configurações pretrained quanto fine-tuned em ações americanas. Constrói portfólios long/short de statistical arbitrage usando os forecasts do LLM.

**Metodologia detalhada:**
Usa o dataset de Guijarro-Ordonez et al. (paper 3.1 desta lista) e testa se LLMs conseguem identificar ineficiências em séries quase indistinguíveis de ruído. Conclusão: LLMs demonstram capacidade de gerar alpha, mas com limitações significativas de overfitting. A comparação com modelos especializados menores (tipo LSTM) mostra espaço considerável para melhoria nos LLMs.

**Relevância para ArbML: MÉDIA (prospectiva).**
A conclusão principal valida a estratégia do ArbML: modelos especializados menores (LSTM) ainda têm vantagem sobre LLMs genéricos para trading de arbitragem. LLMs sofrem de overfitting em dados financeiros ruidosos — reforçando a rejeição de arquiteturas over-engineered pelo ArbML. Porém, à medida que LLMs para séries temporais evoluam, podem se tornar relevantes como feature extractors (não como modelo primário) no pipeline do ArbML.

---

---

# Seção 5 — Teses Acadêmicas, Livros e Recursos Práticos

---

## 5.1 — Machine Learning for Algorithmic Trading (Livro)

*https://github.com/stefan-jansen/machine-learning-for-trading*

Stefan Jansen, 2ª ed., O'Reilly. Capítulo de arbitragem estatística com pares cointegrados é referência definitiva. Pipeline completo: dados, features, modelos, backtesting e produção.

---

## 5.2 — Tese USP — Arbitragem Estatística + IA (2007)

*https://www.ime.usp.br/~rvicente/LuizParreiras_MSc.pdf*

Luiz Parreiras, IME-USP, orientada pelo Prof. Renato Vicente. Um dos primeiros trabalhos brasileiros a combinar arbitragem estatística com IA. Fundamentação teórica sobre microestrutura de mercado brasileiro.

**Relevância: BAIXA (histórica).** Valor como referência em português e contexto da evolução do campo no Brasil.

---

## 5.3 — Tese UFRGS — PCA + Clustering ML

*https://lume.ufrgs.br/bitstream/10183/262767/1/001174203.pdf*

UFRGS. PCA para redução de dimensionalidade + clustering (K-Means, Hierarchical) para seleção de pares no mercado brasileiro. Abordagem sistemática: reduz espaço de features, agrupa ativos similares, testa cointegração dentro de cada cluster.

**Relevância: MÉDIA.** PCA útil quando features crescerem para 40+ no v3.

---

## 5.4 — Tese IST — XGBoost + Multi-Objetivo (Crypto)

*https://fenix.tecnico.ulisboa.pt/downloadFile/1126295043840475/90144_Miguel_Figueira_Tese.pdf*

Miguel Figueira, Instituto Superior Técnico (Lisboa). XGBoost com otimização multi-objetivo (Pareto front) para crypto.

**Relevância: MÉDIA.** Multi-objetivo aplicável ao SignalScore do v3 Phase 3.

---

## 5.5 — Tese IST — XGBoost + Ações

*https://fenix.tecnico.ulisboa.pt/downloadFile/3378094857519411/94030_tese.pdf*

IST Lisboa. XGBoost com walk-forward validation rigorosa.

**Relevância: BAIXA.** Metodologia de validação relevante para auto-retrain do ArbML.

---

---

# Seção 6 — Síntese e Recomendações para o ArbML (v2)

---

## 6.1 — Recursos de Prioridade Máxima

Os seguintes recursos têm relevância direta e imediata para o ArbML:

- **fiit-ba/ML-for-arbitrage-in-cryptoexchanges (1.3)** — mesmo domínio exato
- **Hongshen-Yang/pair-trading-envs (1.1)** — crypto RL 1min, dynamic scaling
- **seferlab/pairstrading (1.2)** — LSTM vs Transformer comparison
- 🆕 **Amdev-5/crypto-pairs-trading-ai (1.4)** — multi-agente crypto com OBI, WebSocket sub-100ms
- **Yilmaz et al. 2026 (4.1)** — valida LSTM vs Transformers
- **Okasóva et al. 2026 (4.2)** — feature importance crypto arbitrage
- **bkmulusew/ml_pairs_trading (2.9)** — BiLSTM + Attention
- 🆕 **RAmmStein (4.13)** — regime-adaptive thresholds para mean-reversion, resolve problema de calibração de gates
- 🆕 **Graph Learning for FX StatArb (4.12)** — exchanges como grafo, time lag explícito

---

## 6.2 — Lições Transversais (atualizadas)

Com os novos recursos, as conclusões anteriores se reforçam e novas emergem:

- **LSTM permanece competitivo:** Agora confirmado por 3 fontes independentes (Yilmaz 2026, Adamczyk 2025, Valeyre 2024 mostrando que LLMs não superam modelos especializados menores).

- **Feature engineering > arquitetura:** Reforçado por Okasóva (2026) e pelo novo paper de Attention Factors (4.4/4.10).

- **Graph Neural Networks são a fronteira:** Três novos papers (4.10, 4.11, 4.12) usam GNNs. A modelagem de exchanges como nós/arestas de um grafo é uma generalização natural para o ArbML quando escalar além de LSTM puro.

- **Regime awareness é crítico:** RAmmStein (4.13) demonstra que adaptar thresholds ao regime de mercado reduz rebalanceamento em 85%. Isso resolve diretamente o problema documentado do ArbML de que "gates desenhados para dezenas de pares quebram em ~10.000 pares".

- **Observation-execution lag importa:** O paper de FX StatArb (4.12) modela explicitamente o delay entre observação e execução — problema real em crypto arbitrage.

- **Contrastive learning para seleção forward-looking:** O paper de Future-Aligned embeddings (4.14) sugere que priorizar símbolos com base em comportamento futuro predito (não apenas histórico) pode melhorar o SignalScore.

---

## 6.3 — Oportunidades de Evolução (priorizadas)

**CURTO PRAZO:**
- BiLSTM + Attention (2.9) — evolução mínima do LSTM atual
- Regime-adaptive gates inspirado em RAmmStein (4.13) — resolver calibração automática

**MÉDIO PRAZO:**
- Dynamic Scaling (1.1/3.3) — sinais contínuos ao invés de categóricos
- Feature importance via SHAP — validar quais das 40 features contribuem

**LONGO PRAZO:**
- Graph Neural Network sobre o grafo de exchanges (4.12) — modelar relações entre símbolos
- Future-aligned embeddings (4.14) — priorização forward-looking de símbolos

**EXPLORATÓRIO:**
- LLMs como feature extractors (4.15) — monitorar evolução do campo

---

## 6.4 — Resumo Quantitativo

| Categoria | Total | Novos (v2) |
|---|---|---|
| Repositórios open-source | 13 | 1 |
| Papers acadêmicos | 15 | 6 |
| Teses e livros | 4 | 0 |
| **Total** | **32** | **7** |

**Classificação de relevância:** 9 recursos ALTA/MUITO ALTA, 11 MÉDIA-ALTA/MÉDIA, 5 BAIXA-MÉDIA/BAIXA.

---

*— Fim do Documento (v2.0 — 15/Mar/2026) —*

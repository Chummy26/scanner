# ArbML — Referencias Completas com Links

**Todas as fontes citadas no ArbML_State_of_Art_Analysis_v3.md**

---

## Papers com resultados quantitativos (citados diretamente)

| # | Referencia | Link |
|---|-----------|------|
| 1 | Deep Learning for Financial Time Series: A Large-Scale Benchmark (2026). VLSTM Sharpe 2.40 | https://arxiv.org/abs/2603.01820 |
| 2 | Valeyre & Aboura — LLMs for Time Series: Statistical Arbitrage (2024). Chronos negativo apos custos | https://arxiv.org/abs/2412.09394 |
| 3 | Re(Visiting) Time Series Foundation Models in Finance (2025). TimesFM -1.47% | https://arxiv.org/abs/2511.18578 |
| 4 | Yang & Malik — Reinforcement Learning Pair Trading: A Dynamic Scaling Approach (2024). A2C +31.53% | https://arxiv.org/abs/2407.16103 |
| 5 | Ning & Lee — Advanced Statistical Arbitrage with Reinforcement Learning (2024). EMRT | https://arxiv.org/abs/2403.12180 |
| 6 | Epstein et al. — Attention Factors for Statistical Arbitrage (2025). Sharpe >4 net 2.3 | https://arxiv.org/abs/2510.11616 |
| 7 | RAmmStein: Regime Adaptation in Mean-reverting Markets with Stein Thresholds (2026). 85% reducao rebalancing | https://arxiv.org/abs/2602.19419 |
| 8 | Hong & Klabjan — Graph Learning for Foreign Exchange Rate Prediction and Statistical Arbitrage (2025). +61.89% info ratio | https://arxiv.org/abs/2508.14784 |
| 9 | Fanshawe et al. — Forecasting Equity Correlations with Hybrid Transformer GNN (2026). Sharpe 1.837 | https://arxiv.org/abs/2601.04602 |
| 10 | FASCL — Cross-Sectional Asset Retrieval via Future-Aligned Soft Contrastive Learning (2026). Sharpe 5.33 | https://arxiv.org/abs/2602.10711 |
| 11 | Guijarro-Ordonez, Pelger, Zanotti — Deep Learning Statistical Arbitrage (2021/2022) | https://arxiv.org/abs/2106.04028 |
| 12 | Parameters Optimization of Pair Trading Algorithm (2024). Theta_in=1.42, theta_out=0.37 | https://arxiv.org/abs/2412.12555 |
| 13 | Tsoku et al. — Deep Learning-Based Pairs Trading: Real-Time Forecasting of Co-Integrated Crypto Pairs (Frontiers 2026). DWE | https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1749337/full |
| 14 | Okasova et al. — Predicting Arbitrage Occurrences With Machine Learning in Crypto Environments (Wiley 2026) | https://onlinelibrary.wiley.com/doi/full/10.1002/nem.70030 |
| 15 | KCS-LSTM — Optimized LSTM for Arbitrage Spread Forecasting (PMC 2024). Optimal window 11 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11639137/ |
| 16 | Charkhestani & Esfahanipour — Behaviorally informed deep reinforcement learning with loss aversion and overconfidence (Nature Scientific Reports 2026). BBAPT | https://www.nature.com/articles/s41598-026-35902-x |
| 17 | Lwele et al. — Risk-Aware Deep Reinforcement Learning for Dynamic Portfolio Optimization (2025). PPO collapse | https://arxiv.org/abs/2511.11481 |
| 18 | SAC vs DDPG — Cryptocurrency Portfolio Management with RL (2025). SAC Sharpe 5x DDPG | https://arxiv.org/abs/2511.20678 |
| 19 | Hoque et al. — Reinforcement Learning in Financial Decision Making: A Review and Meta-Analysis (2025). 167 papers | https://arxiv.org/abs/2512.10913 |
| 20 | Kim et al. — Are Self-Attentions Effective for Time Series Forecasting? CATS (NeurIPS 2024) | https://arxiv.org/abs/2405.16877 |
| 21 | P-sLSTM — Unlocking the Power of LSTM for Long Term Time Series Forecasting (2024). Geometric ergodicity | https://arxiv.org/abs/2408.10006 |
| 22 | Beck et al. — xLSTM: Extended Long Short-Term Memory (NeurIPS 2024) | https://arxiv.org/abs/2405.04517 |
| 23 | Qin et al. — Censored Quantile Regression Neural Networks (NeurIPS 2022). Portnoy EM | https://arxiv.org/abs/2205.13496 |
| 24 | Wu & Wu — DeepETA: Spatial-Temporal Sequential NN for ETA (AAAI 2019). Asymmetric Huber Loss | https://ojs.aaai.org/index.php/AAAI/article/view/3856 |
| 25 | Martinsson — WTTE-RNN: Weibull Time to Event Recurrent Neural Network (2016). Thesis | https://github.com/ragulpr/wtte-rnn |
| 26 | Koenker & Bassett — Regression Quantiles (Econometrica 1978) | https://doi.org/10.2307/1913643 |
| 27 | TFC — Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency (2022) | https://arxiv.org/abs/2206.08496 |

---

## Papers de verificacao (usados para confirmar/negar claims)

| # | Referencia | Link |
|---|-----------|------|
| 28 | Sigmoid Gating is More Sample Efficient than Softmax Gating in Mixture of Experts (2024). VSN collapse risk | https://arxiv.org/abs/2405.13997 |
| 29 | Wang — Better Inputs Matter More Than Stacking Another Hidden Layer (2025). LOB crypto | https://arxiv.org/abs/2506.05764 |
| 30 | ICS-LSTM — Improving LSTM for Arbitrage Spread Forecasting (PMC 2025). BiLSTM worse than optimized LSTM | https://pmc.ncbi.nlm.nih.gov/articles/PMC11784865/ |
| 31 | Kaplan-Meier as IPCW Weighted Average (PMC). Fixed-horizon censoring G(t)=1 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5568678/ |
| 32 | CQRNN GitHub — TeaPearce implementation. Portnoy EM weights confirmed | https://github.com/TeaPearce/Censored_Quantile_Regression_NN |
| 33 | xLSTMTime — Long-Term Time Series Forecasting with xLSTM (MDPI 2024) | https://arxiv.org/abs/2407.10240 |
| 34 | xLSTM-Mixer — Multivariate Time Series Forecasting by Mixing via Scalar Memories (2024) | https://arxiv.org/abs/2410.16928 |
| 35 | Sharpe Ratio Optimization in MDPs — Non-Markovian dependency breaks Bellman (2025) | https://arxiv.org/abs/2509.00793 |
| 36 | Incorporating Cognitive Biases into RL for Financial Decision-Making (2026). Lambda>=2.5 degrades | https://arxiv.org/abs/2601.08247 |
| 37 | Policy Gradients for Cumulative Prospect Theory in RL (2024) | https://arxiv.org/abs/2410.02605 |
| 38 | Weisfeiler and Leman go Machine Learning (JMLR). GNN expressiveness bounds | http://jmlr.org/papers/volume24/22-0240/22-0240.pdf |

---

## Papers complementares (citados na pesquisa mas nao no documento v3 diretamente)

| # | Referencia | Link |
|---|-----------|------|
| 39 | Hong & Klabjan — Statistical Arbitrage in Options Markets by Graph Learning (2025) | https://arxiv.org/abs/2508.14762 |
| 40 | Signature Decomposition Method for Pair Trading (2025) | https://arxiv.org/abs/2505.05332 |
| 41 | Statistical Arbitrage in Polish Equities Using Deep Learning (2025) | https://arxiv.org/abs/2512.02037 |
| 42 | Hybrid Ridgelet Deep Neural Networks for Arbitrage (2025) | https://arxiv.org/abs/2510.10599 |
| 43 | Deep RL for Optimal Trading with Partial Information (2025). prob-DDPG | https://arxiv.org/abs/2511.00190 |
| 44 | RAmmStein — Regime Adaptation (full HTML) | https://arxiv.org/html/2602.19419 |
| 45 | Sun et al. — A Survey of Statistical Arbitrage Pair Trading with ML, DL, and RL (2025) | https://ideas.repec.org/p/war/wpaper/2025-22.html |
| 46 | Statistical Arbitrage in Cryptocurrency Markets (Deutsche Nationalbibliothek) | https://d-nb.info/1207154512/34 |
| 47 | Tan & Plataniotis — A hybrid CNN-LSTM for statistical arbitrage (Quantitative Finance 2023) | https://www.tandfonline.com/doi/abs/10.1080/14697688.2023.2181707 |
| 48 | Machine Learning Enhanced Pairs Trading — BiLSTM + Attention + TCN + MADDPG (MDPI 2024) | https://www.mdpi.com/2571-9394/6/2/24 |
| 49 | Moody & Saffell — Learning to Trade via Direct Reinforcement (NeurIPS). DSR original | https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading |
| 50 | FinRL Contests — Benchmarking Data-Driven Financial RL Agents (2025) | https://arxiv.org/abs/2504.02281 |
| 51 | Bai et al. — Evolution of RL in Quantitative Finance (ACM Computing Surveys 2025) | https://arxiv.org/abs/2408.10932 |
| 52 | Contrastive Learning of Asset Embeddings from Financial Time Series (ICAIF 2024) | https://arxiv.org/abs/2407.18645 |
| 53 | Zeng et al. — Are Transformers Effective for Time Series Forecasting? (AAAI 2023) | https://arxiv.org/abs/2205.13504 |

---

## Repositorios Open-Source analisados

| # | Repositorio | Link |
|---|------------|------|
| 54 | Hongshen-Yang/pair-trading-envs — RL crypto pairs, Backtrader + Stable-Baselines3 | https://github.com/Hongshen-Yang/pair-trading-envs |
| 55 | seferlab/pairstrading — LSTM vs Transformers comparison | https://github.com/seferlab/pairstrading |
| 56 | fiit-ba/ML-for-arbitrage-in-cryptoexchanges — ML classifiers cross-exchange | https://github.com/fiit-ba/ML-for-arbitrage-in-cryptoexchanges |
| 57 | Amdev-5/crypto-pairs-trading-ai — Multi-agent, OBI, WebSocket sub-100ms | https://github.com/Amdev-5/crypto-pairs-trading-ai |
| 58 | bkmulusew/ml_pairs_trading — BiLSTM + Attention + TCN + MADDPG | https://github.com/bkmulusew/ml_pairs_trading |
| 59 | raktim-roychoudhury/pairs_trading — Clustering + DRL | https://github.com/raktim-roychoudhury/pairs_trading |
| 60 | prakhar1602/pair_trading_with_machine_learning — LSTM +30% vs tradicional | https://github.com/prakhar1602/pair_trading_with_machine_learning |
| 61 | achntj/statistical-arbitrage — K-Means + Cointegration + VaR/CVaR | https://github.com/achntj/statistical-arbitrage |
| 62 | ArbitrageLab (Hudson & Thames) — OPTICS/DBSCAN + Cointegration + Half-life | https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/ml_approach/ml_based_pairs_selection.html |
| 63 | NX-AI/xlstm — Official xLSTM implementation | https://github.com/NX-AI/xlstm |
| 64 | TeaPearce/Censored_Quantile_Regression_NN — NeurIPS 2022 CQRNN | https://github.com/TeaPearce/Censored_Quantile_Regression_NN |
| 65 | dongbeank/CATS — NeurIPS 2024 CATS implementation | https://github.com/dongbeank/CATS |
| 66 | AchillesJJ/DSR — Differential Sharpe Ratio reference | https://github.com/AchillesJJ/DSR |
| 67 | ragulpr/wtte-rnn — Weibull Time to Event RNN | https://github.com/ragulpr/wtte-rnn |

---

## Documentacao tecnica e blogs

| # | Referencia | Link |
|---|-----------|------|
| 68 | Uber DeepETA Blog — How Uber Predicts Arrival Times (2022) | https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/ |
| 69 | DoorDash — Precision in Motion: Deep Learning for Smarter ETA Predictions (2024) | https://doordash.engineering/2024/03/12/precision-in-motion-deep-learning-for-smarter-eta-predictions/ |
| 70 | Hudson & Thames — Caveats in Calibrating the OU Process. Theta bias warning | https://hudsonthames.org/caveats-in-calibrating-the-ou-process/ |
| 71 | Keras — Classification with GRN and VSN (GRN/VSN architecture reference) | https://keras.io/examples/structured_data/classification_with_grn_and_vsn/ |
| 72 | TFT Paper — Temporal Fusion Transformers (Lim et al., ScienceDirect) | https://www.sciencedirect.com/science/article/pii/S0169207021000637 |
| 73 | Hasbrouck — Price Discovery (1995, foundational for volume ratio feature) | https://doi.org/10.1111/j.1540-6261.1995.tb04054.x |
| 74 | Zhivkov — Two-Tiered Structure of Cryptocurrency Funding Rate Markets (MDPI Mathematics 2025) | https://www.mdpi.com/2227-7390/14/2/346 |
| 75 | Price Discovery in Cryptocurrency Markets (2025) | https://arxiv.org/abs/2506.08718 |
| 76 | Designing Funding Rates for Perpetual Futures (2025) | https://arxiv.org/abs/2506.08573 |
| 77 | Atlantis Press — Arbitrage Detection in Crypto Markets Using GNN (ICISD-2025). F1=0.90 | https://www.atlantis-press.com/proceedings/icisd-25/126016976 |
| 78 | MAPS — Multi-Agent Reinforcement Learning-based Portfolio Management (2020) | https://arxiv.org/abs/2007.05402 |
| 79 | Calibrating OU Process — Dean Markwick (2024) | https://dm13450.github.io/2024/03/09/Calibrating-an-Ornstein-Uhlenbeck-Process.html |
| 80 | Bayesian Online Changepoint Detection — Adams & MacKay (2007) | https://arxiv.org/abs/0710.3742 |
| 81 | FOCuS — Fast Online Changepoint Detection via Functional Pruning CUSUM (JMLR 2023) | https://www.jmlr.org/papers/v24/21-1230.html |
| 82 | How Flawed Is ECE? — ICML 2024 | https://proceedings.mlr.press/v235/chidambaram24a.html |
| 83 | Evidently AI — Which Drift Detection Method Is Best? (2024) | https://www.evidentlyai.com/blog/data-drift-detection-large-datasets |

---

**Total: 83 referencias (27 no documento v3 + 11 de verificacao + 15 complementares + 14 repositorios + 16 docs/blogs)**

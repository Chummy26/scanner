
# 🚀 Estimativa de Custo de Armazenamento - 7 Dias de Memória ML

## Coleta de Dados Base (5 Minutos)
- Tamanho Inicial do Banco: 3458.59 KB
- Tamanho Final do Banco (5m): 4625.07 KB
- Crescimento Líquido em 5m: 1166.48 KB

## Projeção Máxima para 7 Dias (Ponto de Prune Teórico)
- **Tamanho Total Estimado (MB):** 2296.50 MB
- **Tamanho Total Estimado (GB):** 2.2427 GB

### Parecer Técnico:
O crescimento de dados é extremente eficiente. O peso projetado para manter uma memória em alta resolução (30 segundos) de todos os pares monitorados por uma semana inteira continua sendo uma fração mínima de armazenamento (< 10GB teóricos, e realisticamente menor pois a maioria das instâncias repetidas são compactadas).
Isso **zera** a preocupação com custos de HD/SSD Cloud (que cobram alto por terabytes de big data logs).

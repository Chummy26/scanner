# Objetivo do ArbML

## Introdução

O ArbML nasce para resolver um problema muito específico:

o trabalho de analisar o histórico das moedas é lento demais quando feito manualmente.

Ver que existe uma diferença de preço entre dois mercados não é a parte difícil. A parte difícil é entender se aquela diferença atual realmente faz sentido quando comparada ao que a mesma moeda acabou de mostrar no seu histórico recente.

Este documento explica, de forma mais detalhada, qual é o objetivo do ArbML e como a machine learning entra nessa leitura.

---

## 1. O problema real não está em enxergar a diferença

Hoje, um operador pode até conseguir ver que existe uma diferença de preço entre dois mercados.

Mas isso, por si só, ainda diz pouco.

O problema real está em responder perguntas como:

- essa diferença atual está fraca, normal ou forte?
- essa moeda costuma alcançar níveis maiores do que o atual?
- quando essa moeda entra em uma faixa mais forte, para onde essa diferença costuma caminhar depois?
- esse comportamento apareceu muitas vezes no histórico recente ou quase não apareceu?

Essas respostas não aparecem em um único número na tela.

Para responder bem, o operador precisa abrir históricos, comparar momentos, lembrar padrões e repetir esse mesmo processo para várias moedas ao mesmo tempo.

É exatamente aí que o processo manual deixa de escalar.

---

## 2. O que o ArbML pretende fazer

O objetivo do ArbML não é apertar botões no lugar do operador.

O objetivo do ArbML é automatizar a parte mais pesada da leitura.

Em termos claros, o sistema existe para:

- observar o histórico recente de muitas moedas ao mesmo tempo
- medir como o spread de cada moeda vem se comportando
- comparar o spread atual com o padrão recente daquela moeda
- destacar quando o momento atual parece mais forte, mais fraco ou apenas comum

Em outras palavras, o ArbML não tenta transformar um número isolado em uma resposta pronta.

Ele tenta transformar histórico recente em contexto analítico.

---

## 3. O que significa machine learning aqui

Neste projeto, **machine learning** não significa magia e não significa previsão perfeita.

Machine learning, aqui, significa usar métodos estatísticos e computacionais para aprender regularidades observadas nos dados recentes e usar essas regularidades para interpretar melhor o presente.

Dito de forma mais técnica: o sistema não observa apenas o valor atual do spread. Ele tenta entender a posição relativa do spread atual dentro do comportamento recente da moeda.

Isso é importante porque o mesmo número pode significar coisas diferentes em moedas diferentes, ou até na mesma moeda em momentos diferentes.

Um spread atual de `2%`, por exemplo, pode ser:

- alto para uma moeda que normalmente trabalha perto de `0,5%`
- irrelevante para uma moeda que repetidamente chega a `6%`

Portanto, a machine learning entra para responder uma pergunta mais inteligente:

**o valor atual, isoladamente, parece interessante ou não quando comparado ao que essa moeda acabou de mostrar no seu próprio histórico?**

Na prática, isso significa olhar fatores como:

- quais faixas aparecem repetidamente no histórico recente
- quão distante o spread atual está dessas faixas
- com que frequência certas entradas são seguidas por certas saídas
- se o comportamento recente parece organizado ou disperso

O sistema não precisa prometer o futuro com certeza absoluta para ser útil.

Ele já se torna valioso quando consegue interpretar o presente melhor do que uma leitura manual apressada.

---

## 4. Como a ML faz essa leitura

A leitura que o ArbML pretende fazer é simples no conceito, mas poderosa na prática.

### 4.1. Primeiro, ele observa uma janela recente

O sistema olha para um período recente da moeda, como as últimas 2 horas, 24 horas ou mais.

Essa janela recente é importante porque ela mostra o comportamento que a moeda está exibindo agora, e não apenas o que ela fez em um passado distante que talvez já não represente o momento atual.

### 4.2. Depois, ele procura repetição

O sistema tenta identificar se certas faixas aparecem de forma recorrente.

Se uma moeda mostra várias vezes entradas perto de `6%`, isso é muito mais relevante do que um único toque isolado em `6%`.

O mesmo vale para a saída.

Se várias saídas aparecem perto de `-3%`, isso sugere que essa faixa faz parte do comportamento recente da moeda.

### 4.3. Em seguida, ele compara o presente com esse padrão

Depois de observar o histórico recente, o sistema olha para o spread atual e pergunta:

- o valor atual está perto das faixas mais recorrentes?
- ele está abaixo delas?
- ele está acima delas?

Essa comparação é o centro da análise.

A pergunta principal deixa de ser "quanto está agora?" e passa a ser "quanto está agora em relação ao que essa moeda vem mostrando?".

### 4.4. Por fim, ele tenta medir a força do contexto atual

Quando a ML compara o presente com o histórico recente, ela pode ajudar a classificar o momento atual de forma mais útil:

- parece um contexto fraco
- parece um contexto intermediário
- parece um contexto forte

Uma ideia simples e realista que cabe muito bem no projeto é o sistema mostrar exatamente isso: se o spread atual está abaixo, dentro ou próximo das faixas que mais se repetiram recentemente.

Outra ideia igualmente direta é comparar janelas curtas e mais longas, por exemplo:

- o spread atual parece fraco nas últimas 2 horas
- mas parece relativamente forte nas últimas 24 horas

Isso não foge da ideia principal. Pelo contrário: aprofunda a mesma leitura com mais contexto.

---

## 5. O exemplo que resume a ideia inteira

Suponha novamente a seguinte leitura:

- nas últimas 2 horas, uma moeda mostrou várias vezes spread de entrada perto de `6%`
- nessas mesmas últimas 2 horas, ela também mostrou várias vezes spread de saída perto de `-3%`
- agora, neste instante, o spread de entrada está em `2%`

Um operador experiente não olha para esse `2%` de forma isolada.

Ele compara o presente com o comportamento recente da moeda.

Ao fazer isso, ele percebe algo importante:

- a moeda vem mostrando, repetidamente, que os momentos mais fortes de entrada aparecem perto de `6%`
- a moeda também vem mostrando, repetidamente, que depois desses momentos a diferença costuma caminhar até uma faixa perto de `-3%`
- portanto, um spread atual de `2%` parece mais fraco do que a faixa de entrada que o histórico recente vinha sugerindo

Esse exemplo mostra exatamente como a ML deve pensar.

Ela não precisa dizer que `2%` é "ruim" em termos absolutos.

Ela precisa dizer algo mais útil:

**diante do que essa moeda vem mostrando recentemente, `2%` parece menos alinhado com os momentos que mais chamaram atenção no histórico.**

Se mais tarde essa mesma moeda voltar a `5,8%` ou `6,1%`, e o padrão recente continuar parecido, então o sistema poderá concluir que o momento atual está muito mais próximo da faixa que vem se repetindo como entrada forte.

Perceba que a lógica não mudou.

O que mudou foi apenas a posição do momento atual dentro do contexto recente.

---

## 6. O que o sistema deve entregar para o operador

O ArbML deve ajudar o operador a chegar mais rápido a uma leitura de qualidade.

Sem entrar em detalhes de implementação, o núcleo do que ele deve entregar é este:

- mostrar o spread atual já acompanhado de contexto
- mostrar quais faixas do histórico recente aparecem com mais recorrência
- mostrar se o momento atual parece distante ou próximo dessas faixas mais relevantes

Em termos práticos, isso pode aparecer de formas muito objetivas, por exemplo:

- o spread atual está abaixo da faixa que mais apareceu como entrada nas últimas 2 horas
- a moeda tem histórico recente de saídas concentradas perto de `-3%`
- o momento atual parece mais fraco do que os momentos que mais se repetiram

Uma sugestão realista e útil é o sistema conseguir explicar sua leitura em linguagem curta e direta, algo como:

"Nas últimas 2 horas, esta moeda repetiu entradas perto de 6%. O valor atual de 2% está abaixo da faixa mais recorrente e, por isso, parece menos forte do que os melhores momentos recentes."

Isso não adiciona complexidade desnecessária. Apenas torna a análise mais clara.

---

## 7. O que o ArbML não promete

O projeto não deve ser entendido como uma máquina que garante lucro.

Também não deve ser entendido como um sistema que elimina toda decisão humana.

O que ele faz é mais objetivo:

- organizar históricos
- detectar repetições relevantes
- comparar presente e passado recente
- priorizar o que parece mais consistente

O ponto central é este:

recorrência não é garantia, mas é uma informação muito melhor do que olhar apenas o número isolado do instante.

---

## 8. Por que isso tem valor

Sem um sistema desse tipo, o operador depende de tempo, memória e velocidade para fazer a leitura manual de vários cenários ao mesmo tempo.

Com um sistema desse tipo, passa a existir a possibilidade de:

- analisar muitas moedas simultaneamente
- manter o mesmo critério de leitura em todas elas
- reduzir a dependência de leitura subjetiva apressada
- destacar primeiro os casos em que o spread atual parece mais alinhado com o padrão recente

O valor do ArbML, portanto, não está em inventar uma nova lógica de mercado.

O valor está em automatizar, com método, a comparação entre o spread atual e o histórico recente da própria moeda.

---

## 9. Resumo final

O objetivo do ArbML é simples:

pegar a análise que hoje depende de abrir históricos e comparar padrões manualmente, e transformar isso em um processo contínuo, rápido e consistente.

Se uma moeda mostrou repetidamente entradas perto de `6%` e saídas perto de `-3%`, enquanto agora está em `2%`, a função do sistema é perceber que o momento atual parece mais fraco do que o padrão recente mais interessante.

Essa é a base do projeto.

Não se trata de adivinhar o futuro com perfeição.

Trata-se de ler melhor o presente com base no que a moeda acabou de mostrar no seu histórico.

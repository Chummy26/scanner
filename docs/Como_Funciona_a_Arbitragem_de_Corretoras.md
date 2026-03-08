# Como Funciona a Arbitragem de Corretoras

## Introdução

Este documento explica, do zero, uma ideia simples:

a mesma moeda pode ter preços diferentes em corretoras diferentes, e essa diferença pode criar uma oportunidade.

Mas a oportunidade real não está apenas em ver uma diferença grande em um único instante. O ponto principal está em entender como essa diferença vem se comportando no histórico recente.

---

## 1. O que é uma corretora

Uma corretora é uma plataforma onde pessoas e empresas compram e vendem ativos, como moedas digitais.

Dentro de uma corretora, o preço não é inventado pela plataforma. Ele surge do encontro entre quem quer comprar e quem quer vender.

Em termos simples:

- quem quer comprar aceita pagar até certo preço
- quem quer vender aceita receber a partir de certo preço
- quando essas intenções se encontram, os negócios acontecem

Por isso, o preço de uma moeda não é um número fixo no mundo inteiro. Ele depende do que está acontecendo em cada corretora naquele momento.

---

## 2. Como o preço é formado na prática

Em qualquer mercado, existe um livro de ofertas. Esse livro mostra:

- os preços em que há gente querendo comprar
- os preços em que há gente querendo vender
- a quantidade disponível em cada nível de preço

Dois pontos são especialmente importantes:

- o melhor preço de compra disponível naquele instante
- o melhor preço de venda disponível naquele instante

Essa diferença entre compra e venda já mostra que executar uma operação real nunca depende de um número teórico. Depende dos preços que realmente estão disponíveis para executar.

Por isso, quando falamos de arbitragem, a comparação correta não deve usar um preço imaginário. Ela deve usar os preços que de fato permitiriam abrir e fechar a operação.

---

## 3. Por que a mesma moeda pode ter preços diferentes

A mesma moeda pode aparecer com preços diferentes em corretoras diferentes por vários motivos:

- cada corretora tem participantes diferentes
- a quantidade de ordens disponíveis não é igual
- a liquidez muda de uma corretora para outra
- a velocidade com que o mercado se ajusta não é idêntica em todos os lugares

Isso significa que, por alguns instantes ou por períodos maiores, uma corretora pode mostrar a moeda relativamente mais cara e outra relativamente mais barata.

É exatamente dessa distância entre dois mercados que nasce a arbitragem.

---

## 4. O que é a arbitragem de corretoras

Arbitragem de corretoras é uma operação montada em dois lados ao mesmo tempo, usando a mesma moeda em dois mercados relacionados.

Em termos tecnicamente corretos, a estrutura é pensada para que o resultado dependa principalmente da diferença entre os dois mercados, e não apenas da direção isolada da moeda.

Em uma explicação simples:

- um lado da operação fica exposto de forma comprada
- o outro lado fica exposto de forma vendida, ou em posição economicamente equivalente
- depois, o operador acompanha a mudança da distância entre esses dois lados

O lucro não vem de acertar se a moeda vai subir ou cair sozinha.

O lucro vem da mudança da diferença entre os dois mercados.

Essa é a parte mais importante do conceito.

---

## 5. O que este projeto chama de spread

Neste contexto, a palavra **spread** significa a diferença percentual entre os dois lados da operação.

De forma didática:

- **spread de entrada**: a diferença percentual observada no momento de abrir a operação
- **spread de saída**: a diferença percentual observada no momento de encerrar a operação

Se o spread de entrada está em `6%`, isso significa que a distância entre os dois lados da operação está, naquele instante, em um nível de 6%.

Se depois o spread de saída aparece em `-3%`, isso significa que a mesma distância mudou de lugar e passou para o outro lado da referência usada no cálculo.

O ponto central não está apenas no sinal positivo ou negativo. O ponto central está no percurso dessa diferença ao longo do tempo.

Em uma leitura prática:

- entrar em `6%`
- sair em `-3%`

significa que houve um deslocamento grande da diferença entre os dois lados da operação.

Em termos de distância percentual observada no spread, esse exemplo representa um deslocamento bruto de `9` pontos percentuais entre a entrada e a saída.

É essa mudança que interessa.

---

## 6. Onde a análise realmente acontece

Muita gente imagina que a oportunidade aparece quando alguém olha para a tela e encontra um spread alto.

Isso é incompleto.

O raciocínio correto é outro:

1. observar o spread atual
2. abrir o histórico recente da mesma moeda
3. entender quais níveis de entrada apareceram várias vezes
4. entender quais níveis de saída apareceram várias vezes
5. comparar o momento atual com esse comportamento recente

Em outras palavras, a pergunta certa não é:

"o spread está alto agora?"

A pergunta certa é:

"o spread atual está forte ou fraco quando comparado ao que essa moeda vem mostrando recentemente?"

---

## 7. Exemplo simples e central

Suponha a seguinte situação:

- nas últimas 2 horas, uma moeda mostrou várias vezes spread de entrada perto de `6%`
- nessas mesmas últimas 2 horas, essa moeda também mostrou várias vezes spread de saída perto de `-3%`

O que isso ensina?

Ensina que, no comportamento recente dessa moeda, houve repetição de um mesmo padrão:

- a diferença entre os dois lados frequentemente chegou a uma faixa de entrada perto de `6%`
- depois, em vários momentos, essa diferença caminhou até uma faixa de saída perto de `-3%`

Agora imagine que, neste instante, o spread de entrada está em `2%`.

Esse número de `2%` pode parecer interessante para quem olha apenas o momento atual. Mas, quando ele é comparado com o histórico recente, a leitura muda:

- se a moeda vinha mostrando entradas fortes perto de `6%`
- e agora está apenas em `2%`

então o spread atual pode estar mais fraco do que o padrão recente mais interessante.

Ou seja: não basta existir diferença. É preciso ver se a diferença atual está realmente em uma faixa que, historicamente e no curto prazo, tem se mostrado relevante.

Por outro lado, se essa mesma moeda voltar a mostrar um spread de entrada perto de `6%`, e o histórico recente continuar mostrando várias saídas perto de `-3%`, então passa a existir um motivo mais forte para olhar a oportunidade com seriedade.

Não porque o passado garante o futuro.

Mas porque o passado recente mostra que esse comportamento já apareceu várias vezes em um período próximo do momento atual.

---

## 8. O valor do histórico recente

O histórico recente é valioso porque ajuda a responder perguntas objetivas:

- essa moeda costuma abrir diferenças grandes ou pequenas?
- essas diferenças aparecem de forma repetida ou isolada?
- quando a entrada fica interessante, a saída costuma caminhar até que faixa?
- o momento atual está acima, abaixo ou no meio do comportamento recente?

Esse tipo de leitura é mais forte do que uma decisão baseada em um único print da tela.

Uma oportunidade bem lida não nasce de um número solto. Ela nasce da comparação entre o número atual e o contexto recente.

---

## 9. O que precisa ser checado na prática

Mesmo quando o histórico recente parece favorável, ainda existem cuidados básicos:

- a operação precisa ser executável nos preços reais disponíveis
- precisa existir liquidez suficiente para o tamanho desejado
- os custos precisam caber dentro do movimento esperado
- o tempo necessário para a operação continuar fazendo sentido precisa ser aceitável

Esses pontos não anulam a oportunidade. Eles apenas separam uma leitura bonita na tela de uma operação que realmente pode ser montada.

---

## 10. Resumo final

Arbitragem de corretoras não é apenas encontrar uma moeda com diferença de preço.

O trabalho correto é observar como essa diferença vem se comportando no histórico recente e comparar esse padrão com o momento atual.

Se uma moeda mostrou várias vezes entradas perto de `6%` e várias saídas perto de `-3%`, essa informação vale mais do que olhar apenas um spread atual de `2%` isoladamente.

O centro da análise está exatamente aí:

entender quando a diferença atual está realmente forte diante do comportamento recente da moeda.

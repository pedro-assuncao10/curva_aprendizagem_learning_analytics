# Passo 12 — Curva de aprendizagem em 2 estágios

## Definição do projeto (decisão formal)

Existem duas formas possíveis de definir "curva de aprendizagem do aluno", e
o projeto ficou dando voltas entre elas até isso ser decidido explicitamente:

- **Definição A — curva "por prova"**: hábito de preparação em torno de
  CADA avaliação individual (ex.: estuda com quantos dias de antecedência
  antes de cada prova). Foi a abordagem original do passo 6
  (`antecedencia`/janela W). Testada à exaustão (W=7/14/21/28/adaptativo) e
  descartada — nunca carregou sinal forte (correlação ≤0,18 com aprovação) e
  piorava conforme a janela crescia.
- **Definição B — curva "do curso inteiro"**: como o esforço se distribui ao
  longo de TODO o calendário do curso (dia 0 até o fim). As provas servem só
  de marcos que dividem o tempo, não são o assunto em si.

**Decisão**: o projeto usa a **Definição B**. Os 4 perfis (desistente,
antecipado, equilibrado, fim-pesado) são conceitos de trajetória do curso
inteiro, não de hábito repetido por prova — e é assim que o passo 3 (nossa
referência independente) também mede, por terços do curso. Esse passo 12 (e o
passo 13, que o valida) usam essa definição.

## Objetivo

Separar o aluno em 4 perfis pela **forma da curva de esforço ao longo do curso
inteiro** (não só em torno da prova):

- **desistente** — some de vez, sem nenhuma atividade no último terço do curso
- **antecipado** — esforço concentrado no início, cai depois
- **equilibrado** — esforço parecido do início ao fim
- **fim-pesado** — esforço cresce e se concentra perto do fim

## Por que abandonamos a ideia de "janela antes da prova" (W)

A versão original (passo 6) media o comportamento do aluno só nos **W dias
antes de cada prova** (W=7). Testamos várias formas de melhorar isso antes de
decidir abandonar o conceito de janela:

| Método | corr. com `ativo_no_fim` | corr. com aprovação |
|---|---|---|
| W=7 fixo (antecedência escalar) | 0,112 | 0,181 |
| W=14 fixo | 0,080 | 0,160 |
| W=28 fixo | -0,023 | 0,078 |
| W adaptativo = max(28, gap até a prova anterior) | -0,062 | 0,032 |

**Conclusão do teste**: aumentar a janela sempre piorou a separação, nunca
melhorou. Isso acontece porque, com janela maior, praticamente todo clique do
aluno passa a cair "dentro" de alguma janela pré-prova — o que é capturado
deixa de ser "comportamento de preparação para aquela prova" e passa a ser
apenas "engajamento geral do curso", que é um sinal mais fraco e mais
redundante com métricas de volume que já existem.

A tentativa de usar o menor gap real entre provas como W (a ideia inicial)
também não funcionou: o menor gap *global* é de 2 dias, mas é um artefato de
um curso só (datas de CMA/TMA muito próximas); a maioria dos gaps reais fica
entre 28 e 76 dias, então não existe um W único que sirva bem para todos os
cursos sem cortar dado real de uns ou vazar dado de outros.

## O cálculo que substituiu a janela

Em vez de escolher um W, o Passo 12 **não usa janela nenhuma** — a cada aluno
é atribuído 100% do seu volume de cliques, distribuído nos intervalos reais do
calendário do curso:

1. **Intervalos = datas de prova únicas do curso**, mais um intervalo final
   entre a última prova e o **fim oficial do curso** (`courses.csv`,
   `module_presentation_length`). Sem esse intervalo final, entre 13% e 18%
   dos cliques de vários cursos (ex. CCC, DDD_2014J) ficavam de fora só por
   acontecerem depois da última prova — não é ruído, é atividade real do
   aluno que a versão anterior descartava.
2. **Datas duplicadas viram uma só**: várias provas no mesmo dia (comum em
   avaliações tipo CMA agrupadas com a prova principal) formam um intervalo de
   largura zero, que nunca recebe clique nenhum e distorce a curva — por isso
   usamos `np.unique` nas datas antes de montar os intervalos.
3. **Cada intervalo vira uma TAXA** (cliques ÷ dias do intervalo), não uma
   soma bruta — assim um intervalo de 56 dias não "vence" um de 19 dias só por
   ser mais comprido.
4. A taxa é normalizada em **proporção** (soma 1) — essa é a "forma" da curva
   daquele aluno.
5. A forma é **reamostrada em 6 pontos fixos**, usando a **posição real no
   calendário** do curso (fração de dias percorridos, não o índice da prova).
   Isso é o que permite comparar um curso de 5 provas com um de 14 provas na
   mesma régua — o ponto "40% do curso" significa a mesma coisa
   independentemente de quantas provas existirem.

Verificação: soma de cliques capturados = soma de cliques disponíveis para os
alunos válidos, **100,00% exato** (não é estimativa).

## Estágio 1 — regra, não clustering

Desistente é separado por `ativo_no_fim` (já calculado e validado no passo 3):
teve ou não teve qualquer clique no último terço do curso. Esse sinal é quase
determinístico (1,2% de aprovação entre quem não teve atividade no fim, contra
84,3% entre quem teve) e substitui `durabilidade`, que confundia desistente
com fim-pesado (ambos têm poucas provas com pré-estudo, por motivos opostos).

## Estágio 2 — K-means na curva, só entre quem persiste

Só entre os alunos que passam no Estágio 1, roda-se K-means (K=3) sobre os 6
pontos da curva (padronizados). Cada cluster é nomeado automaticamente pelo
**centro de massa temporal** da curva média (soma ponderada da posição pelos
valores de esforço): menor centro de massa = esforço concentrado cedo =
"antecipado"; maior = "fim-pesado"; o do meio = "equilibrado".

## Resultado

| perfil | n | % da base | aprovação |
|---|---|---|---|
| desistente | 3.135 | 14,7% | 1,2% |
| antecipado | 9.241 | 43,3% | 80,7% |
| equilibrado | 6.674 | 31,3% | 87,9% |
| fim-pesado | 2.268 | 10,6% | 88,5% |

Validação cruzando com os perfis de referência do passo 3 (`timing`/`subtipo`,
que usam uma regra independente, não K-means):

- **desistente**: 100% de acerto (é a mesma regra em ambos, serve de sanity check)
- **fim-pesado**: 84,7% cai no cluster certo
- **antecipado**: 72,9% cai no cluster certo
- **equilibrado**: o mais fraco, se espalha entre antecipado (65,2%) e o
  próprio grupo (27,0%) — é o perfil "resíduo", mais difícil de isolar porque
  fica no meio dos outros dois por definição.

ARI (K-means da curva vs. perfis de referência) = **0,175**. Silhueta média do
estágio 2 = 0,208 (K=3; K=2 teria silhueta um pouco maior, 0,263, mas colapsa
"equilibrado" e "fim-pesado" ou "antecipado" num grupo só — K=3 foi mantido
porque é o que corresponde aos 3 perfis que queremos comparar, mesma lógica
usada nos passos 6 e 8).

## Arquivos gerados

- `01_cotovelo_silhueta.png` — escolha de K para o estágio 2
- `02_perfis_curvas.png` — as 4 curvas médias (o resultado principal)
- `03_resultado_por_cluster.png` — aprovação/reprovação por perfil
- `04_silhueta_facas_e_mapa.png` — diagnóstico de silhueta do estágio 2 + PCA
- `curvas_e_perfis.csv` — 6 pontos de curva + perfil final por aluno
- `perfil_dos_grupos.csv` — tabela-resumo (n, % base, aprovação)
- `validacao_crosstab.csv` — cruzamento com os perfis de referência do passo 3
- `comparacao_metodos_w.csv` — tabela dos métodos de janela testados

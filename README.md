# llm-json

Recupera um valor JSON utilizável do que o modelo **de fato** mandou — e diz o que precisou ser consertado.

```python
from llm_json import loads

loads('Claro! ```json\n{"status": "ok",}\n```')
# {'status': 'ok'}
```

`json.loads` recusa essa resposta. E a resposta já foi paga.

---

## O problema

Você pede JSON para um modelo. Ele responde com JSON — mais uma frase de cortesia na frente, dentro de uma cerca de código, com uma vírgula sobrando, com comentários `//` explicando os campos, em sintaxe Python, ou com os primeiros dois terços de um objeto grande porque bateu o limite de tokens.

Todos esses casos quebram `json.loads`. Nenhum deles é culpa do seu código, e todos aparecem em produção, não em teste.

## O que diferencia esta biblioteca

**Nada é consertado em silêncio.** Esse é o ponto inteiro.

As bibliotecas de "json repair" devolvem um valor e pronto. Mas uma resposta que só precisou perder a cerca de código e uma resposta que só parseou porque *fecharam as chaves que o modelo não mandou* são coisas radicalmente diferentes: a primeira é a resposta do modelo, a segunda é dois terços dela.

```python
from llm_json import extract

found = extract('{"itens": [1, 2, 3], "total":')
found.value  # {'itens': [1, 2, 3]}
found.truncated  # True
found.repairs  # ('closed-truncation',)
```

O campo `total` não está ali porque o modelo nunca o enviou. Uma biblioteca que devolve esse dict sem avisar entrega um registro com um campo faltando silenciosamente — e `total` costuma ser justamente o campo que vai para o banco.

Quem não quer decidir caso a caso recusa de uma vez:

```python
loads(reply, allow_truncated=False)  # levanta TruncatedJSON
loads(reply, expect=dict)  # levanta UnexpectedType se vier uma lista
```

## O que ele conserta

| Reparo | Entrada típica |
| --- | --- |
| `unfenced` | ```` ```json … ``` ```` |
| `removed-prose` | `Aqui está: {"a": 1}` |
| `normalised-quotes` | `{'a': 1}`, `{“a”: 1}` |
| `removed-comments` | `{ // o identificador\n "a": 1 }` |
| `removed-trailing-comma` | `{"a": 1,}` |
| `python-literals` | `{"a": True, "b": None}` |
| `closed-truncation` | `{"a": 1, "b":` |

Os reparos são aplicados em ordem, do mais barato ao mais destrutivo, e a tentativa para no primeiro que der certo. Então `repairs` lista o que foi **necessário**, não o que foi tentado: uma resposta que só estava cercada nunca aparece como tendo tido as aspas reescritas.

---

## Os cinco bugs que moldaram o código

Esta seção é o motivo de o pacote não ser um punhado de `re.sub`. Cada item abaixo é a versão óbvia da solução, e por que ela está errada.

### 1. Remover comentários `//` com regex destrói toda URL do payload

```json
{"url": "https://exemplo.com"}
```

Tem `//` ali. Um removedor por regex transforma isso em `{"url": "https:}` — que *parece* válido, está silenciosamente errado, e é o tipo de bug que se descobre em produção e não em revisão.

A correção não é um regex melhor: é parar de editar JSON como se fosse texto plano. [`classify()`](src/llm_json/scanner.py) percorre o texto uma vez e rotula cada caractere como código, string ou comentário. Uma `//` vista dentro de uma string não é comentário; uma `"` vista dentro de um comentário não abre string. Todo reparo depois consulta o rótulo em vez de adivinhar.

O mesmo erro, duas vezes mais: `,\s*}` corrompe qualquer string que contenha `, }`, e trocar `True` por `true` acerta a palavra dentro de `{"nota": "True story"}`.

### 2. Fechar um truncamento que não salvou nada é fabricar, não reparar

```
Use {placeholder} para variáveis. Resultado: {"a": 1}
```

`{placeholder}` tem chaves balanceadas, falha ao parsear — e aí o fechamento de truncamento o "conserta" para um `{}` perfeitamente válido, que ganha do objeto real mais adiante na mesma resposta. O resultado era `{}`.

Fechar agora exige que **algo completo** tenha sido visto antes do corte. Devolver `{}` para uma resposta que morreu dentro da primeira chave não é um valor recuperado, é um valor inventado — e lá na frente ele se lê como "o modelo não achou nada" em vez de "o modelo foi interrompido".

### 3. Um objeto guarda os campos que chegaram; um array guarda só elementos completos

Esse foi o acerto mais difícil, e só apareceu com uma resposta real:

```json
[{"id": 1}, {"id": 2}, {"id": 3, "stat
```

Fechar tudo que está aberto dá `[{"id": 1}, {"id": 2}, {"id": 3}]` — três registros, sendo que o terceiro tem campos faltando e é **indistinguível** de um registro completo para quem consome. É exatamente assim que uma linha lida pela metade chega ao banco parecendo inteira.

Mas a regra oposta também está errada: num objeto truncado de nível raiz, os três campos que chegaram valem ter.

A assimetria é o contrato:

```
{"a": 1, "b":                      ->  {"a": 1}
[{"id": 1}, {"id": 2}, {"id": 3,   ->  [{"id": 1}, {"id": 2}]
```

O objeto é um envelope e guarda seus membros; o array descarta o elemento inacabado inteiro. Implementado com um quadro por contêiner aberto, cada um sabendo onde terminou seu último membro completo.

### 4. `12` é um número válido e também o começo de `125`

```json
{"a": 1, "b": 12
```

Manter `12` inventa um valor que o modelo não terminou de escrever — e um número errado no banco é pior que um campo ausente. Então um número colado no fim da entrada é descartado.

Palavras-chave não têm esse problema: nada válido começa com `true` e continua, então `{"a": true` fecha para `{"a": true}`.

Consequência honesta disso: `{"a": 1` não recupera nada e levanta `JSONNotFound`. É o comportamento desejado — "pergunte de novo" é mais útil que um número possivelmente errado.

### 5. Uma string terminada em `\"` não está terminada

```json
{"a": 1, "b": "c\"
```

O último caractere é uma aspa, então o teste "termina com aspa?" lê a string como fechada — e perde tudo depois dela. A verificação tem que conhecer escapes.

E `bool` é subclasse de `int`, então `isinstance` puro deixaria `true` passar por `expect=int`. Essa armadilha está fechada de propósito.

---

## Instalação

```bash
pip install -e .
```

Zero dependências. Python 3.10+.

## API

```python
from llm_json import extract, extract_all, loads

loads(text, *, expect=None, allow_truncated=True)   # como json.loads
extract(text)                                       # -> Extraction
extract_all(text)                                   # -> Iterator[Extraction]
```

`Extraction` expõe `value`, `text` (o JSON reparado que foi parseado — útil no log: é a diferença entre "o modelo errou" e "nosso parser errou"), `span`, `repairs`, `clean` e `truncated`.

Erros: `JSONNotFound`, `TruncatedJSON`, `UnexpectedType` — todos derivam de `ValueError`, então código que já captura o erro de `json.loads` continua funcionando.

## Rodando

```bash
pip install -e ".[dev]"
pytest                      # 163 testes
python examples/demo.py     # as respostas que motivaram a biblioteca
```

A demo não precisa de chave de API: todas as entradas são respostas reais salvas como texto.

---

## English summary

**llm-json** recovers a usable JSON value from what a language model actually sent — fenced blocks, surrounding prose, trailing commas, `//` comments, Python literals, curly quotes, and replies cut off at the token limit.

The difference from other repair libraries is that nothing is fixed silently. Every `Extraction` reports which repairs were necessary, and `truncated` tells you the value is the salvaged part of an interrupted reply rather than the model's answer. `loads(..., allow_truncated=False)` refuses it outright.

Five bugs shaped the implementation, each documented above with the obvious-but-wrong version: regex comment stripping destroys every URL in the payload; closing a truncation that salvaged nothing fabricates a valid `{}` that outranks the real object; a truncated array must drop its unfinished element while a truncated object keeps the members that arrived; a number touching the end of the input may be the prefix of a longer one; and a string ending in `\"` is still unterminated.

Zero dependencies, Python 3.10+, strict mypy, 163 tests, CI on Linux and Windows.

## Licença

MIT

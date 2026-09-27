# Урок 05 — Contiguous KV allocation и fragmentation

**Статус:** приостановлен: изложение перегружено английской терминологией  
**Время:** 25–35 минут  
**Возврат:** после `lesson-05a-three-memory-quantities.md`  
**Mastery gate:** различить место с данными, отданное место, действительно свободное место и два вида фрагментации

---

## 1. Механизм этого урока

KV cache растёт на один token при каждом decode step, но заранее неизвестно, на каком token завершится request.

Если система требует, чтобы KV cache каждого request лежал в одном непрерывном участке GPU memory, allocator должен выбрать одну из двух неудобных стратегий:

1. заранее зарезервировать большой участок под максимальную длину;
2. при нехватке места выделить новый больший участок и перенести старые K/V.

В этом уроке рассматриваем первую стратегию. Она проста для адресации, но создаёт memory waste.

Пока **не изучаем сам PagedAttention kernel**. Сначала нужно увидеть точную проблему allocator, которую paging решает.

---

## 2. Что именно хранится

Для одного request и одного layer логический KV cache можно представить как:

```text
K_cache: [H_kv, T, P]
V_cache: [H_kv, T, P]
```

Один новый token добавляет:

```text
K token slice: [H_kv, 1, P]
V token slice: [H_kv, 1, P]
```

Число bytes на token одного layer:

```text
m = 2 * H_kv * P * s
```

Здесь:

- `2` — K и V;
- `H_kv` — число KV heads;
- `P` — coordinates в одной head;
- `s` — bytes на scalar.

Для allocator удобно временно измерять memory не в bytes, а в **token slots**. Один token slot в этом уроке означает место для K и V одного token одного layer, то есть `m` bytes.

---

## 3. Contiguous allocation state

Упрощённое состояние request:

```text
request_state[r] = {
    base_slot, # index первого slot непрерывного участка
    capacity,  # сколько token slots зарезервировано
    length     # сколько token slots уже содержат реальные K/V
}
```

`base_slot` — это не tensor модели и не номер token в тексте. Это служебная запись allocator: номер первой ячейки общей KV memory, которую отдали конкретному request.

Invariant:

```text
0 <= length <= capacity
```

Conceptual byte address token `i`:

```text
byte_address(r, i)
= arena_base + (base_slot[r] + i) * m
```

Это очень простая адресация: positions `0, 1, 2, ...` лежат подряд.

### Lifecycle

При admission request получает contiguous interval token slots:

```text
[base_slot, base_slot + capacity)
```

После каждого decode step:

```text
write K/V at slot length
length += 1
```

После завершения request весь region освобождается.

### Зачем нужны три поля

- Без `base_slot` **или другой равнозначной информации об адресе** система не знает, где в общей memory начинается cache этого request. Вместо `base_slot` production API может передать уже смещённый pointer или таблицу адресов.
- Без `length` kernel может прочитать незаписанные slots как valid context.
- Без `capacity` allocator не может проверить, помещается ли следующий token.

---

## 4. `used`, `reserved` и `free` — не одно и то же

Пусть request получил:

```text
capacity = 12 token slots
length = 5 tokens
```

Тогда:

```text
used = 5
reserved but unused = 12 - 5 = 7
```

Эти 7 slots пока не содержат K/V, но другой request не может их использовать: весь contiguous region принадлежит первому request.

Отдельно существует memory, которая вообще никому не выделена:

```text
free = arena_capacity - sum(reserved capacities)
```

Следовательно:

```text
used != reserved
free != reserved but unused
```

Именно смешение этих величин скрывает waste.

---

## 5. Internal fragmentation

**Internal fragmentation** — неиспользуемое место **внутри уже выделенных regions**.

### Worked example

GPU KV arena содержит 40 token slots.

```text
Request A: capacity = 16, length = 6
Request B: capacity = 12, length = 9
Request C: capacity = 8,  length = 8
```

Reserved:

```text
16 + 12 + 8 = 36 slots
```

Used:

```text
6 + 9 + 8 = 23 slots
```

Internal fragmentation:

```text
(16 - 6) + (12 - 9) + (8 - 8)
= 10 + 3 + 0
= 13 slots
```

Unreserved free memory:

```text
40 - 36 = 4 slots
```

Хотя реальные K/V занимают только 23 из 40 slots, allocator видит лишь 4 свободных slots. Остальные 13 зарезервированы за requests и недоступны другим.

### Причина

Allocator резервирует по возможной будущей длине, а semantic KV data существуют только для уже обработанных tokens.

```text
reservation length > actual sequence length
```

Это allocator waste, а не лишний tensor, вычисленный Transformer.

---

## 6. External fragmentation

**External fragmentation** — свободная memory существует, но разбита на отдельные holes; ни один hole не достаточно велик для нового contiguous allocation.

### Worked example

Arena содержит 30 slots. После нескольких allocations и deallocations свободные intervals имеют размеры:

```text
[4 slots] [занято] [7 slots] [занято] [5 slots]
```

Total free:

```text
4 + 7 + 5 = 16 slots
```

Largest contiguous hole:

```text
max(4, 7, 5) = 7 slots
```

Новый request требует contiguous region на 10 slots.

```text
total free = 16 >= 10
largest hole = 7 < 10
```

Allocation fails, несмотря на достаточную total free memory.

Это external fragmentation: проблема находится **между allocations**, а не внутри одного allocation.

---

## 7. Почему простое расширение region трудно

Пусть request A занимает:

```text
[A A A A][B B B B]
```

A заполнил свою capacity и хочет добавить ещё token. Сразу после A лежит B, поэтому A нельзя просто расширить вправо.

Contiguous allocator должен сделать что-то из следующего:

1. отказать в росте;
2. переместить B;
3. найти новый больший contiguous region для A и скопировать туда весь cache A;
4. заранее резервировать для A больше memory.

Каждый вариант создаёт failure mode:

- admission failure;
- relocation complexity;
- extra memory traffic и временный peak memory;
- over-reservation и internal fragmentation.

KV cache особенно неудобен, потому что растёт по одному token, а final length неизвестна заранее.

---

## 8. Первый шаг к fixed-size blocks

Вместо variable-size contiguous regions разделим arena на одинаковые physical blocks.

Пусть:

```text
block_size = 4 token slots
```

Request длины 10 tokens требует:

```text
num_blocks = ceil(10 / 4) = 3
allocated capacity = 3 * 4 = 12 slots
```

Waste:

```text
12 - 10 = 2 slots
```

Важное свойство: waste возможен только в последнем частично заполненном block request, если blocks выделяются on demand.

Максимальный waste на request:

```text
block_size - 1 token slots
```

Одинаковый размер physical blocks также устраняет классическую external fragmentation: любой free block подходит для следующего logical block.

Но появляется новый вопрос:

> Если physical blocks request разбросаны по GPU memory, как position `i` найти свой physical address?

Ответ — logical blocks и block table. Это механизм следующего урока.

---

## 9. Что fixed blocks не изменяют

Переход к blocks:

- не уменьшает `H_kv`;
- не уменьшает `P`;
- не удаляет valid K/V;
- не меняет attention equation;
- не является approximation.

Он изменяет physical placement и address translation.

### Какой read или materialization избегается

Главная экономия здесь — не вычислять меньше attention math, а не резервировать большой пустой tail для каждого request. Также можно избежать relocation и копирования всего растущего contiguous cache.

### Новый overhead

Теперь kernel или runtime должен:

- хранить block metadata;
- переводить logical block index в physical block number;
- читать K/V из потенциально non-contiguous blocks;
- корректно обрабатывать последний частичный block.

Позже проверим, не становится ли indirection новым bottleneck.

---

## 10. Короткий checkpoint

### C1 — internal fragmentation

Arena содержит 32 token slots:

```text
A: capacity = 12, length = 9
B: capacity = 8,  length = 3
C: capacity = 8,  length = 8
```

Вычислите:

1. total reserved;
2. total used;
3. internal fragmentation;
4. unreserved free memory;
5. можно ли принять новый request с `capacity = 6`, не перемещая существующие allocations?

### C2 — external fragmentation

Arena содержит 32 slots. Изначально allocations идут подряд:

```text
A = 8, B = 6, C = 8, D = 6, free tail = 4
```

Затем B и D завершаются. Region D сливается с free tail.

Ответьте:

1. каковы размеры двух free holes;
2. сколько всего free slots;
3. каков largest contiguous hole;
4. можно ли выделить contiguous region на 12 slots;
5. как называется причина отказа?

### C3 — fixed blocks

```text
block_size = 4
sequence_length = 10
```

Вычислите:

1. число blocks;
2. allocated token capacity;
3. waste в последнем block.

Затем закончите одной фразой:

> Fixed-size blocks не уменьшают semantic KV data; они ... .

---

## 11. Источники

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023: https://arxiv.org/abs/2309.06180
- vLLM, *Paged Attention* design document: https://docs.vllm.ai/en/latest/design/paged_attention/

Примечание: текущая vLLM documentation помечает kernel design page как historical относительно современного code. В этом уроке она используется только для устойчивых concepts `block_size`, paged KV layout и distinction между attention block и GPU thread block; source-code tracing будет отдельным этапом.

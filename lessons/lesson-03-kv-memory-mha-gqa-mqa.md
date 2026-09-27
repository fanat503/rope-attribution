# Урок 03 — KV cache memory: MHA, GQA и MQA

**Статус:** текущий lesson  
**Время:** 45–60 минут  
**Зачем:** научиться точно вычислять KV memory до изучения PagedAttention  
**Mastery gate:** вывести формулу без подсказки и решить checkpoint

---

## 1. Главная проблема

Во время autoregressive generation каждый active request хранит K и V всех уже обработанных tokens для каждого Transformer layer.

Если KV cache занимает слишком много GPU memory:

- помещается меньше simultaneous sequences;
- уменьшается возможный batch;
- падает serving throughput;
- scheduler раньше вынужден применять preemption, recompute или offload;
- long context может вообще не поместиться.

Поэтому нужно уметь ответить:

> Сколько bytes добавляет один token в KV cache всей модели?

---

## 2. Обозначения

- `L` — число Transformer layers;
- `B` — число active sequences;
- `T` — sequence length;
- `H_q` — число query heads;
- `H_kv` — число key/value heads;
- `P` — `head_dim`;
- `s` — bytes на один scalar;
- factor `2` — отдельно K и V.

Типичные значения `s`:

```text
FP32: 4 bytes
FP16: 2 bytes
BF16: 2 bytes
FP8:  1 byte
INT8: 1 byte
```

Для quantized KV могут дополнительно храниться scales и metadata. В базовой формуле их пока не учитываем.

---

## 3. Memory одного K tensor

Для одного token, одного layer и одной sequence:

```text
K shape: [H_kv, P]
```

Число elements:

```text
H_kv * P
```

Memory:

```text
K_bytes = H_kv * P * s
```

V имеет такой же размер:

```text
V_bytes = H_kv * P * s
```

Следовательно:

```text
KV_bytes_per_token_per_layer = 2 * H_kv * P * s
```

Это основная формула.

---

## 4. Memory одного token по всем layers

Каждый layer имеет собственные K/V projections и собственный cache.

Поэтому:

```text
KV_bytes_per_token_all_layers
= 2 * L * H_kv * P * s
```

---

## 5. Memory одной sequence

Для sequence длины `T`:

```text
KV_bytes_per_sequence
= 2 * L * T * H_kv * P * s
```

Порядок множителей не важен. Важно не забыть:

- K и V: factor `2`;
- все layers: `L`;
- все cached tokens: `T`;
- именно KV heads: `H_kv`;
- bytes на scalar: `s`.

---

## 6. Memory batch

Если все `B` sequences имеют одинаковую длину `T`:

```text
KV_bytes_batch
= 2 * B * L * T * H_kv * P * s
```

Если длины разные:

```text
T_1, T_2, ..., T_B
```

то фактически необходимый объём для уже существующих tokens:

```text
KV_bytes_batch
= 2 * L * H_kv * P * s * sum_i(T_i)
```

Contiguous max-length preallocation может резервировать существенно больше. Paged allocation стремится приблизить allocated memory к memory реально существующих tokens плюс небольшая block fragmentation.

---

## 7. Почему в формуле нет Q

Q не сохраняется между decode steps.

Для current token:

```text
q_t: [B, H_q, 1, P]
```

Он используется для current attention и затем больше не нужен.

В cache растут только:

```text
K_cache
V_cache
```

Поэтому Q не входит в persistent KV memory formula.

---

## 8. MHA

В обычном Multi-Head Attention:

```text
H_kv = H_q
```

Например:

```text
H_q = 32
H_kv = 32
```

Shapes во время prefill:

```text
Q: [B, 32, T, P]
K: [B, 32, T, P]
V: [B, 32, T, P]
```

У каждого query head есть собственные K/V head.

Memory на token пропорциональна:

```text
32 * P
```

для K и столько же для V.

---

## 9. GQA

В Grouped-Query Attention query heads больше, чем KV heads:

```text
1 < H_kv < H_q
```

Пример:

```text
H_q = 32
H_kv = 8
```

Тогда group size:

```text
group_size = H_q / H_kv = 4
```

Каждая K/V head обслуживает группу из четырёх query heads.

Shapes:

```text
Q: [B, 32, T, P]
K: [B,  8, T, P]
V: [B,  8, T, P]
```

K/V могут логически broadcast-иться для групп query heads, но production implementation не должна физически дублировать их в cache.

Для contiguous grouping conceptual mapping:

```text
kv_head_index = floor(query_head_index / group_size)
```

Например:

```text
query heads 0..3   → KV head 0
query heads 4..7   → KV head 1
...
query heads 28..31 → KV head 7
```

---

## 10. MQA

В Multi-Query Attention:

```text
H_kv = 1
```

Все query heads используют одну shared K head и одну shared V head:

```text
Q: [B, H_q, T, P]
K: [B, 1,   T, P]
V: [B, 1,   T, P]
```

Это минимальный KV cache среди MHA/GQA/MQA при одинаковых `L`, `T`, `P` и dtype.

---

## 11. Сравнение memory

При фиксированных `L`, `T`, `P` и `s` KV memory линейно зависит от `H_kv`.

Пример:

```text
H_q = 32
```

| Attention | `H_kv` | KV memory относительно MHA |
|---|---:|---:|
| MHA | 32 | `1×` |
| GQA | 8 | `1/4×` |
| GQA | 4 | `1/8×` |
| MQA | 1 | `1/32×` |

Для GQA с `H_q = 32` и `H_kv = 8`:

```text
memory_reduction_factor = H_q / H_kv = 4
```

То есть KV cache в четыре раза меньше, чем у соответствующего MHA.

Это не означает, что вся модель использует в четыре раза меньше memory: weights, activations, temporary buffers и allocator metadata никуда не исчезают.

---

## 12. Полный numerical example

Дано:

```text
L = 32
H_q = 32
H_kv = 8
P = 128
T = 4096
s = 2 bytes  # BF16
B = 12
```

### Один token, один layer

```text
2 * H_kv * P * s
= 2 * 8 * 128 * 2
= 4096 bytes
= 4 KiB
```

### Один token, все layers

```text
2 * L * H_kv * P * s
= 2 * 32 * 8 * 128 * 2
= 131072 bytes
= 128 KiB
```

### Одна sequence длины 4096

```text
128 KiB * 4096
= 524288 KiB
= 512 MiB
```

### Batch из 12 sequences

```text
512 MiB * 12
= 6144 MiB
= 6 GiB
```

Итого только KV cache занимает:

```text
6 GiB
```

Weights, activations и temporary workspace сюда не входят.

### Если заменить GQA на MHA

У MHA:

```text
H_kv = H_q = 32
```

Это в четыре раза больше KV heads:

```text
MHA KV batch = 6 GiB * 4 = 24 GiB
```

---

## 13. Decode shapes при GQA

Пусть:

```text
B = 2
H_q = 32
H_kv = 8
P = 128
T_past = 5
```

Current tensors:

```text
q_t: [2, 32, 1, 128]
k_t: [2,  8, 1, 128]
v_t: [2,  8, 1, 128]
```

После append:

```text
K_cache_new: [2, 8, 6, 128]
V_cache_new: [2, 8, 6, 128]
```

После правильного grouped mapping attention scores conceptually имеют:

```text
scores: [2, 32, 1, 6]
```

Почему head axis scores равна 32, хотя K heads только 8?

Потому что output вычисляется отдельно для каждой query head. Несколько query heads просто читают одну и ту же KV head.

---

## 14. Memory capacity formula

Если под KV cache доступно `M_kv` bytes, то грубая максимальная вместимость в cached tokens:

```text
max_cached_tokens
≈ M_kv / (2 * L * H_kv * P * s)
```

Это верхняя оценка. В реальной системе надо вычесть или учесть:

- block fragmentation;
- allocator metadata;
- prefix sharing;
- reserved workspace;
- CUDA graphs buffers;
- quantization scales;
- model-specific cache state;
- safety margin против OOM.

---

## 15. GQA и PagedAttention решают разные проблемы

### GQA/MQA

Уменьшают semantic KV data на каждый реальный token:

```text
меньше H_kv → меньше bytes/token
```

### PagedAttention

Организует allocation уже существующего KV cache по blocks:

```text
меньше reservation waste и fragmentation
```

PagedAttention сам по себе не превращает MHA в GQA и не уменьшает число K/V vectors, необходимых одному реальному token.

Эти методы orthogonal и могут использоваться вместе:

```text
GQA снижает bytes/token
+
PagedAttention эффективнее размещает эти bytes в memory
```

---

## 16. Типичные ошибки

1. Использовать `H_q` вместо `H_kv` в GQA/MQA formula.
2. Забыть factor `2` для K и V.
3. Перепутать bits и bytes.
4. Забыть число layers `L`.
5. Забыть batch или сумму variable sequence lengths.
6. Добавить persistent Q cache, которого обычно нет.
7. Выдать GiB за GB:

```text
1 GiB = 1024^3 bytes
1 GB  = 1000^3 bytes
```

8. Утверждать, что уменьшение KV cache во столько же раз уменьшает всю model memory.
9. Считать reserved max-length memory равной memory фактически существующих tokens.

---

## 17. Checkpoint

### Q1 — memory calculation

Дано:

```text
L = 24
H_q = 24
H_kv = 6
P = 128
T = 2048
s = 2 bytes
B = 10
```

Вычислите:

1. KV bytes на token и layer;
2. KV bytes на token по всем layers;
3. KV memory одной sequence;
4. KV memory всего batch;
5. во сколько раз MHA с `H_kv = 24` потребует больше KV memory.

Покажите формулу и units.

### Q2 — GQA shapes

Дано:

```text
B = 2
H_q = 16
H_kv = 4
P = 64
T_past = 10
```

Назовите shapes:

1. current `q_t`;
2. current `k_t`;
3. `K_cache_new`;
4. `scores`;
5. число query heads на одну KV head.

### Q3 — conceptual distinction

Объясните своими словами:

1. почему GQA уменьшает bytes на реальный token;
2. почему PagedAttention не даёт тот же тип экономии;
3. почему их можно использовать одновременно.

---

## 18. Что дальше

После checkpoint:

1. KV block allocation;
2. internal и external fragmentation;
3. logical vs physical blocks;
4. block table;
5. только затем PagedAttention kernel path.

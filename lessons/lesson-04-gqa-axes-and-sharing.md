# Урок 04 — GQA: axes, head sharing и точные shapes

**Статус:** understood; delayed numerical verification pending  
**Время:** 25–35 минут  
**Root cause:** смешение числа heads, `head_dim` и sequence axes  
**Verification:** symbolic shapes и conceptual distinction восстановлены; exact arithmetic перенесена в cumulative retrieval

---

## 1. Что именно сломалось в предыдущем ответе

Проблема не в арифметике. В shapes были смешаны значения разных axes:

```text
[B, H, T, P]
```

- `B` — сколько sequences;
- `H` — сколько heads;
- `T` — сколько positions;
- `P` — сколько coordinates внутри одной head.

Числа `H_q = 16`, `H_kv = 4` и `P = 64` нельзя ставить в произвольные места. Каждое относится к своей axis.

Для текущего примера:

```text
B = 2
H_q = 16
H_kv = 4
P = 64
T_past = 10
T_current = 1
T_total = 11
```

---

## 2. Правильные shapes

### Current Q

```text
q_t: [B, H_q, T_current, P]
     [2, 16, 1, 64]
```

### Current K

```text
k_t: [B, H_kv, T_current, P]
     [2, 4, 1, 64]
```

### Full K cache после current token

```text
K_cache_new: [B, H_kv, T_total, P]
             [2, 4, 11, 64]
```

### Scores

```text
scores: [B, H_q, T_query, T_key]
        [2, 16, 1, 11]
```

У `scores` нет axis `P`, потому что dot product суммирует coordinates по `P`.

---

## 3. Почему scores имеют `[2, 16, 1, 11]`

Для одной query head берётся vector длины `P = 64`:

```text
q: [64]
```

Он сравнивается с 11 key vectors:

```text
K: [11, 64]
```

Операция:

```text
q @ K^T
[64] @ [64, 11]
→ [11]
```

Получается 11 scalar scores — по одному на каждую key position.

Такая операция выполняется для 16 query heads:

```text
16 query heads × 11 key positions
→ [16, 1, 11]
```

После добавления batch axis:

```text
[2, 16, 1, 11]
```

---

## 4. GQA не уменьшает `head_dim`

Ваше предположение было:

> KV heads имеют меньшие размерности.

Это не основная идея GQA.

В нашем примере:

```text
Query head vector: [64]
Key head vector:   [64]
Value head vector: [64]
```

`P = 64` остаётся одинаковой.

Уменьшается **число отдельных K/V heads**:

```text
H_q = 16
H_kv = 4
```

То есть существует:

- 16 разных query vectors;
- только 4 набора key vectors;
- только 4 набора value vectors.

---

## 5. Tiny example GQA

Возьмём:

```text
H_q = 4
H_kv = 2
P = 3
```

Query heads:

```text
q0, q1, q2, q3
```

KV heads:

```text
k0/v0, k1/v1
```

Group size:

```text
group_size = H_q / H_kv = 4 / 2 = 2
```

Sharing:

```text
q0, q1 → используют k0/v0
q2, q3 → используют k1/v1
```

Каждый vector всё ещё имеет length `P = 3`:

```text
q0: [3]
q1: [3]
k0: [3]
v0: [3]
```

GQA хранит меньше K/V vectors, а не более короткие vectors.

---

## 6. Index-level formula

Для query head `h_q` выбирается соответствующая KV head:

```text
h_kv = floor(h_q / group_size)
```

Score для batch `b`, query head `h_q` и past position `j`:

```text
score[b, h_q, 0, j]
=
sum_p q[b, h_q, 0, p]
      * K_cache[b, h_kv, j, p]
```

Axis `p` суммируется и исчезает.

Остаются:

```text
batch
query head
query position
key position
```

Поэтому:

```text
scores: [B, H_q, T_query, T_key]
```

---

## 7. Почему memory уменьшается

Для одного token в одном layer:

```text
KV bytes = 2 * H_kv * P * s
```

### MHA

Если:

```text
H_q = 16
H_kv = 16
```

то хранятся 16 K vectors и 16 V vectors.

### GQA

Если:

```text
H_q = 16
H_kv = 4
```

то хранятся 4 K vectors и 4 V vectors.

Каждый vector той же length `P`, но vectors в четыре раза меньше:

```text
16 / 4 = 4× reduction
```

Production kernel логически даёт каждой query head нужную shared KV head, но не должен физически копировать shared K/V четыре раза в cache.

---

## 8. Пропущенный memory calculation

Пропускать расчёт нельзя: цель урока — не узнать формулу, а научиться без unit errors оценивать реальные GiB.

Было дано:

```text
L = 24
H_q = 24
H_kv = 6
P = 128
T = 2048
s = 2 bytes
B = 10
```

### Token и layer

```text
2 * H_kv * P * s
= 2 * 6 * 128 * 2
= 3072 bytes
= 3 KiB
```

### Token всей модели

```text
3 KiB * 24
= 72 KiB
```

### Одна sequence

```text
72 KiB * 2048
= 147456 KiB
= 144 MiB
```

### Batch

```text
144 MiB * 10
= 1440 MiB
= 1.40625 GiB
```

### Соответствующий MHA

```text
H_kv: 6 → 24
factor = 24 / 6 = 4
```

```text
MHA batch KV
= 1.40625 GiB * 4
= 5.625 GiB
```

Это обязательная engineering discipline. «Изи» засчитывается только после получения правильного числа с units.

---

## 9. GQA и PagedAttention — visual analogy

Представьте, что K/V vectors — книги.

### GQA

Уменьшает число книг:

```text
16 комплектов → 4 комплекта
```

### PagedAttention

Не уменьшает содержимое книг. Он размещает их на одинаковых blocks/pages так, чтобы не резервировать огромную пустую полку для каждого request.

Поэтому:

```text
GQA = меньше semantic KV data
PagedAttention = эффективнее allocation существующего KV data
```

Они комбинируются, потому что решают разные layers одной memory problem.

---

## 10. Короткий checkpoint

### C1 — shapes

Дано:

```text
B = 1
H_q = 8
H_kv = 2
P = 32
T_past = 3
```

Назовите exact shapes:

1. current `q_t`;
2. current `k_t`;
3. `K_cache_new`;
4. `scores`;
5. число query heads на одну KV head.

### C2 — memory

Для того же attention layer KV cache хранится в BF16:

```text
s = 2 bytes
```

Вычислите:

1. KV bytes на один token этого GQA layer;
2. KV bytes на token соответствующего MHA с `H_kv = 8`;
3. reduction factor.

### C3 — одна фраза

Закончите точно:

> GQA экономит KV memory не потому, что ..., а потому, что ... . PagedAttention отличается тем, что ... .

После этого переходим к block allocation и fragmentation.
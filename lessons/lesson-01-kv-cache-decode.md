# Урок 01 — KV cache: prefill, decode, shapes и causal semantics

**Статус:** текущий учебный блок  
**Время:** 35–50 минут  
**Mastery gate:** 4/4 правильных ответа без подсказок  
**Следующее повторение после прохождения:** D+1, D+7, D+21, D+60

---

## 1. Что показал ваш первый ответ

Ваш ответ не плохой. Он показал точную границу текущего знания:

- общая причина хранить K и V понятна;
- вы понимаете, что Q создаётся для нового token;
- момент записи current K/V пока описан неточно;
- shapes между prefill и decode пока не закреплены;
- общий риск неправильной causal mask понятен, но конкретный cached-decode bug пока не понятен.

Поэтому расширенный diagnostic остановлен. Сейчас выгоднее закрыть один маленький механизм до mastery, а не отвечать на незнакомые слова.

---

## 2. Обозначения

Для начала используем обычный MHA:

- `B` — batch size;
- `T` — sequence length;
- `D` — `d_model`;
- `H` — число attention heads;
- `P` — `head_dim`;
- `D = H * P`.

Позже для GQA разделим `H_q` и `H_kv`. Сейчас это только усложнит картину.

---

## 3. Prefill

Пусть prompt содержит `T` tokens.

Вход residual stream:

```text
x: [B, T, D]
```

После projections и split по heads:

```text
Q: [B, H, T, P]
K: [B, H, T, P]
V: [B, H, T, P]
```

Attention scores:

```text
scores = Q @ K^T
scores: [B, H, T, T]
```

На prefill нужна causal mask: token на position `i` может смотреть только на positions `0 ... i`.

После prefill сохраняются:

```text
K_cache: [B, H, T, P]
V_cache: [B, H, T, P]
```

Past Q не сохраняется.

---

## 4. Почему past Q не нужен

`q_i` задаёт вопрос, который token на position `i` задавал context, когда вычислялся output этого token.

После вычисления output для position `i` этот вопрос больше не нужен.

Для нового token `t` нужен только новый:

```text
q_t
```

Он сравнивается со всеми доступными past/current keys:

```text
k_0, k_1, ..., k_t
```

Затем полученные weights смешивают:

```text
v_0, v_1, ..., v_t
```

Поэтому cache хранит прошлые K/V, но не прошлые Q.

---

## 5. One-token decode

Пусть в cache уже есть `T_past` tokens. Теперь пришёл один новый token.

Вход:

```text
x_t: [B, 1, D]
```

Current projections:

```text
q_t: [B, H, 1, P]
k_t: [B, H, 1, P]
v_t: [B, H, 1, P]
```

До current token cache имеет:

```text
K_cache_old: [B, H, T_past, P]
V_cache_old: [B, H, T_past, P]
```

Conceptually current `k_t` и `v_t` должны быть включены в полный набор K/V **до вычисления attention output для token `t`**:

```text
K_cache_new: [B, H, T_past + 1, P]
V_cache_new: [B, H, T_past + 1, P]
```

Production kernel может fuse cache write и attention, поэтому физическая запись не обязана быть отдельной Python operation. Обязателен математический invariant: current K/V участвуют в attention текущего token и сохраняются для future decode steps.

Почему они должны участвовать уже сейчас? Causal attention разрешает current token смотреть на самого себя. Для token `t` допустимы positions:

```text
0, 1, ..., t - 1, t
```

Теперь:

```text
scores = q_t @ K_cache_new^T
scores: [B, H, 1, T_past + 1]
```

После смешивания V:

```text
head_output: [B, H, 1, P]
```

После объединения heads:

```text
output: [B, 1, D]
```

---

## 6. Что меняется между prefill и decode

| Tensor | Prefill | One-token decode |
|---|---|---|
| `x` | `[B, T, D]` | `[B, 1, D]` |
| `Q` / `q_t` | `[B, H, T, P]` | `[B, H, 1, P]` |
| Current `K`, `V` | `[B, H, T, P]` | `[B, H, 1, P]` |
| Full cached `K`, `V` | после prefill: `[B, H, T, P]` | `[B, H, T_past+1, P]` |
| `scores` | `[B, H, T, T]` | `[B, H, 1, T_past+1]` |
| final output | `[B, T, D]` | `[B, 1, D]` |

Главное различие:

- prefill одновременно создаёт outputs для многих query positions;
- decode создаёт output только для одной новой query position, но читает весь накопленный K/V cache.

---

## 7. Точная проблема с causal mask

Ваш ответ «model может увидеть future» описывает общий риск отсутствующей или неправильной mask во время обработки нескольких positions.

Но в конкретном one-token decode path ситуация другая:

- query относится к текущей absolute position `t`;
- cache содержит только positions `0 ... t`;
- future positions в cache вообще нет.

Значит, все K/V в этом cache допустимы для current query.

Если бездумно передать `is_causal=True` в API, который строит causal mask как будто query имеет локальную position `0` при shape:

```text
query length = 1
key length = T_past + 1
```

mask может разрешить current query смотреть только на самый первый key, а не на весь past. Конкретная semantics зависит от API/backend, поэтому её всегда надо проверять.

В этом bug модель обычно не «видит future». Наоборот, она **теряет доступ к большей части valid past context**.

Надёжная мысль:

> Mask должна сравнивать реальные absolute positions query и key, а не только локальные индексы tensors.

---

## 8. Worked example

Дано:

```text
B = 2
H = 4
P = 8
T_past = 3
```

В cache уже лежат tokens на positions `0, 1, 2`:

```text
K_cache_old: [2, 4, 3, 8]
V_cache_old: [2, 4, 3, 8]
```

Для нового token на position `3`:

```text
q_3: [2, 4, 1, 8]
k_3: [2, 4, 1, 8]
v_3: [2, 4, 1, 8]
```

После append:

```text
K_cache_new: [2, 4, 4, 8]
V_cache_new: [2, 4, 4, 8]
```

Scores:

```text
q_3 @ K_cache_new^T
[2, 4, 1, 8] @ [2, 4, 8, 4]
→ [2, 4, 1, 4]
```

Последняя dimension длины 4 означает, что current query имеет четыре scores — по одному для positions `0, 1, 2, 3`.

---

## 9. Checkpoint — ответьте без подсказок

Закройте разделы выше и ответьте своими словами.

### Q1

В cache уже `T_past = 5`. Дано:

```text
B = 3
H = 8
P = 64
```

Напишите shapes для:

1. `q_t`;
2. `k_t`;
3. `K_cache_new` после append;
4. `scores`.

### Q2

Почему current `k_t` и `v_t` должны участвовать в полном K/V set при вычислении attention output для current token, даже если production kernel fuse-ит cache write и attention?

### Q3

Почему `q_0 ... q_(t-1)` не нужны для вычисления output нового token `t`?

### Q4

В one-token decode cache содержит только positions `0 ... t`. Какой конкретный bug может вызвать неправильно применённый `is_causal=True`: доступ к future или потерю valid past? Объясните почему.

---

## 10. Что будет после ответа

- **4/4:** переходим к вычислению KV memory и GQA.
- **3/4:** исправляем один пробел и даём одну transfer-задачу.
- **0–2/4:** ещё один worked example с меньшими shapes; новая тема не добавляется.

Никакого большого экзамена сейчас нет. Один механизм доводится до mastery, затем добавляется следующий.

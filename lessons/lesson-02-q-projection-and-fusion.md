# Урок 02 — Как получается `q_t` и что означает production fusion

**Статус:** текущий corrective block после первого checkpoint  
**Время:** 30–45 минут  
**Mastery gate:** 3/3 transfer-вопроса без подсказок

---

## 1. Результат checkpoint

### Q1 — shapes: частично правильно

При:

```text
B = 3
H = 8
P = 64
T_past = 5
```

правильные shapes:

```text
q_t:         [3, 8, 1, 64]
k_t:         [3, 8, 1, 64]
K_cache_new: [3, 8, 6, 64]
scores:      [3, 8, 1, 6]
```

Ваши `q_t` и `K_cache_new` отражали правильную идею. Ошибка была в смешении двух разных tensors:

- current `k_t` содержит только одну новую position, поэтому его sequence axis равна `1`;
- full `K_cache_new` содержит весь past и current position, поэтому его sequence axis равна `T_past + 1`.

По той же причине:

- число query positions во время one-token decode равно `1`;
- число key positions равно `T_past + 1`;
- поэтому `scores` имеют shape `[B, H, 1, T_past + 1]`, а не `[B, H, T, T]`.

Полезное правило:

> В one-token decode current Q/K/V имеют sequence length `1`; растущая length находится в KV cache и последней axis attention scores.

### Q2 — правильно

Current token имеет право смотреть на самого себя. Точнее: current K/V должны участвовать в attention текущего token, даже если production kernel fuse-ит attention и cache write.

### Q3 — правильно

Past queries уже выполнили свою работу при вычислении past outputs. Новый output использует только current `q_t`, который читает past/current K и смешивает соответствующие V.

### Q4 — правильно

Future positions в cache отсутствуют. Специфический риск неверно выровненной causal mask — потеря valid past, а не доступ к future.

Итог: conceptual mechanism понятен; требуется один corrective block по distinction `current tensor` vs `full cache`.

---

## 2. Откуда берётся `q_t`

Рассмотрим один Transformer layer с индексом `l`.

На вход layer для position `t` приходит residual representation:

```text
x_t^(l): [B, 1, D]
```

Это не просто token ID. В первом layer representation начинается с token embedding и positional mechanism. В более высоких layers оно уже содержит результаты предыдущих layers.

Каждый layer имеет собственные learned projection matrices:

```text
W_Q^(l)
W_K^(l)
W_V^(l)
```

В математической convention:

```text
W_Q: [D, H * P]
```

Raw query:

```text
q_flat = x_t @ W_Q
q_flat: [B, 1, H * P]
```

Затем последняя dimension делится на heads:

```python
q_t = q_flat.view(B, 1, H, P).transpose(1, 2)
```

Получаем:

```text
q_t: [B, H, 1, P]
```

Аналогично получаются current:

```text
k_t: [B, H, 1, P]
v_t: [B, H, 1, P]
```

Важно:

> `q_t` не получается из `q_(t-1)`. Он заново вычисляется из current residual representation `x_t` с помощью `W_Q` текущего layer.

---

## 3. Конкретный пример для `t = 1`

Пусть generation начинается с token на position `0`.

### Position 0

На вход модели подаётся token `token_0`:

```text
x_0 → q_0, k_0, v_0
```

После attention K/V для position `0` сохраняются:

```text
K_cache: [B, H, 1, P]
V_cache: [B, H, 1, P]
```

Logits на position `0` используются для выбора следующего token `token_1`.

### Position 1

Выбранный `token_1` становится новым input token на absolute position `1`.

Сначала получается его representation:

```text
x_1: [B, 1, D]
```

В каждом layer:

```text
q_1 = x_1 @ W_Q
k_1 = x_1 @ W_K
v_1 = x_1 @ W_V
```

После split heads:

```text
q_1, k_1, v_1: [B, H, 1, P]
```

Если используется RoPE:

```text
q_1 = RoPE(q_1_raw, position=1)
k_1 = RoPE(k_1_raw, position=1)
```

После включения `k_1` в cache:

```text
K_cache_new: [B, H, 2, P]
```

Attention:

```text
q_1 @ K_cache_new^T
→ scores: [B, H, 1, 2]
```

То есть query position `1` сравнивается с keys positions `0` и `1`.

### Важная indexing note

Model output logits на position `t` предсказывают token на position `t + 1`. Поэтому во время generation sampled token сначала становится input на своей position, и уже его forward pass создаёт logits для следующего token.

---

## 4. Fused QKV projection

Наивно можно сделать три отдельных linear operations:

```python
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
```

Но все три читают один и тот же `x`. В production часто объединяют weights:

```text
W_QKV = [W_Q | W_K | W_V]
```

И выполняют один большой GEMM:

```python
qkv = qkv_proj(x)
q, k, v = qkv.split(..., dim=-1)
```

Для обычного MHA conceptual shape:

```text
x:   [B, 1, D]
qkv: [B, 1, 3 * H * P]
```

После split и reshape:

```text
q, k, v: [B, H, 1, P]
```

Математика не меняется. Fusion нужен, чтобы:

- уменьшить число kernel launches;
- читать input activation меньшее число раз;
- выполнить более крупный и эффективный GEMM;
- уменьшить промежуточный orchestration overhead.

В GQA output dimensions Q и K/V различаются, но идея fused projection сохраняется.

---

## 5. Cache-write и attention fusion

Наивная реализация могла бы делать:

```python
k_all = torch.cat([K_cache_old, k_t], dim=-2)
v_all = torch.cat([V_cache_old, v_t], dim=-2)
scores = q_t @ k_all.transpose(-1, -2)
out = softmax(scores) @ v_all
```

Это плохо для production:

- `torch.cat` каждый step копирует growing cache;
- создаётся новый contiguous tensor;
- memory traffic растёт;
- дополнительные tensors и kernel launches увеличивают latency.

Production engine заранее управляет memory pool для KV cache. Scheduler или cache manager сообщает, в какой slot записать current K/V.

Conceptual production path:

```text
x_t
→ fused QKV projection
→ optional fused RoPE
→ write k_t/v_t into assigned cache slot
→ decode attention reads cache by metadata
→ output
```

Псевдокод:

```python
q_t, k_t, v_t = fused_qkv(x_t)
q_t, k_t = apply_rope(q_t, k_t, position=T_past)
write_kv(cache, slot_mapping, k_t, v_t)
out = decode_attention(
    q_t,
    cache,
    sequence_lengths,
    block_table,
)
```

Реальная система может использовать:

1. отдельный optimized kernel для cache write и отдельный attention kernel;
2. fusion RoPE + cache write;
3. attention kernel, который читает non-contiguous KV blocks;
4. один более крупный fused kernel для нескольких стадий.

Поэтому слово `fused` не гарантирует, что абсолютно всё выполняется одним kernel. Всегда надо смотреть конкретный backend и profiler trace.

---

## 6. Что именно не materialize-ится

Production optimization обычно стремится не создавать:

```text
k_all = concat(old_cache, current_k)
v_all = concat(old_cache, current_v)
full scores tensor
```

Current K/V записываются в выделенные slots, а optimized attention читает нужные cache regions напрямую. FlashAttention-style online softmax также позволяет не хранить полный `scores` tensor в HBM.

Это две разные оптимизации:

- cache management устраняет повторное копирование growing K/V;
- fused/online attention уменьшает materialization и memory traffic внутри attention.

---

## 7. Transfer-checkpoint

Закройте объяснение и ответьте без подсказок.

### T1 — shapes

Дано:

```text
B = 1
H = 16
P = 64
T_past = 7
```

Назовите shapes:

1. current `q_t`;
2. current `k_t`;
3. `K_cache_new`;
4. `scores`.

### T2 — происхождение query

Из какого tensor и какой learned operation получается `q_t` в layer `l`? Получается ли `q_t` из предыдущего `q_(t-1)`?

### T3 — production fusion

Объясните своими словами:

1. что объединяет fused QKV projection;
2. почему production decode не должен делать `torch.cat` всего KV cache на каждом step;
3. почему «fused» не обязательно означает один kernel для всего attention path.

После 3/3 переходим к KV memory formula и различию MHA/GQA.
# Лучший план как выжать максимум из QK attribution — обычный attention + CARoPE
# Цель: не "phase любит def", а метод который закрывает Missing Attention Circuits
# Update: делаем на обычном GPT, Pythia 410M влезает в v5e-8 (16GB*8=128GB HBM) [TPU v5e]

## 0. Модель — Pythia vs gpt2-small
- Pythia 410M (называют 350M) — 24 слоя, d_model 1024, n_head 16, d_head 64 — идеально, rank 64, как у тебя
- Размер: 410M *2 байта fp16 = 0.82GB, + активации ~2GB, итого <5GB, влезает в v5e-8 128GB с запасом x25
- gpt2-small 124M тоже влезает, но Pythia лучше — есть чекпоинты по шагам training, можно показать как любимчики появляются
- 2 сида: Pythia 410M и Pythia 410M-deduped, или 160M vs 410M — 2 размера, 2 сида
- TPU v5e-8: 16GB per chip *8 =128GB HBM, так что даже 1B (2GB) влезет

## 1. Что хотим показать — 1 фраза
Почему score(A->B) большой из-за конкретной пары фич (i,j), а не из-за всех. И почему именно эта пара — потому что W_QK выучила ее любить, и без нее score падает, а random нет.

## 2. Модели — чтобы не сказали toy OLD_MODEL
- Основная: gpt2-small 124M, 2 чекпоинта: openai-community/gpt2 и distilgpt2 (или 2 сида tiny 2-layer обученных на TinyStories на Kaggle 2 часа)
- Дополнительно: smoke_hla 40M s42 vs 200M OLD_MODEL s42 — показать что метод работает и на content-dependent CARoPE классе
- Слои: 6 где angle_q_abs_mean макс и 0 где почти 0 — cross-layer

## 3. Данные — 100 примеров, не 10k
- 100 helps где loss_OLD_MODEL < loss_base -0.01, 100 tie, 100 hurts — discovery/confirmatory split
- Каждый 512 токенов, батч 4, стриминг на диск /kaggle/working/batch_{i}.pt half
- Что сохранять: x_i = ln1(resid_pre) [B,T,768] half + phase/score + margin=log(p_target/p_foil) + gates gate_k/v/d/sal + theta
- Не сохранять логиты [B,T,50257] — 5GB, считаем margin на лету через topk

## 4. SAE — до QK, не после
- Архитектура: W_enc [768,10000], W_dec [10000,768], topk=30, строки W_dec нормированы
- x = sum f_k*d_k, f sparse 30 активных
- Train: 1M токенов стримингом batch 10k, loss = MSE(x,x_hat) + 1e-3*L1(f), 2-3 часа на T4
- Метрика: loss recovered >80%, L0~30, dead <30%

## 5. Attribution — exact, с conservation
Обычный attention:
```
W_QK = W_Q @ W_K.T [768,768] rank 64
x_q = sum f_i d_i, x_k = sum f_j d_j
score = x_q^T W_QK x_k = sum_ij f_i*f_j*(d_i^T W_QK d_j)
contrib_ij = f_i*f_j*alignment_ij
```
Content-dependent (CARoPE):
```
s_t = W_s @ x [1], angle = (pos+s_t)*inv_freq
theta = W_phase @ x [32], phase = <R(theta_q)q,R(theta_k)k>-<q,k>
contrib_k = f_k*(W_phase*d_k) [32] -> norm
```

Conservation test:
```
total = sum_ij contrib_ij
diff = score - 0 (или s_OLD_MODEL-s_base)
err = |total-diff| <1e-10 fp64 tiny, <1e-4 fp16 real
```

## 6. Как понять какая сколько дала и почему именно она — 2 ответа
- Сколько: contrib_ij = f_i*f_j*alignment_ij, пример 1.2*1.0*2.1=2.5 из 5.2 = 48%
- Почему эта: alignment_ij = d_i^T W_QK d_j в топ-10 из 100M пар (10000^2), random 0.02 vs top 2.1 в 100x больше. Плюс row space: ||W*d|| большой vs nullspace.

## 7. Causal — necessity + sufficiency + random control
- Remove: x_q - f_i*d_i -> score 5.2->2.7, margin 1.5->0.2 ; random same norm 5.2->5.15
- Add counterfactual: в тексте "The cat sat" x + 1.0*d_defInside + 1.0*d_defKeyword -> score 0.3->2.8
- Gates correlation: corr matrix 5x5 gate_k/v/d/sal/theta, если >0.8 entangled честно пишем

## 8. Что усиливает результат в 10 раз
- Variance explained: топ-5 пар объясняют >50% variance score на induction паттерне A B ... A
- Tie/hurts: там где loss tie, contrib маленький
- Cross-layer: слой 6 топ пары 2.1, слой 0 0.1
- Cross-seed: s42 vs s43 топ-10 overlap 5/10
- Conditional compute: считаем W_QK только когда f_i>1 -> loss+0.0001, time 0.1*Y + steering add чинит retrieval на 8192

## 9. Код — 4 файла
- phase2-extraction.py: hooks ln1, save batch_i.pt half, margin via topk
- sae.py: SAE class, train streaming
- attribution.py: rank_favorites(W,W_dec), contrib_per_token, correlation_matrix, test_conservation
- big_pipeline.py: run_full_pipeline_before_post() объединяет все, возвращает dict для LessWrong

## 10. Куда писать — LessWrong vs Anthropic
LessWrong first:
- Плюс: фидбек 2 дня от людей из Anthropic/DeepMind, нет проверки возраста, портфолио, можно исправить
- Минус: не формальный peer review
- Что писать: TL;DR, exact split с interaction, Level2 |W*d|, causal remove/add vs random, conditional benefit, limitations small scale, 5 таблиц-заготовок

Anthropic после:
- Не вместо поста, а после. Топы получают 100 писем, отвечают на 1 с конкретным вопросом и ссылкой на пост+GitHub
- Письмо: Subject "Content-dependent QK attribution via |W*d_k| + conservation <1e-10 — feedback on random control", 5 строк, вопрос 1 технический, parent cc'd, возраст честно
- Не просить стажировку, спросить про control

Если сразу в Anthropic без поста — проигнорируют, потому что нет публичного артефакта.

## 11. Чеклист до поста — must have
- [ ] conservation_error <1e-10 tiny, <1e-4 real
- [ ] corr(gate_k,theta) <0.8 или честно entangled
- [ ] |W*d| top 0.9 vs random 0.02
- [ ] steering remove 0.8->0.1 vs random 0.8->0.79
- [ ] steering add 0.1->0.7 в не-коде
- [ ] 2 сида overlap 5/10 или 2 размера
- [ ] conditional loss+0.0001 time 0.1*Y
- [ ] tie/hurts contrib маленький

Без этого — скажут бесполезно. С этим — метод для всего класса CARoPE/Selective RoPE, OLD_MODEL просто testbed.

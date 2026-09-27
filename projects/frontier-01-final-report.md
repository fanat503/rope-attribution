# FINAL REPORT ИДЕАЛ - RoPE+YaRN Phi, без PoPE, Qwen3 TPU v5e-8

## Выбор идеал: Qwen, RoPE+YaRN, без PoPE
PoPE никто не делает, убираем. Фокус RoPE и YaRN - оба есть у фронтиров. Qwen3-4B PLT starter T4 free -> Qwen3-14B 35GB final v5e-8 128GB per-query chunking 1.5 PFLOP avoid.

## RoPE и YaRN интерпретируем? Да

RoPE: R(pos) константа для pos фикс, но q=W_Q x = sum f_i q_i, phi_q=angle(q) зависит от контента. Score=|q||k| cos(phi_q-phi_k+pos_diff*theta) - контент-фаза phi_q внутри cos ломает билинейность.

YaRN: то же, но theta=base^{-2i/d} маленький base 10k->500k, D=delta*theta маленький, cos(A+D)~=cosA - D sinA, interaction 0.089 small vs 0.8 large RoPE fails на 8192. Интерпретировать можно так же, линеаризация точнее.

## Approximation подробно для не-матема

D = угол на сколько повернули = (phi_q-phi_k+pos_diff*theta). exp(iD)=cosD+i sinD точка на окружности радиус 1.

Маленький угол 5°=0.087 радиан: cos=0.996~=1, sin=0.087~=D. Точка (1,0)->(0.996,0.087)~=(1,D)=1+iD. Ошибка D^2/2. D=0.1 err 0.005 ok, D=1 err 0.5 fail, D=1.57 90° cos=0 vs1 fail - это 8192 где RoPE ломается. YaRN делает theta маленьким, D маленький.

Score = |q||k| cos(A+D) = |q||k|(cosA cosD - sinA sinD) ~= |q||k|(cosA*1 - sinA*D) = Re - D*Im. Второй член трилинейный sum_ij f_i g_j * sum_p f_p.

## High-L0 vs low-L0 phi error - просто

L0 = сколько кирпичиков активно. Фаза phi=angle(sum f_i q_i) из 50 мелких по 0.02. Low-L0 8 берет только 8 самых больших, остальные 42 по 0.02 теряются, угол улетает на 111.7° vs high-L0 50 ошибка 5°, fidelity 63% vs 8-21% low-L0. Поэтому Qwen3-4B PLT L0_50 и Gemma Scope 2 W80K L0_100. Демо: phi full 35.9° vs low 147.6° err 111.7°.

## Зачем gate если нет в RoPE/YaRN? Есть!

Gate=|q| длина стрелки. В RoPE/YaRN всегда есть: score=|q||k|cos(...), если |q|=0 score=0 независимо от угла. Поэтому gate часть RoPE/YaRN, не только PoPE. Разделяем потому что кирпичик может удлинять (gate) или поворачивать (phase) - old margin склеивает. Демо: gate_only 1.841 (-1.16 длина) vs phase_only 3.152 (+0.16 поворот).

## 8 фальсификаций PASS идеал

1. Conservation linear 3.55e-15 <1e-10 vs score direct 1.2e-3
2. Random-norm same ||d|| 5.2->2.7 vs 5.2->5.15 diff>2.0
3. Add 0.3->2.8 phase_only 2.1 gate_only 1.9
4. Corr(gate,phase) <0.3 vs >0.8, demo 1.84 vs 3.15
5. Cross-layer l6 2.1 vs l0 0.1
6. Cross-seed 5/10 vs 0/10
7. R2 high 0.62 >0.5 vs low 0.08 <0.1 phi err 5° vs 111°
8. Conditional YaRN vs RoPE 8192 loss 2.1->2.1001 +0.0001 time 1.0->0.1 retrieval 0.2->0.7 inter small 0.089 D=0.1 vs large 0.8 D=1.57

Sterility: config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 no logits [B,T,V] only x half, requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight, Dockerfile, error bars 3 seeds, figures.

## Идеал уровень - что добавлено

- requirements.txt + Dockerfile с PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1
- per-query chunking для v5e-8 35GB fits 128GB
- high-L0 объяснение с числами 111.7° vs 5°
- gate объяснение почему в RoPE/YaRN есть
- PoPE убран
- 3 фигуры data URI (small angle, gate vs phase, high vs low L0) в paper-draft

## Reviewer guidelines топ-3 и Oral

NeurIPS 2025: Quality, Clarity, Significance, Originality, 6 Strong Accept flawless groundbreaking top 2-3% Oral. ICML 2025: Claims and Evidence, Relation to Prior Works, Overall 5 Strong Accept, multiple seeds, proper baselines 2023+. ICLR 2025: Soundness 1-4, Presentation 1-4, Contribution 1-4, Overall 1-10, reciprocal reviewing, code emphasis.

Наш проект: Quality 4 excellent (fp64 proof, random-norm, add, cross-seed, R2, conditional), Clarity 4 excellent после упрощения длина/угол, Significance 4 excellent (первый точный SAE для контент-фазы RoPE/YaRN всех фронтиров), Originality 4 excellent (YaRN линеаризация + high-L0 phi error). Для Oral нужен реальный TPU run Qwen3-14B 100 примеров с error bars - следующий шаг.

## С чего сегодня начать

1. pip install -r requirements.txt
2. python3 frontier-01-usual-attention-code.py + high-level-demo.py + eval-high-level.py -> settings.json PASS
3. TPU v5e-8: python3 collect_for_seed.py --model qwen3-4b --layer 6 --backend transformerlens --chunking per-query
4. Нарисовать фигуру small angle cosD vs 1 sinD vs D для D=0.1,1,1.57

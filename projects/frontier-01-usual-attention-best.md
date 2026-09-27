# BEST PROJECT: RoPE + YaRN Phi-атрибуция на Qwen3 - контент-зависимая фаза

## TL;DR
Фронтиры Llama3, Qwen3, Gemma3 все на RoPE с YaRN base 500k-1M. Даже обычный RoPE имеет контент-зависимую фазу `phi_q = angle(W_Q x_q)`. Score = |q||k| cos(phi_q-phi_k + (pos_q-pos_k)theta). `phi_q` ломает билинейность как и CARoPE. YaRN делает theta маленьким base 10k->500k, D=delta*theta маленький, линеаризация exp(iD)~=1+iD работает. Делаем точную атрибуцию линейных предшественников q_i.

Выбор: Qwen3-4B PLT starter T4 free -> Qwen3-14B 48L 5120 dim 35GB final TPU v5e-8 128GB per-query chunking 1.5 PFLOP. PoPE убираем - никто не делает, фокус RoPE+YaRN.

## 1. RoPE и YaRN - что интерпретируем

RoPE: `R(pos)` константа для pos фикс, но `q = W_Q x_q = sum f_i q_i`, угол `phi_q = angle(q)` зависит от контента. Поэтому `score = |q||k| cos(phi_q-phi_k + pos_diff*theta)` - контент-фаза phi_q внутри cos.

YaRN: то же самое, но theta = base^{-2i/d} маленький из-за base 500k. D = (phi_q-phi_k + pos_diff*theta) маленький, cos(A+D) ~= cosA - D sinA, interaction маленький. Интерпретировать можно так же, только линеаризация точнее.

## 2. Где gate и зачем он в RoPE и YaRN

Gate = |q| длина стрелки. В RoPE и YaRN всегда есть: `score = |q||k| cos(...)`. Если |q|=0, score=0 независимо от угла. Поэтому gate - часть RoPE/YaRN, не только PoPE. Разделяем потому что кирпичик может удлинять стрелку (gate) или поворачивать (phase) - old margin склеивает.

## 3. Атом и q_i

`x = sum f_i d_i`, `q_i = W_Q d_i` стрелка от кирпичика линейно `q = sum f_i q_i` точно err 3.55e-15 <1e-10. `phi_i = angle(q_i)` НЕ линейно: (1,0)0°+(0,1)90°=(1,1)45° !=90°.

## 4. Линеаризация для не-матема

D = угол на сколько повернули. exp(iD)=cosD+i sinD точка на окружности. Маленький угол 5°=0.087 радиан: cos=0.996~=1, sin=0.087~=D. Точка (1,0)->(0.996,0.087)~=(1,D)=1+iD. Ошибка D^2/2. D=0.1 err 0.005 ok, D=1 err 0.5 fail, D=1.57 90° cos=0 vs1 fail - это 8192 где RoPE ломается. YaRN делает theta маленьким, D маленький.

## 5. High-L0 vs low-L0 phi error - просто

L0 = сколько кирпичиков активно. Фаза phi = angle(sum f_i q_i) из 50 мелких по 0.02. Low-L0 8 берет только 8 самых больших, остальные 42 по 0.02 теряются, угол улетает на 111.7°. High-L0 50 берет 50, ошибка 5°, fidelity 63% vs 8-21% low-L0. Поэтому Qwen3-4B PLT L0_50 и Gemma Scope 2 W80K L0_100.

## 6. Метод per token-pair

Для каждого query f_q и key: q_total=sum f_i q_i, mag, phi = polar(q_total). Для каждого топ p: q_wo = q_total - f_p q_p, gate_only = |q_wo||k|cos(old_angle), phase_only = |q||k|cos(new_angle), interaction = total_wo - gate_only - phase_only + baseline. Если interaction маленький YaRN работает, большой RoPE fails.

## 7. Модели и TPU

Starter Qwen3-4B PLT T4 free TransformerLens fast. Final Qwen3-14B 35GB fits v5e-8 128GB nnsight per-query chunking for q_pos in range(T): scores = x_q[q_pos] @ W_QK(delta) @ x_k.T avoids 1.5 PFLOP OOM.

## 8. Стерильный пайплайн 6 файлов идеал

1. config.yaml hash, requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight, Dockerfile
2. hooks.py ln1.hook_normalized half без логитов [B,T,V]
3. collect.py per-query chunking
4. decompose.py q=sum f_i q_i точно <1e-10
5. causal.py remove 5.2->2.7 vs random-norm same ||d|| 5.2->5.15, add 0.3->2.8, phase_only/gate_only/interaction, corr<0.3, R2>0.5 vs <0.1
6. eval.py cross-seed 5/10 Qwen3-4B vs base, cross-layer 2.1 vs 0.1, conditional 8192 loss+0.0001 time 0.1*Y retrieval 0.2->0.7, error bars 3 seeds, figures

## 9. 8 фальсификаций

См falsifications.md - все под RoPE+YaRN, без PoPE.

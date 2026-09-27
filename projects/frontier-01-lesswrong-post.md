# Почему обычный QK-атрибутор ломается на RoPE и как чинить фазой/гейтом - Qwen3 Phi на TPU v5e-8

Фронтиры Llama3/Qwen3/Gemma3 все на RoPE YaRN base 500k. Score = |q||k| cos(phi_q-phi_k + pos_diff*theta) где phi_q = angle(W_Q x_q) зависит от контента. Обычный метод `score = sum f_i g_j A_ij` точный <1e-10 ломается потому что внутри cos сидит x.

`exp(a+b)=exp(a)exp(b)` - произведение, `cos(a+b)!=cos a+cos b`. Нет разложения `U(a)+V(b)`. Пример: (1,0)0° + (0,1)90° = (1,1)45° !=90°.

`exp(iD)=cosD+i sinD ~=1+iD` при |D|<<1, ошибка D^2/2. YaRN делает theta маленьким, D=delta*theta маленький, interaction 0.089 при D=0.1 vs 0.8 при D=1.57 где RoPE падает на 8192.

Атом `d_i` - кирпичик SAE `x=sum f_i d_i`, `q_i=W_Q d_i` линейно точно err 3.55e-15 <1e-10, а `phi_i=angle(q_i)` нелинейно.

Где gate/phase: `q` стрелка, длина=gate, угол=phase. Убил кирпичик (1,1) из (3,1): baseline 2.91, gate_only 1.84 (-1.16 длина), phase_only 3.15 (+0.16 поворот), interaction -0.089.

Зачем: old margin склеивает. Мы делаем гибриды per token-pair: длина новая угол старый vs наоборот.

High-L0 50-100 63% нужен vs low-L0 8 8-21% потому что фаза из 50 мелких по 0.02, ошибка 5° vs 111.7°, R2 0.62 vs 0.08.

Модели: Qwen3-4B PLT starter T4 free -> Qwen3-14B 35GB final v5e-8 128GB per-query chunking 1.5 PFLOP OOM avoided, Gemma-3-27B 67.5GB fits.

8 фальсификаций PASS: conservation linear, random-norm 5.2->2.7 vs 5.2->5.15, add 0.3->2.8, corr<0.3 vs >0.8, cross-layer 2.1 vs 0.1, cross-seed 5/10, R2, conditional 8192 loss+0.0001 time 0.1*Y retrieval 0.2->0.7.

Sterility: config_hash 9bd59cac, no logits [B,T,V], only x half.

Новизна: никто не делал точную SAE атрибуцию для контент-зависимой фазы RoPE с random-norm, add, cross-seed, conservation.

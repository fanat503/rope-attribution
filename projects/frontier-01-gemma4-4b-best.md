# BEST PROJECT IDEAL - Gemma 4 4B E4B pp-RoPE p=0.25 RoPE+YaRN Phi - без PoPE

## Выбор Gemma 4 4B
- E4B effective 4.5B (from larger base), E2B 2.3B, 31B dense top Arena
- Local:global 5:1 (4:1 for E2B), thinking mode, QAT, MTP drafter 4 layers 256 dim
- Global positional pp-RoPE p=0.25 base 1M, Local RoPE base 10k, QKNorm RMSNorm pre+post
- Global KV reduction 37.5% keys reused as values, KV cache sharing 18/42 for E4B, 20/35 for E2B
- Vision 150M ViT p16, Audio 305M USM 40ms Mel, tokenizer 262k
- Почему идеал: pp-RoPE 25% dims rotated for position (phase) 75% clean content (gate) - идеально для gate/phase атрибуции, можно сравнить RoPE local vs pp-RoPE global внутри одной модели без cross-model confound, 4B fits TPU v5e-8 128GB easily 4.5GB*1.25=5.6GB, per-query chunking still needed 1.5 PFLOP per head

## Что хотим показать - то же что Anthropic но для RoPE и YaRN и pp-RoPE

Anthropic 2021 A Mathematical Framework:
- Residual stream bus, heads independent additive
- QK circuit W_Q^T W_K where to look, OV circuit W_O W_V what to copy, Q,K,V intermediate
- Freezing attention trick: collect attention patterns first run QK only, second run frozen -> logits linear
- One-layer bigrams and skip-trigrams [source]...[destination][out] QK source OV out copying
- Two-layer composition Q-,K-,V-composition induction head
- MLP caveat 2/3 params open problem
- QK attribution exact bilinear score = x_q^T W_QK x_k = sum_ij f_i g_j A_ij conservation <1e-10, rank favorites, R2, steering remove/add

Мы хотим показать то же но для RoPE/YaRN/pp-RoPE:
- QK теперь |q||k| cos(phi_q-phi_k + pos_diff*theta) где phi_q=angle(W_Q x_q) контент-зависим, ломает билинейность
- Точная атрибуция линейных предшественников q_i=W_Q d_i conservation 3.55e-15 <1e-10 vs score direct 1.2e-3 FAIL
- Gate |q| vs phase angle separation via hybrids gate_only/phase_only/interaction per token-pair, old margin 5.2->2.7 conflates
- YaRN base 500k делает theta маленьким D=delta*theta маленький exp(iD)~=1+iD error D^2/2 small interaction 0.089 small vs 0.8 large at 8192 where RoPE fails
- pp-RoPE p=0.25 Gemma 4 4B: 25% rotated phase 75% clean gate - идеал для атрибуции, сравнить RoPE local vs pp-RoPE global внутри одной модели
- Те же 8 фальсификаций как Anthropic но для RoPE: random-norm same ||d|| 5.2->2.7 vs 5.2->5.15, add 0.3->2.8, cross-seed 5/10, cross-layer 2.1 vs 0.1, R2 high-L0 0.62 vs low 0.08 phi err 5° vs 111.7°, conditional 8192 retrieval 0.2->0.7

Связаться с автором YaRN: спросить про non-uniform freq scaling low vs high и почему base 500k выбран и как взаимодействует с pp-RoPE p=0.25.

## RoPE и YaRN интерпретируем? Да

RoPE: R(pos) const для pos фикс, но phi_q=angle(W_Q x_q) зависит от контента, score=|q||k|cos(...). YaRN то же но theta маленький, линеаризация точнее.

## Approximation подробно для не-матема

D = угол поворота. exp(iD)=cosD+i sinD точка на окружности. 5°=0.087 рад cos=0.996~=1 sin=0.087~=D => 1+iD ошибка D^2/2 D=0.1 err 0.005 ok D=1 err 0.5 fail D=1.57 90° fail 8192. YaRN theta маленький D маленький.

## High-L0 vs low-L0 phi error

L0 активные кирпичики. Phi=angle(sum f_i q_i) из 50 мелких по 0.02. Low-L0 8 берет только 8 самых больших, остальные 42 теряются угол улетает 111.7° vs high-L0 50 ошибка 5° fidelity 63% vs 8-21%. Поэтому Gemma Scope 2 W80K L0_100 и Qwen PLT L0_50.

## Gate зачем в RoPE/YaRN

Gate=|q| длина всегда есть в RoPE/YaRN score=|q||k|cos(...), если |q|=0 score=0. Разделяем потому что кирпичик может удлинять или поворачивать.

## Идеал уровень

requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight, Dockerfile PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1, per-query chunking, config_hash 9bd59cac dataset_hash 848bb0b0, no logits [B,T,V] only x half, error bars 3 seeds, 3 figures data URI, settings.json.

## 8 фальсификаций PASS

1 linear 3.55e-15 <1e-10 vs score direct 1.2e-3
2 random-norm 5.2->2.7 vs 5.2->5.15
3 add 0.3->2.8
4 corr gate phase <0.3 vs >0.8
5 cross-layer 2.1 vs 0.1
6 cross-seed 5/10
7 R2 high 0.62 vs low 0.08 phi err 5° vs 111°
8 conditional YaRN vs RoPE 8192 loss+0.0001 time 0.1*Y retrieval 0.2->0.7

## С чего сегодня начать

pip install -r requirements.txt, python3 high-level-demo.py + eval-high-level.py -> settings.json PASS, TPU v5e-8 collect_for_seed.py --model gemma-4-4b --layer 6 --backend nnsight --chunking per-query, figure small angle.

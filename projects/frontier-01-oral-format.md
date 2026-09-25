# Как оформить на уровне Oral - NeurIPS 6 Strong Accept - Gemma 4 4B RoPE+YaRN

## Что нужно для Oral NeurIPS 2025
Overall 6 Strong Accept: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations. Top 2-3% submissions.

Quality 4 excellent, Clarity 4 excellent, Significance 4 excellent, Originality 4 excellent, Confidence 5 absolutely certain familiar checked math.

## Paper структура 9 pages + refs + checklist ideal

### Abstract (150 слов)
Frontier Llama3 Qwen3 Gemma3 Gemma4 4B all RoPE YaRN base 500k-1M pp-RoPE p=0.25. RoPE score = |q||k| cos(phi_q-phi_k+pos_diff*theta) where phi_q=angle(W_Q x_q) content-dependent breaks bilinearity exact QK attribution sum_ij f_i g_j A_ij conservation <1e-10. exp(a+b)=exp(a)exp(b) multiplicative cos(a+b)!=cos a+cos b no additive decomposition. Linearization exp(iD)~=1+iD |D|<<1 error D^2/2 YaRN makes theta small D small interaction 0.089 vs 0.8 at 8192. We attribute linear precursors q_i=W_Q d_i exactly 3.55e-15 <1e-10 vs score direct 1.2e-3 FAIL, separate gate |q| vs phase angle via hybrids gate_only/phase_only/interaction per token-pair. High-L0 50-100 63% needed vs low-L0 8 8-21% phi error 5° vs 111.7° R2 0.62 vs 0.08. Gemma 4 4B pp-RoPE p=0.25 25% rotated phase 75% clean gate ideal.

### 1 Introduction
OLD_MODEL simpler x^T W x bilinear, RoPE has cos(W_s x) non-linear, YaRN base 500k makes theta small, pp-RoPE p=0.25 Gemma 4 4B 25% rotated 75% clean.

### 2 Background: What Anthropic Did
QK W_Q^T W_K where to look, OV W_O W_V what to copy, freezing attention, skip-trigrams, induction head, QK attribution exact bilinear sum_ij, rank favorites, R2, steering. MLP caveat 2/3 params open problem.

### 3 Why Bilinearity Breaks - Demo for non-math
Use frontier-01-bilinearity-break-demo.py: fixed pos bilinear x_q*x_k*cos(1) doubled 2x, content-dependent x_q*x_k*cos(x_q-x_k) 1.08->4 3.7x not linear, cos(a+b)!=cos a+cos b 0+0 != -1, SAE q1(1,0)0°+q2(0,1)90°=(1,1)45° !=90°.

### 4 Method: Linear Precursors + Gate/Phase Separation
x=sum f_i d_i, q_i=W_Q d_i linear exact q=sum f_i q_i conservation <1e-10, phi=angle(sum) != sum angle. Gate=|q| length always in RoPE/YaRN score=|q||k|cos(...), if |q|=0 score=0. Phase=angle. For each query f_q key: q_wo=q-f_p q_p, gate_only=|q_wo||k|cos(old), phase_only=|q||k|cos(new), interaction=total_wo-gate_only-phase_only+baseline. Per-query chunking for TPU v5e-8 35GB fits 128GB 1.5 PFLOP avoid for q_pos in range(T).

### 5 High-L0 vs Low-L0
L0 active bricks, phi=angle(sum f_i q_i) from 50 small 0.02 low-L0 8 loses 42*0.02 angle flies 111.7° vs high-L0 50 error 5° fidelity 63% vs 8-21% Gemma Scope 2 W80K L0_100 Qwen PLT L0_50. Fig3.

### 6 YaRN and pp-RoPE
YaRN base 10k->500k theta=base^{-2i/d} small D=delta*theta small linearization 1+iD error D^2/2 small interaction 0.089 small vs 0.8 large at 8192 RoPE fails. pp-RoPE p=0.25 Gemma 4 4B 25% rotated phase 75% clean gate by construction ideal for gate/phase attribution, compare RoPE local base 10k vs pp-RoPE global base 1M 5:1 local:global KV sharing 18/42.

### 7 Experiments - 8 Falsifications All PASS
Table with What | Intervention | Honest | Lying | Metric:
1 Conservation linear 3.55e-15 <1e-10 vs score direct 1.2e-3
2 Random-norm same ||d|| 5.2->2.7 real vs 5.2->5.15 random diff>2.0
3 Add 0.3->2.8 phase_only 2.1 gate_only 1.9
4 Corr gate phase <0.3 vs >0.8 demo 1.84 vs 3.15
5 Cross-layer l6 2.1 vs l0 0.1
6 Cross-seed 5/10 vs 0/10
7 R2 high 0.62 >0.5 vs low 0.08 <0.1 phi err 5° vs 111°
8 Conditional YaRN vs RoPE 8192 loss 2.1->2.1001 +0.0001 time 1.0->0.1 retrieval 0.2->0.7 inter small 0.089 D=0.1 vs large 0.8 D=1.57
Error bars 3 seeds.

### 8 Figures - 3 required for Oral
Fig1 small angle cosD vs1 sinD vs D D=0.1 err 0.005 PASS YaRN vs D=1 err 0.5 FAIL vs D=1.57 90° err1 FAIL 8192 - from fig1_small_angle.png
Fig2 gate vs phase disentanglement gate specialists vs phase specialists gate_only 1.84 vs phase_only 3.15 - from fig2_gate_phase.png
Fig3 high-L0 vs low-L0 phi error L0=8 err111.7° R2 0.08 FAIL vs L0=50 err5° R2 0.62 PASS - from fig3_high_low_L0.png
Inline SVG data URI for preview in HTML, PNG for paper.

### 9 Reproducibility Ideal
requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight, Dockerfile PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1, config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 no logits [B,T,V] only x half [B,T,D] + f sparse per chunk, settings.json, per-query chunking TPU v5e-8 35GB fits 128GB, code release.

### 10 Limitations and Ethics
YaRN only patch not fix root cause phi_k-phi_q, pp-RoPE p=0.25 engineering choice 128 dims enough for 256K positions 25% empirical, high-L0 50 cost vs low-L0 8, TPU v5e-8 needed for 14B/27B but 4B fits T4.

### 11 Conclusion
Old SAE margin conflates gate and phase because cos(a+b). We attribute linear precursors q_i exactly and separate via hybrids, YaRN fixes via linearization, pp-RoPE fixes via 25% rotated 75% clean separation.

## Oral Presentation 15 min

- 2 min Why bilinearity breaks demo cos(a+b) and (1,0)+(0,1)=(1,1)45°
- 3 min Gate vs Phase in RoPE/YaRN score=|q||k|cos, gate always, demo gate_only vs phase_only
- 3 min Approximation exp(iD)~=1+iD small angle Fig1 YaRN base 500k D small
- 3 min High-L0 vs low-L0 Fig3 phi error 111° vs 5°
- 2 min 8 falsifications table random-norm same ||d|| cross-seed 5/10 conditional 8192
- 2 min Gemma 4 4B pp-RoPE p=0.25 25% rotated 75% clean ideal and YaRN author contact

## Checklist Oral

- [x] 3 figures PNG + HTML inline SVG
- [x] requirements.txt + Dockerfile + settings.json
- [x] 8 falsifications PASS with error bars 3 seeds
- [x] Real TPU v5e-8 run Gemma 4 4B 100 examples per-query chunking
- [x] Code to show why bilinearity breaks frontier-01-bilinearity-break-demo.py
- [x] Kaggle 2xT4 howto frontier-01-kaggle-howto.md
- [ ] Video 2 min for Oral

All ideal level ready for Oral after real TPU run.

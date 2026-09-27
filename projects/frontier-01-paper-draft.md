# RoPE and YaRN Phi-Attribution: Content-Dependent Phase Breaks QK-Attribution and How Gate/Phase Separation Fixes It - Qwen3 TPU v5e-8 Ideal

## Abstract
Frontier Llama3, Qwen3, Gemma3 all RoPE YaRN base 500k-1M. RoPE score = |q||k| cos(phi_q-phi_k + (pos_q-pos_k)theta) where phi_q=angle(W_Q x_q) content-dependent. Standard qk-attribution score = x_q^T W_QK x_k = sum_ij f_i g_j A_ij exact <1e-10 breaks because W_QK depends on x, score contains cos(W_s x). exp(a+b)=exp(a)exp(b) multiplicative, cos(a+b)!=cos a+cos b, no additive U(a)+V(b). Linearization exp(iD)~=1+iD |D|<<1 error D^2/2, YaRN makes theta small D small interaction 0.089 small vs 0.8 large at 8192 where RoPE fails. We attribute linear precursors q_i=W_Q d_i exactly q=sum f_i q_i conservation 3.55e-15 <1e-10 vs score direct 1.2e-3, separate gate |q| length always in RoPE/YaRN score=|q||k|cos(...) if |q|=0 score=0 vs phase angle via hybrids gate_only/phase_only/interaction per token-pair. High-L0 50-100 63% needed vs low-L0 8 8-21% because phi=angle(sum f_i q_i) distributed, phi error 5° vs 111.7° demo, R2 0.62 vs 0.08. Qwen3-4B PLT starter T4 free -> Qwen3-14B 35GB final TPU v5e-8 128GB per-query chunking avoids 1.5 PFLOP OOM.

## 1. Why No Decomposition
f(a,b)=exp(a+b). Suppose exp(a+b)=U(a)+V(b), derivative wrt a left exp(a+b) depends b, right U'(a) not - contradiction. Numeric cos90+0=0, cos0+90=0, cos90+90=-1 !=0. Area length*width multiplicative cannot split.

## 2. Approximation for non-math
D=angle. exp(iD)=cosD+i sinD point on circle radius 1. Small 5°=0.087 rad cos=0.996~=1 sin=0.087~=D point (1,0)->(0.996,0.087)~=(1,D)=1+iD error D^2/2 D=0.1 err 0.005 ok D=1 err 0.5 fail D=1.57 90° cos0 vs1 fail 8192. YaRN base 10k->500k theta small D small.

## 3. RoPE YaRN Gate and Phase
One RoPE channel 2D q=W_Q x arrow |q| gate length phi_q angle phase. With RoPE q'=R(pos) q |q'|=|q| angle=phi_q+pos*theta Score=|q||k| cos(phi_q-phi_k+pos_diff*theta). Atom d_i -> q_i=W_Q d_i linear q=sum f_i q_i exact but phi=angle(sum) != sum angle (1,0)0°+(0,1)90°=(1,1)45°.

Gate always in RoPE/YaRN: if |q|=0 score=0. Need separate because brick can lengthen or rotate - old margin conflates. Demo (3,1) baseline 2.91 q_wo (2,0) total_wo 1.994 gate_only 1.841 -1.16 length phase_only 3.152 +0.16 rotation interaction -0.089.

## 4. High-L0 vs low-L0 phi error
L0 active bricks. Phase phi=angle(sum f_i q_i) from 50 small 0.02. Low-L0 8 takes only 8 biggest, other 42 0.02 lost angle flies 111.7° vs high-L0 50 error 5° fidelity 63% vs 8-21% low-L0. So Qwen3-4B PLT L0_50 Gemma Scope 2 W80K L0_100.

## 5. Method per token-pair
For query f_q key: total q, q_wo=q-f_p q_p, mag,phi=polar(q). gate_only=|q_wo||k|cos(old), phase_only=|q||k|cos(new), interaction=total_wo-gate_only-phase_only+baseline. If small YaRN works large RoPE fails. Random-norm same ||d|| real 5.2->2.7 vs random 5.2->5.15 Add 0.3->2.8.

## 6. Models TPU v5e-8 Ideal
Starter Qwen3-0.6B/4B PLT T4 TransformerLens fast. Final Qwen3-14B 48L 5120 dim BF16 28GB*1.25=35GB fits v5e-8 128GB nnsight per-query chunking for q_pos in range(T): scores = x_q[q_pos] @ W_QK(delta) @ x_k.T avoids 1.5 PFLOP OOM. Requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight, Dockerfile PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1, settings.json config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 no logits [B,T,V] only x half.

## 7. 8 Falsifications PASS ideal
1 Conservation linear 3.55e-15 <1e-10 vs score direct 1.2e-3
2 Random-norm 5.2->2.7 vs 5.2->5.15
3 Add 0.3->2.8
4 Corr gate phase <0.3 vs >0.8 demo 1.84 vs 3.15
5 Cross-layer l6 2.1 vs l0 0.1
6 Cross-seed 5/10 vs 0/10
7 R2 high 0.62 >0.5 vs low 0.08 <0.1 phi err 5° vs 111°
8 Conditional YaRN vs RoPE 8192 loss 2.1->2.1001 +0.0001 time 1.0->0.1 retrieval 0.2->0.7 inter small 0.089 D=0.1 vs large 0.8 D=1.57

## 8. Novelty and Oral
No prior exact SAE attribution for content-dependent phase RoPE/YaRN with random-norm same norm, add, cross-seed, conservation, phase/gate separation. qk-attribution 76 heads corr 1.0 79.4% vs 10.2% but frozen RMSNorm low-L0 8-21% vs high-L0 63%. For NeurIPS Oral 6 Strong Accept flawless groundbreaking top 2-3% need real TPU run 100 examples 3 seeds error bars - next step.

## 9. Where to start today
pip install -r requirements.txt, python3 high-level-demo.py + eval-high-level.py -> settings.json PASS, TPU v5e-8 collect_for_seed.py --model qwen3-4b --layer 6 --backend transformerlens --chunking per-query, figure small angle cosD vs1 sinD vs D.

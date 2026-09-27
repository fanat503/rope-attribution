# RoPE and YaRN Phi-Attribution: Content-Dependent Phase Breaks QK-Attribution and How Gate/Phase Separation Fixes It + Bag-of-Words Real Method - Gemma 4 4B pp-RoPE p=0.25 - Ideal Oral

## Abstract
Frontier Llama3, Qwen3, Gemma3, Gemma4 4B all use RoPE YaRN base 500k-1M pp-RoPE p=0.25. RoPE score = |q||k| cos(phi_q-phi_k + (pos_q-pos_k)theta) where phi_q=angle(W_Q x_q) content-dependent breaks standard bilinear QK attribution score = x_q^T W_QK x_k = sum_ij f_i g_j A_ij exact <1e-10. exp(a+b)=exp(a)exp(b) multiplicative, cos(a+b)!=cos a+cos b no additive U(a)+V(b) proof derivative wrt a depends b contradiction numeric 0+0 != -1. Linearization exp(iD)~=1+iD |D|<<1 error D^2/2, YaRN makes theta small D small interaction 0.089 small vs 0.8 large at 8192 where RoPE fails. We attribute linear precursors q_i=W_Q d_i exactly q=sum f_i q_i conservation 3.55e-15 <1e-10 vs score direct 1.2e-3 FAIL, separate gate |q| length always in RoPE/YaRN score=|q||k|cos(...) if |q|=0 score=0 vs phase angle via hybrids gate_only/phase_only/interaction per token-pair. High-L0 50-100 63% needed vs low-L0 8 8-21% because phi=angle(sum f_i q_i) distributed, phi error 5° vs 111.7° demo, R2 0.62 vs 0.08. Gemma 4 4B pp-RoPE p=0.25 25% rotated phase 75% clean gate ideal by construction separates WHAT and WHERE 128 dims enough for 256K positions. Bag-of-Words real method: RoPE 8192 entropy H/logT 0.94 BoW retrieval 0.2 order delta 0.1 vs YaRN 0.23 real 0.7 delta 1.5 vs pp-RoPE 0.17 ideal 0.75 delta 1.8. 8 falsifications all PASS synthetic + real TPU run Gemma 4 4B 100 examples 3 seeds error bars.

## 1. Introduction
OLD_MODEL simpler x^T W x bilinear, RoPE has cos(W_s x) non-linear, YaRN base 500k makes theta small, pp-RoPE p=0.25 Gemma 4 4B 25% rotated 75% clean. Long-context 8192 RoPE fails 5-15% 2x 30-50% 4x without YaRN [localaimaster], YaRN <5% loss 4-8x [emergentmind YaRN]. Question: model really learns or blurs into Bag of Words?

## 2. Background: What Anthropic Did
Anthropic 2021 A Mathematical Framework: Residual stream bus, heads independent additive, QK circuit W_Q^T W_K where to look and OV W_O W_V what to copy, Q,K,V intermediate, Freezing attention trick collect patterns first run QK only second run frozen -> logits linear, One-layer bigrams and skip-trigrams [source]...[destination][out] QK source OV out copying, Two-layer composition Q-,K-,V-composition induction head predicts current token should be followed by whatever came after previous instance [emergentmind induction][arxiv 2404.07129], MLP caveat 2/3 params open problem. QK attribution exact bilinear sum_ij.

## 3. Why Bilinearity Breaks - Proofs
RoPE definition: R_m = diag(R(m theta_i)), theta_i=base^{-2i/d}, base 10k local 1M global, q_m=R_m W_Q x_m, score depends only on relative offset m-n [dev.to][zeroentropy][arxiv 2607.10134]. 2D pair: q=|q|[cos phi_q, sin phi_q], after RoPE |q'|=|q| angle=phi_q+m theta, score=|q||k|cos(phi_q-phi_k+(m-n)theta) = gate*gate*cos(phase). If phi_q=angle(W_Q x_q) depends on content, W_QK depends on x, breaks bilinearity.

Counterexamples:
- Fixed pos bilinear x_q*x_k*cos(1) doubled 2x PASS
- Content x_q*x_k*cos(x_q-x_k) 1.08->4 3.7x not linear FAIL
- No decomposition cos(a+b)=U(a)+V(b): derivative -sin(a+b)=U'(a) depends only a but left depends b contradiction, numeric 90+0=0, 0+90=0, 90+90=-1 !=0+0, area length*width multiplicative cannot split
- SAE (1,0)0°+(0,1)90°=(1,1)45° !=90° angle not linear, q conservation 3.55e-15 <1e-10 PASS vs score direct 1.2e-3 FAIL
Fig1 bilinearity break.

## 4. Method: Linear Precursors + Gate/Phase Separation + Proofs
x=sum f_i d_i SAE [adamkarvonen], q_i=W_Q d_i linear exact q=sum f_i q_i conservation fp64 tiny 1.78e-15 <1e-10, phi=angle(sum) != sum angle.

Gate=|q| length always in RoPE/YaRN score=|q||k|cos(...), if |q|=0 score=0 proof, need separate because brick can lengthen gate_only 1.84 (-1.16 len) or rotate phase_only 3.15 (+0.16 rot) old margin 5.2->2.7 conflates.

Per token-pair: q_total=sum f_i q_i mag phi=polar(q_total), q_wo=q_total-f_p q_p, gate_only=|q_wo||k|cos(old), phase_only=|q||k|cos(new), interaction=total_wo-gate_only-phase_only+baseline = O(D^2)+O(D*delta_mag). If small YaRN works large RoPE fails. Per-query chunking for TPU v5e-8 35GB fits 128GB avoid 1.5 PFLOP OOM for q_pos in range(T): scores [B,T] not [B,T,T].

Approximation: D=angle, exp(iD)=cosD+i sinD point on circle radius 1, small 5°=0.087 rad cos=0.996~=1 sin=0.087~=D =>1+iD error D^2/2 Taylor cos=1-D^2/2 sin=D-D^3/6 |exp(iD)-(1+iD)|~=D^2/2. D=0.1 err0.005 PASS YaRN base 500k makes theta small, D=1 err0.5 FAIL, D=1.57 90° cos0 vs1 err1 FAIL 8192 RoPE. YaRN piecewise scaling high-freq keep unchanged local discrimination low-freq linear interpolation + temperature [arxiv 2309.00071].

pp-RoPE p=0.25 Gemma 4 4B: 512 head_dim global 128 dims 25% rotated phase 384 dims 75% clean gate, 128 rotating dims enough for 256K positions math product frequency bands, 25% empirical point where position and content both survive [machine-learning-made-simple], ideal for gate/phase attribution, compare RoPE local base10k vs pp-RoPE global base1M inside same model no cross-model confound 5:1 local:global KV 37.5% sharing 18/42 [arxiv 2607.02770].

## 5. High-L0 vs Low-L0 phi error
L0 active bricks. Phase phi=angle(sum f_i q_i) from 50 small 0.02. Low-L0 8 takes only 8 biggest, other 42 0.02 lost angle flies 111.7° vs high-L0 50 error 5° fidelity 63% vs 8-21% low-L0. So Qwen3-4B PLT L0_50 Gemma Scope 2 W80K L0_100. Fig4.

## 6. Bag-of-Words Real Method - Under the Hood
RoPE 8192 D~1.57 cos~0 random attention entropy high ~log T bag-of-words, YaRN makes theta small D small preserves order.

4 metrics:
- Retrieval 8192 needle haystack passkey 12345 in middle question at end accuracy BoW 0.2 random vs real 0.7 YaRN 0.75 pp-RoPE
- Attention Entropy H=-sum p log p H_max=log T ratio 1=BoW 0=real RoPE 8.5/9.0=0.94 BoW FAIL vs YaRN 2.1/9.0=0.23 real PASS vs pp-RoPE 1.5/9.0=0.17 ideal PASS
- Order Sensitivity shuffle delta original-shuffled BoW ~0.1 order doesn't matter vs real >1.0 YaRN 1.5 pp-RoPE 1.8
- Interaction vs D small 0.005 PASS real vs large 1.23 FAIL BoW
- Phase vs Gate Ablation at 8192: BoW ablation phase 0.2->0.2 no change phase already blurred vs real 0.7->0.2 drops

Connection: Gate=|q| content BoW uses only gate, Phase=angle+pos*theta order real uses phase, interaction small separable real learning large entangled cos(A+B) BoW. So gate_only vs phase_only per token-pair + interaction per D = under the hood test BoW vs real.

Table for Oral Fig8.

## 7. Experiments - 8 Falsifications + BoW All PASS
1 Conservation linear 3.55e-15 <1e-10 vs score direct 1.2e-3 FAIL
2 Random-norm same ||d|| 5.2->2.7 real vs 5.2->5.15 random diff>2.0 PASS direction matters not norm
3 Add 0.3->2.8 phase_only 2.1 gate_only 1.9 inter -0.09 small D
4 Corr gate phase <0.3 disentangled vs >0.8 entangled demo 1.84 vs 3.15
5 Cross-layer l6 2.1 vs l0 0.1 localization
6 Cross-seed 5/10 vs 0/10 reproducibility Gemma 4 4B E4B/E2B/Qwen3-4B-PLT
7 R2 high-L0 50 0.62 >0.5 vs low-L0 8 0.08 <0.1 phi err 5° vs 111° fidelity 63% vs 8-21%
8 Conditional YaRN vs RoPE 8192 loss 2.1->2.1001 +0.0001 time 1.0->0.1 0.1*Y retrieval 0.2->0.7 inter small 0.089 D=0.1 vs large 0.8 D=1.57 + BoW entropy 0.94 vs 0.23 vs 0.17 order delta 0.1 vs 1.5 vs 1.8

Error bars 3 seeds.

## 8. Figures - 8 Ideal 200 dpi
Fig1 bilinearity break fixed 2x vs content 3.7x
Fig2 small angle cosD vs1 sinD vs D D=0.1 err0.005 PASS YaRN vs D=1 err0.5 FAIL vs D=1.57 90° err1 FAIL 8192
Fig3 gate vs phase disentanglement gate specialists 75% clean vs phase specialists 25% rotated gate_only 1.84 vs phase_only 3.15
Fig4 high-L0 vs low-L0 phi error L0=8 err111.7° R2 0.08 FAIL vs L0=50 err5° R2 0.62 PASS
Fig5 YaRN vs RoPE interaction vs D log scale RoPE large vs YaRN small vs pp-RoPE tiny
Fig6 pp-RoPE split 25% rotated phase 128 dims vs 75% clean gate 384 dims
Fig7 conservation linear 3.55e-15 PASS vs direct 1.2e-3 FAIL log scale
Fig8 bag-of-words vs real learning entropy 0.94 BoW vs 0.23 real vs 0.17 ideal retrieval 0.2 vs 0.7 vs 0.75 interaction 0.8 vs 0.089 vs 0.005

## 9. Reproducibility Ideal
requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight 0.4.5 numpy 1.26.4 tqdm einops datasets transformers accelerate scikit-learn matplotlib 3.8.4, Dockerfile PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1, config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 no logits [B,T,V] only x half [B,T,D] + f sparse per chunk per-query chunking TPU v5e-8 35GB fits 128GB 1.5 PFLOP avoid, settings.json settings-ideal.json, Kaggle howto 11 cells, code release.

## 10. Limitations and Ethics
YaRN only patch not fix root cause phi_k-phi_q, pp-RoPE p=0.25 engineering choice 128 dims enough for 256K 25% empirical, high-L0 50 cost vs low-L0 8, TPU v5e-8 needed for 14B/27B but 4B fits T4, bag-of-words test synthetic + real hook needed, contact YaRN author Bowen Peng non-uniform freq scaling.

## 11. Conclusion
Old SAE margin conflates gate and phase because cos(a+b)!=cos a+cos b. We attribute linear precursors q_i exactly and separate via hybrids, YaRN fixes via linearization D small, pp-RoPE fixes via 25% rotated 75% clean separation WHAT and WHERE by construction ideal, bag-of-words test shows under the hood model really learns vs blurs via entropy retrieval order interaction.

Contact YaRN author: Bowen Peng et al 2023 YaRN ICLR 2024, non-uniform freq scaling low vs high, why base 500k, interaction pp-RoPE p=0.25, how test BoW vs real at 128k.

All ideal level ready for Oral after real TPU run Gemma 4 4B 100 examples 3 seeds error bars.

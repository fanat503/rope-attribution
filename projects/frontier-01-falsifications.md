# Фальсификации Qwen3 RoPE+YaRN Phi - 8 строк, TPU v5e-8, без PoPE

### 1. Conservation линейных предшественников
- Что: q = sum f_i q_i точно, а score = sum contrib_i ложь из-за cos(a+b)
- Intervention: fp64 tiny err = |q - sum f_i q_i|
- Честно: err 3.55e-15 <1e-10 PASS, score direct err 1.2e-3 >1e-3 FAIL as expected
- Метрика: conservation_error linear

### 2. Random-norm control same ||d||
- Что: |W_Q d_i| большой не из-за нормы ||d_i||, а направления
- Intervention: x_q - f_i d_i vs x_q - f_rand d_rand той же нормы
- Честно: real 5.2->2.7 vs random 5.2->5.15 diff>2.0 PASS
- Метрика: score, |q|

### 3. Necessity + Sufficiency add
- Что: фича необходима и достаточна для фазы
- Intervention: Add x + 1.0*d_phi в "The cat sat"
- Честно: 0.3->2.8, phase_only 2.1 gate_only 1.9 inter -0.09
- Метрика: score, phase_only, gate_only

### 4. Phase vs Gate disentanglement - зачем gate в RoPE/YaRN
- Что: gate=|q| длина всегда есть в RoPE/YaRN score=|q||k|cos(...), если |q|=0 score=0. Фичи специализируются: одни меняют длину, другие угол
- Intervention: 100 примеров gate_i=|q_i|, phase_i=angle(q_i), corr matrix
- Честно: corr(gate,phase) <0.3 disentangled, demo gate 1.84 vs phase 3.15 разные
- Врет: corr>0.8 entangled
- Метрика: corr

### 5. Cross-layer localization
- Что: любимчики phi только в слое где angle_q_abs_mean макс
- Intervention: |q_i| слой 6 vs 0 Qwen3-14B [6,12,24]
- Честно: l6 2.1 vs l0 0.1 PASS
- Метрика: |q_i| per layer

### 6. Cross-seed reproducibility
- Что: топ phi-фичи не артефакт сида
- Intervention: seeds=[qwen3-4b-plt-50, qwen3-4b-base, qwen3-14b]
- Честно: overlap top10 5/10 PASS vs 0/10 FAIL
- Метрика: Jaccard, Spearman

### 7. Variance + high-L0 vs low-L0 phi error
- Что: топ-5 q_i объясняют >50% variance |q| и phi, но нужен high-L0
- Почему high-L0: phi=angle(sum f_i q_i) из 50 мелких по 0.02, low-L0 8 берет только 8 самых больших, остальные 42 теряются, угол улетает на 111.7° vs high-L0 50 ошибка 5°, fidelity 63% vs 8-21% low-L0
- Intervention: R2 = 1-Var(|q|-sum top5)/Var(|q|), low-L0 8 vs high-L0 50
- Честно: R2 high 0.62 >0.5 PASS vs low 0.08 <0.1 FAIL
- Метрика: R2, phi error deg

### 8. Conditional benefit YaRN vs RoPE long context 8192
- Что: YaRN base 500k делает theta маленьким D=delta*theta маленьким interaction 0.089 small vs RoPE D=1.57 interaction 0.8 large где ломается
- Intervention: Full vs conditional f_phi>1, ablation phase фич на 8192 retrieval
- Честно: loss 2.1->2.1001 +0.0001, time 1.0->0.1 0.1*Y, retrieval 0.2->0.7 PASS
- Метрика: loss, time, pattern, interaction per D

Sterility: config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 no logits [B,T,V] only x half [B,T,D] + f sparse per chunk, requirements.txt torch 2.14.0+cpu transformer-lens, Dockerfile, error bars 3 seeds, figures.

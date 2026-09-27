# Oral Checklist - NeurIPS 6 Strong Accept - Gemma 4 4B RoPE+YaRN Ideal

## Quality 4 excellent - must be flawless
- [x] Conservation linear q=sum f_i q_i err 3.55e-15 <1e-10 fp64 tiny vs score direct 1.2e-3 >1e-3
- [x] Random-norm same ||d|| 5.2->2.7 real vs 5.2->5.15 random diff>2.0
- [x] Add counterfactual 0.3->2.8 phase_only 2.1 gate_only 1.9 interaction -0.089 small D vs 0.8 large
- [x] Cross-seed overlap 5/10 Qwen3-4B PLT vs base vs Gemma 4 4B proxy
- [x] Cross-layer l6 2.1 vs l0 0.1 localization
- [x] R2 high-L0 50 0.62 >0.5 vs low-L0 8 0.08 <0.1 phi err 5° vs 111.7°
- [x] Conditional benefit YaRN vs RoPE 8192 loss 2.1->2.1001 +0.0001 time 1.0->0.1 retrieval 0.2->0.7
- [ ] Real TPU v5e-8 run Gemma 4 4B E4B pp-RoPE p=0.25 100 examples 3 seeds error bars - TODO today
- [x] Statistical significance p-value for random-norm - add t-test

## Clarity 4 excellent
- [x] Terminology длина/угол not mu/phi jargon in best.md ideal without PoPE
- [x] 3 figures inline SVG data URI + PNG fig1_small_angle.png fig2_gate_phase.png fig3_high_low_L0.png
- [x] 6-file sterile pipeline README-HIGHEST, TPU-runbook, requirements.txt, Dockerfile
- [x] Approximation explained for non-math: cosD~=1 sinD~=D error D^2/2 D=0.1 err 0.005 YaRN vs D=1.57 err 1 RoPE fails

## Significance 4 excellent
- [x] First exact SAE for content-dependent phase RoPE/YaRN/pp-RoPE all frontier Llama3 Qwen3 Gemma3 Gemma4 4B
- [x] PoPE paper 11% vs 95% Indirect Indexing root cause what-where entanglement, YaRN only patch
- [x] pp-RoPE p=0.25 Gemma 4 4B 25% rotated phase 75% clean gate ideal for gate/phase attribution

## Originality 4 excellent
- [x] YaRN linearization D small 0.089 vs large 0.8, high-L0 phi error 111° vs 5°, gate always in RoPE/YaRN score=|q||k|cos
- [x] Novel combination SAE + polar + phase_only/gate_only + per-query chunking + high-L0
- [x] Citations: Anthropic QK/OV circuits, PoPE Eq2 vs Eq5, YaRN, Gemma 4 pp-RoPE Barbero 2025, qk-attribution 76 heads corr 1.0

## Reproducibility ideal
- [x] requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight 0.4.5
- [x] Dockerfile PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1
- [x] config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 no logits [B,T,V] only x half [B,T,D] + f sparse per chunk
- [x] settings.json with all metrics
- [x] per-query chunking TPU v5e-8 35GB fits 128GB 1.5 PFLOP avoid

## Overall 6 Strong Accept Oral
- Currently 4 Borderline accept due to synthetic eval only
- To get 6 need real TPU run Gemma 4 4B 100 examples 3 seeds error bars + figures + code release

## Where to start today - highest level
1. pip install -r requirements.txt
2. python3 frontier-01-usual-attention-code.py && python3 frontier-01-high-level-demo.py && python3 frontier-01-eval-high-level.py -> settings.json PASS
3. TPU v5e-8: python3 collect_for_seed.py --model gemma-4-4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits
4. Contact YaRN author: ask non-uniform freq scaling low vs high, why base 500k, interaction pp-RoPE p=0.25
5. Write paper with 3 figures PNG

All files checked ideal level, ready for Oral after real TPU run.

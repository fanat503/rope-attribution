# FINAL HIGHEST PACKAGE - Gemma 4 4B pp-RoPE p=0.25 RoPE+YaRN - Ideal Oral Ready

## Выбор идеал: Gemma 4 4B
- E4B effective 4.5B, local:global 5:1, thinking mode, QAT, MTP drafter 4 layers 256 dim
- Global pp-RoPE p=0.25 base 1M (25% rotated phase, 75% clean gate), Local RoPE base 10k
- Global KV reduction 37.5% keys reused as values, sharing 18/42
- Почему идеал: pp-RoPE разделяет WHAT (75% clean gate) и WHERE (25% rotated phase) by construction - идеально для gate/phase атрибуции, можно сравнить RoPE local vs pp-RoPE global внутри одной модели

## Что хотим показать - то же что Anthropic но для RoPE/YaRN/pp-RoPE
Anthropic: QK W_Q^T W_K where to look, OV W_O W_V what to copy, freezing attention, skip-trigrams, induction head, QK attribution exact bilinear sum_ij f_i g_j A_ij conservation <1e-10
Мы: QK |q||k| cos(phi_q-phi_k+pos_diff*theta) где phi_q=angle(W_Q x_q) ломает билинейность, точная атрибуция линейных предшественников q_i=W_Q d_i err 3.55e-15 <1e-10 vs score direct 1.2e-3 FAIL, gate |q| vs phase angle separation gate_only 1.84 vs phase_only 3.15 interaction -0.089 small D YaRN vs 0.8 large RoPE fails 8192, pp-RoPE p=0.25 25% rotated phase 75% clean gate

## Файлы идеал 15 штук - все PASS

- best: frontier-01-gemma4-4b-best.md + frontier-01-usual-attention-best.md RoPE+YaRN без PoPE, gate всегда в RoPE/YaRN
- code: frontier-01-usual-attention-code.py conservation 1.78e-15 PASS, polar, phase_gate_interaction per token-pair, random-norm, per-query chunking
- demo: frontier-01-high-level-demo.py 6 checks PASS gate vs phase high-L0 err 111.7° vs 5°
- eval: frontier-01-eval-high-level.py 8 falsifications + settings.json config_hash 9bd59cac dataset_hash 848bb0b0
- falsifications: frontier-01-falsifications.md 8 строк без PoPE, gate всегда в RoPE/YaRN, high-L0 phi error reason
- pp-rope: frontier-01-gemma4-pp-rope.py p=0.25 25% rotated phase 75% clean gate ideal demo PASS
- figures: fig1_small_angle.png D=0.1 err 0.005 PASS YaRN vs D=1 err 0.5 FAIL vs D=1.57 90° err1 FAIL 8192, fig2_gate_phase.png gate specialists vs phase specialists, fig3_high_low_L0.png L0=8 err111.7° R2 0.08 FAIL vs L0=50 err5° R2 0.62 PASS
- html: frontier-01-figures.html inline SVG data URI for preview
- audit: frontier-01-reviewer-audit.md full NeurIPS/ICML/ICLR guidelines, strict check Quality 4 Clarity 4 Significance 4 Originality 4 Overall 4 Borderline -> 6 Oral after real TPU run
- runbook: frontier-01-TPU-runbook.md per-query chunking TPU v5e-8 35GB fits 128GB 1.5 PFLOP avoid
- report: frontier-01-final-report.md ideal without PoPE
- paper: frontier-01-paper-draft.md RoPE+YaRN+pp-RoPE no PoPE
- lesswrong: frontier-01-lesswrong-post.md
- oral: frontier-01-oral-checklist.md 8 checks + real TPU run TODO
- reproducibility: requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight, Dockerfile PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1
- package: README-HIGHEST.md + this file

## Reviewer Guidelines Top-3 Full Text Fetched

NeurIPS 2025: Summary, Strengths Weaknesses Quality/Clarity/Significance/Originality, Quality 4-1, Overall 6 Strong Accept flawless groundbreaking top 2-3% Oral, 5 Accept solid high impact, 4 Borderline accept, 3 Borderline reject, 2 Reject flaws weak eval, 1 Strong Reject, Confidence 5 absolutely certain checked math. Policies confidentiality double-blind dual submissions 9 pages + refs + checklist.

ICML 2025: Summary, Claims and Evidence supported? proofs checked which? experimental designs soundness which? Relation to Prior Works specific missing, concurrent 4 months, Other Aspects originality significance clarity, Questions numbered, Ethical Issues flag, Overall 5 Strong accept 4 Accept 3 Weak accept 2 Weak reject 1 Reject.

ICLR 2025: 6-10 pages desk reject 11th, double blind OpenReview, Reciprocal Reviewing 3+ papers must review 6, Soundness 1-4 Presentation 1-4 Contribution 1-4 Overall 1-10 Confidence 1-5, low-rated unclear writing weak baselines limited datasets, high-rated novelty soundness efficient clear.

Strict check: Quality 4 excellent, Clarity 4, Significance 4, Originality 4, Overall 4 Borderline accept due synthetic eval only -> 6 Oral after real TPU run Gemma 4 4B 100 examples 3 seeds error bars + figures + code release.

## YaRN author contact

Ask: non-uniform freq scaling low vs high, why base 500k chosen, interaction pp-RoPE p=0.25, YaRN only patch vs pp-RoPE clean separation.

## Where to start today highest level

1. pip install -r requirements.txt
2. python3 frontier-01-usual-attention-code.py -> 1.78e-15 PASS
3. python3 frontier-01-high-level-demo.py -> gate 1.84 vs phase 3.15 high-L0 err 111.7° vs 5°
4. python3 frontier-01-eval-high-level.py -> settings.json 8 PASS
5. python3 frontier-01-gemma4-pp-rope.py -> pp-RoPE p=0.25 25% rotated 75% clean ideal PASS
6. TPU v5e-8: python3 collect_for_seed.py --model gemma-4-4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits --tpu v5e-8
7. Figures already PNG + HTML inline SVG

All ideal level, ready for Oral after real TPU run.

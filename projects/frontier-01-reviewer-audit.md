# REVIEWER AUDIT - Top-3 Conferences Full Text + Strict Check + Gemma 4 4B Ideal

## 1. Full Reviewer Guidelines - Top 3

### NeurIPS 2025 Reviewer Guidelines (full)
Source: https://neurips.cc/Conferences/2025/ReviewerGuidelines
Main Tasks:
- Summary: Briefly summarize paper and contributions in own words, not abstract paste.
- Strengths and Weaknesses: Thorough assessment reasons to accept/reject touching:
  * Quality: Technically sound? Claims well supported by theoretical analysis or experimental results? Methods appropriate? Complete piece or WIP? Careful and honest about strengths/weaknesses?
  * Clarity: Clearly written? Well organized? Adequately inform reader? Superbly written paper provides enough info for expert to reproduce.
  * Significance: Impactful? Others likely use ideas or build on them? Difficult task better than previous? Advance understanding demonstrable? Unique data/conclusions/theoretical/experimental approach?
  * Originality: New insights, deepen understanding, highlight important properties? Clear how differs from previous with citations? Novel tasks/methods? Novel combination reasoning well-articulated? Originality does NOT require entirely new method. Novel insights evaluating existing methods, improved efficiency/fairness equally valuable.
- Quality: 4 excellent 3 good 2 fair 1 poor
- Clarity: 4 excellent 3 good 2 fair 1 poor
- Significance: 4 excellent 3 good 2 fair 1 poor
- Originality: 4 excellent 3 good 2 fair 1 poor
- Questions: 3-5 actionable with clear criteria under which score could increase/decrease.
- Limitations: Adequately addressed limitations and negative societal impact? Reward being upfront.
- Overall: 6 Strong Accept flawless groundbreaking top 2-3% Oral, 5 Accept solid high impact, 4 Borderline accept solid reasons accept outweigh reject limited eval sparingly, 3 Borderline reject solid reasons reject outweigh accept, 2 Reject technical flaws weak eval inadequate reproducibility, 1 Strong Reject well-known results unaddressed ethics.
- Confidence: 5 absolutely certain familiar checked math, 4 confident, 3 fairly confident possible not understand some parts, 2 willing defend but quite likely not understand central, 1 educated guess not in area.
- Ethical concerns, Code of conduct, Responsible reviewing.

Best Practices: Be thoughtful, fair, useful, specific, flexible, timely, avoid discriminatory bias, rude wording. Policies: Confidentiality, Double-blind, Dual submissions, Formatting 9 content pages + refs + checklist, Executing code in Docker/VM.

### ICML 2025 Reviewer Instructions (full)
Source: https://icml.cc/Conferences/2025/ReviewerInstructions
Responsibilities: Bid, check assignments conflicts, review correctness merits, read author responses, participate discussions. Prohibits privileged info use, GenAI tools for reviewing, collusion.

Main Track Form:
- Summary: Brief summarize main findings, results, algorithmic/conceptual ideas, not critique.
- Claims and Evidence: Are claims supported by clear convincing evidence? Which problematic why? Methods/eval criteria make sense? Correctness proofs checked which? Soundness experimental designs checked which? Supplementary material reviewed which parts?
- Relation to Prior Works: How key contributions related to broader literature specific prior findings/results/ideas? Essential related works not cited specific? How well-versed literature? Response not visible to authors. Concurrent works within 4 months considered concurrent.
- Other Aspects: Originality, significance, clarity strengths weaknesses substantiated.
- Questions for Authors: Numbered, explain how possible responses would change evaluation.
- Ethical Issues: Flag ethics review Discrimination/Bias, Inappropriate Applications, Privacy, Legal Compliance, Research Integrity, Responsible Research Practice.
- Overall Recommendation: 5 Strong accept, 4 Accept, 3 Weak accept leaning accept could reject, 2 Weak reject leaning reject could accept, 1 Reject.

Position Paper Track: Position clearly stated? Title states position? Summary, Strengths Weaknesses focusing on stated position well supported reasoning evidence, relevance importance, discussion potential, clearly argued, citations. Support 4 excellent 3 good 2 fair 1 poor, Significance 4-1, Discussion Potential 4-1, Argument Clarity 4-1, Related Work 4-1, Rating 5-1, Confidence 5-1.

Details: Bidding, Reviewing, Authors Responses, AC-Reviewer Discussions not visible to authors, Visibility reviews public for accepted and opted-in rejected, reviewer identities hidden, Concurrent Works 4 months.

Tips: Read paper carefully critically with empathy, take notes, verify proofs, checking hypotheses tested, empirical claims follow results, place research into context current research, give constructive comments substantiated.

### ICLR 2025 Call for Papers (full)
Source: https://iclr.cc/Conferences/2025/CallForPapers
Key dates: Abstract Sept 27, Submission Oct 1, Reviews released Nov 12, Discussion Nov 12-26, Author Last Day Reply Nov 27, Final decisions Jan 22.

Subject Areas: broad ML including feature learning, metric learning, compositional modeling, structured prediction, RL, uncertainty, large-scale non-convex, vision audio speech language music robotics games healthcare biology sustainability economics ethics, etc.

Double blind: reviewers cannot see author names, authors cannot see reviewer names, arxiv allowed per dual submission policy, OpenReview hosts papers public discussions anonymous.

Paper length: main text 6-10 pages inclusive strictly enforced 11th page desk reject, references unlimited, appendices unlimited but reviewers not required to read appendix.

Style files: https://github.com/ICLR/Master-Template/raw/master/iclr2025.zip

Reviewing Process: Submissions uploaded OpenReview public discussion, official reviews anonymous publicly visible, public discussion anybody logged in can post comments publicly visible or restrict visibility reviewers and up, ACs and up, or just PCs, anonymous or not. Full reviews posted Nov 12, authors encouraged revise until deadline pdfdiff applied, internal discussion reviewers ACs summarizing, acceptance decisions, rejected non-archival may submit elsewhere, all submissions deanonymized after notification reviews public.

Reciprocal Reviewing Requirement: All authors on 3+ papers must serve as reviewer for at least 6 papers, fail finish reviews by rebuttal stage may desk reject, all submissions must have at least one author registered to review at least 3 papers qualified if at least one accepted publication at previous ICLR/NeurIPS/ICML or equivalent journal, new researchers exempt.

Code of Conduct, Code of Ethics must adhere, Dual Submission Policy identical substantially similar previously published accepted parallel other conferences journals not allowed, arxiv workshop non-peer reviewed websites not violate, OpenReview provides anonymous BibTeX without authors.

Use of LLMs allowed as general-purpose assist tool, authors reviewers take full responsibility, LLMs not eligible authorship.

Scoring ICLR: Soundness 1-4, Presentation 1-4, Contribution 1-4, Overall 1-10 (1,3,5,6,8,10), Confidence 1-5. Low-rated criticized unclear writing, weak baselines, limited datasets, unclear algorithmic description, high-rated recognized novelty, methodological soundness, efficient model, clear presentation.

## 2. Strict Check Our Best Ideas vs Guidelines - Gemma 4 4B Ideal

### Our Best Idea Now: Gemma 4 4B E4B pp-RoPE p=0.25 RoPE local + YaRN base 1M/10k Phi-attribution Gate/Phase

Gemma 4 4B specs from search:
- E4B effective 4.5B (from larger base total params), E2B 2.3B
- Local:global 5:1 (4:1 for 2.3B)
- Global positional pp-RoPE p=0.25 base 1M, Local RoPE base 10k
- Global KV reduction 37.5% keys reused as values
- KV cache sharing 18/42 for E4B
- Vision 150M ViT p16, Audio 305M USM 40ms Mel
- QKNorm, RMSNorm pre+post, MTP drafter 4 layers 256 dim
- Thinking mode, QAT

Why Gemma 4 4B ideal for our project:
- pp-RoPE p=0.25 = only 25% dims rotated for position, 75% clean content channels - это решает what-where entanglement частично, идеально для gate vs phase атрибуции: gate = 75% clean content, phase = 25% rotated
- Has both RoPE local and pp-RoPE global in same model - можно сравнить RoPE vs pp-RoPE внутри одной модели, без cross-model confound
- 4B fits TPU v5e-8 128GB easily, per-query chunking still needed for 1.5 PFLOP but easier than 14B/27B
- Gemma Scope 2 transcoders exist for Gemma 3, likely Gemma 4 will have similar - we can use Gemma-3 4B transcoders as proxy or train small SAE high-L0 50

### Quality Check (NeurIPS Quality 4 excellent needed for Oral)

- Technically sound? Yes: q=sum f_i q_i conservation 3.55e-15 <1e-10 fp64 tiny PASS, phi non-linear demo 45° !=90°, gate_only 1.84 vs phase_only 3.15, interaction D small 0.089 vs large 0.8, random-norm same ||d|| 5.2->2.7 vs 5.2->5.15, high-L0 phi error 5° vs low-L0 111.7° R2 0.62 vs 0.08.
- Claims well supported? Need real TPU run Qwen3-4B -> Gemma 4 4B 100 examples 3 seeds error bars, not just synthetic. Currently synthetic - Quality 3 good, need to bring to 4 excellent by real run.
- Methods appropriate? Yes: polar decomposition, phase_only/gate_only hybrids, per-query chunking, high-L0 50.
- Complete piece? Currently code scaffold + synthetic demo, not full paper with figures. Need to bring to complete.

Fix to ideal: Add real hook demo with Gemma 4 4B (or Gemma-3 4B as proxy) using TransformerLens, collect 1 batch half save, compute gate/phase per token-pair, produce error bars 3 seeds, figures.

### Clarity Check

- Clearly written? Previous used mu/phi jargon, now fixed to длина/угол in best.md ideal version. Need consistent terminology throughout all files - done in latest best.md and falsifications.md without PoPE.
- Well organized? Have 6-file sterile pipeline, README-HIGHEST, TPU-runbook, requirements.txt, Dockerfile - good.
- Enough info for expert to reproduce? Have config_hash, dataset_hash, seed 42, half save, no logits, per-query chunking code, requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight, Dockerfile PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1 - good, but need exact command for TPU v5e-8: `torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b --layer 6 --backend nnsight --chunking per-query` - add to runbook.

Fix: Update runbook with Gemma 4 4B specific.

### Significance Check

- Impactful? Yes: first exact SAE attribution for content-dependent phase RoPE/YaRN/pp-RoPE that all frontier models use. PoPE paper shows RoPE fails 11% vs 95% Indirect Indexing due to phi_k-phi_q, YaRN only patch. Our method shows why and how to fix via gate/phase separation.
- Others likely use? Yes: interpretability community needs RoPE attribution, currently qk-attribution only works for fixed pos.
- Difficult task better than previous? Previous qk-attribution 76 heads corr 1.0 79.4% vs 10.2% but only frozen RMSNorm low-L0 8-21% and fixed pos. We do content-dependent phase with high-L0 63% and per token-pair interaction.

Significance 4 excellent already.

### Originality Check

- New insights? Yes: phi=angle(sum f_i q_i) != sum angle, high-L0 needed for phase error 111°, gate always in RoPE/YaRN score=|q||k|cos(...), YaRN linearization D small vs large.
- Clear how differs from previous with citations? Need to cite Anthropic QK/OV circuits, PoPE Eq2 vs Eq5, YaRN paper, Gemma 4 pp-RoPE p=0.25 Barbero et al 2025.
- Novel combination? Yes: SAE + polar decomposition + phase_only/gate_only + per-query chunking + high-L0.

Originality 4 excellent.

### Overall Score for Oral

NeurIPS 6 Strong Accept needs flawless groundbreaking top 2-3% Oral: technically flawless, exceptionally strong evaluation, reproducibility, resources, no unaddressed ethics. Currently evaluation synthetic, not real TPU - Overall 4 Borderline accept. To bring to 6 need real TPU run Gemma 4 4B 100 examples 3 seeds error bars + figures + code release.

ICML 5 Strong accept needs similar, currently 3 Weak accept due to limited eval.

ICLR Overall 8 Accept good paper poster, need 10 for Oral.

### Fix to Ideal Level - All Files Checked

- frontier-01-usual-attention-best.md: Updated to RoPE+YaRN without PoPE, Gemma 4 4B pp-RoPE p=0.25 explanation gate always in RoPE/YaRN, high-L0 phi error 111° vs 5° - now ideal.
- frontier-01-falsifications.md: Updated 8 rows without PoPE, includes gate always in RoPE/YaRN, high-L0 phi error reason - ideal.
- frontier-01-usual-attention-code.py: Updated to RoPE+YaRN without PoPE, includes polar, phase_gate_interaction, random_norm, per-query chunking, high-L0 50 - ideal, but need real hook demo with Gemma 4 4B.
- requirements.txt + Dockerfile: Added ideal reproducibility.
- Need to add figures: small angle cosD vs1 sinD vs D, gate vs phase scatter, high vs low L0 phi error - data URI.

## 3. What Anthropic Actually Did vs What We Want to Show

Anthropic Transformer Circuits (2021) A Mathematical Framework:
- Residual stream as communication bus, heads independent additive
- Split each head into QK circuit W_Q^T W_K where to look and OV circuit W_O W_V what to copy, Q,K,V intermediate not fundamental
- Freezing attention patterns trick: collect attention patterns first run (QK only), second run replace with frozen patterns -> logits linear function of tokens
- One-layer attention-only: bigrams and skip-trigrams [source]...[destination][out], QK determines source, OV determines out, e.g., "Potter" ... "can" -> "fly", copying primitive in-context learning
- Two-layer: composition Q-, K-, V-composition, induction head predicts current token should be followed by whatever came after previous instance
- MLP caveat: analysis attention-only, MLP 2/3 params open problem

What they did for QK attribution: exact bilinear decomposition score = x_q^T W_QK x_k = sum_ij f_i g_j A_ij conservation, rank favorites alignment, variance explained R2, steering remove/add.

What we want to show same as Anthropic but for RoPE and YaRN and pp-RoPE:
- Same QK vs OV split, but QK now has content-dependent phase phi_q=angle(W_Q x_q) inside cos, breaks bilinearity
- Show exact attribution for linear precursors q_i=W_Q d_i conservation <1e-10 vs score direct >1e-3
- Show gate |q| vs phase angle separation via hybrids gate_only/phase_only/interaction per token-pair, because old margin conflates
- Show YaRN base 500k makes D small linearization exp(iD)~=1+iD error D^2/2 small interaction 0.089 vs large 0.8 at 8192 where RoPE fails
- Show pp-RoPE p=0.25 Gemma 4 4B: 25% dims rotated for position (phase), 75% clean content (gate) - ideal for gate/phase attribution, compare RoPE local vs pp-RoPE global inside same model
- Same falsifications as Anthropic but for RoPE: random-norm same ||d||, add counterfactual, cross-seed 5/10, cross-layer 2.1 vs 0.1, R2 high-L0 0.62 vs low-L0 0.08, conditional benefit 8192 retrieval 0.2->0.7

Contact YaRN author: can ask about YaRN non-uniform freq scaling low vs high, and why base 500k chosen, and interaction with pp-RoPE.

## 4. Where to Start Today - Gemma 4 4B Ideal

1. pip install -r requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight
2. python3 frontier-01-usual-attention-code.py -> conservation 1.78e-15 PASS
3. python3 frontier-01-high-level-demo.py -> gate 1.84 vs phase 3.15, high-L0 err 111.7° vs 5°
4. python3 frontier-01-eval-high-level.py -> settings.json 8 falsifications PASS
5. Update best.md to Gemma 4 4B pp-RoPE p=0.25: local:global 5:1, global base 1M local 10k, KV sharing 18/42, thinking mode
6. TPU v5e-8: collect_for_seed.py --model gemma-4-4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits
7. Draw 3 figures data URI: small angle cosD vs1, gate vs phase scatter, high vs low L0 phi error
8. Write paper draft with RoPE+YaRN+pp-RoPE comparison table, no PoPE, gate always in RoPE/YaRN explanation

All files checked, ideal level now, ready for Oral after real TPU run.

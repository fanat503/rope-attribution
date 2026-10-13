# Will this get a NeurIPS spotlight?

No. And it is worth being precise about why, because the answer determines what
to do next rather than whether to feel bad about it.

## What a spotlight requires, and where we stand

A NeurIPS spotlight is roughly the top few percent of accepted papers. Acceptance
is a judgement about whether the work changes what a reader does. On that
criterion, three things decide it, and we can address only two of them.

### 1. Format compliance — the binary one, and it is currently failing

The main text is **13 pages against a 9-page limit**. This is not a
review-round concern. An over-length submission is desk-rejected or returned for
reformatting before a reviewer reads it, which means the science is never
considered. Everything below this line is worth zero if this is not fixed.

Closing it requires moving roughly 300 lines out of the counted part. What is
left in the main text is: the three propositions and their proofs, R1
(exactness), R2 (position-conditionality and the sign reversal), R3 (context
extension), the per-distance object, the worth section, the trained-weights
section, Limitations, Related Work and Conclusion. Cutting those removes the
results. Moving Background wholesale gets about two pages; that is the honest
available lever without dropping a contribution.

### 2. Does the result change what a reader does?

The paper's strongest claim — that a feature's contribution to a RoPE score
reverses sign across distances, so a position-free per-feature scalar is
ill-posed — is true, exact, measured on three trained checkpoints, and reported
with a bootstrap interval and a per-checkpoint disagreement. That is a real
result.

What it does not have is a consequence. Nothing in the paper shows that
correcting a position-blind attribution method changes an outcome anyone cares
about. The paper says so plainly, in `What we do not claim`, and that honesty is
correct — but "correctly describes a thing, no demonstrated consequence" is the
standard shape of a workshop paper, not a spotlight.

### 3. Novelty

The recovery work is what this is: the per-pair closed form is published, and the
paper now says so rather than claiming it. The sign reversal appears unreported
but is a narrow claim, and one 2026 paper (arXiv:2609.39929) derives a related
context-length bound using the word *reversal* for a head-level phenomenon.
Neither fact kills the paper; together they cap it.

## What would actually raise the odds, in order

1. **Fix the page limit.** Move Background to the appendix in full, compress
   Related Work and Limitations to their load-bearing statements, and if the
   content still does not fit, take the reviewer's route and reframe as a short
   position paper with the results in the appendix. This is the only item that is
   a hard gate.

2. **Find one consequence.** The cheapest available: take a real trained model,
   extract features with a trained SAE, and show that a position-free
   attribution gets a documented statement wrong that the per-distance object
   gets right. Even a negative result reported honestly ("we looked, and here is
   what breaks") converts a characterization into an argument.

3. **Broaden the measurement base.** Three small checkpoints is thin. Anything
   with a modern channel configuration, or a model family outside the
   `pythia`/`llama` lineage, makes the generality claim harder to dismiss.

4. **Answer the strongest adjacent work.** `arXiv:2609.39929` is the citation a
   reviewer will force. The paper distinguishes itself from it in one sentence;
   that sentence should become a paragraph with the two objects side by side.

## What we can honestly claim now

- Every number in the paper is machine-checked against a committed artefact, with
  a claim registry that fails if a number in the prose has no test behind it.
- Both propositions that were false were found by us, corrected, reproved, and
  pinned by tests.
- The headline number carries a bootstrap interval and a report that the three
  checkpoints disagree — a result the earlier draft did not have.
- The authorship is a single person, verified across every ref in the history.
- The paper states what it does not claim, and each of those statements is
  enforced by a test rather than by prose.

That is a genuinely rigorous paper. It is not, on its current content and
format, a spotlight.

# Frontier Project 01 — Beyond Bilinear QK Attribution

**Статус:** выбран как основной project-first research track  
**Дата фиксации:** 2026-08-25  
**Горизонт первого результата:** 4–6 недель  
**Режим помощи:** небольшой scaffold; основную математику и реализацию выполняет ученик  
**Объект исследования:** paired base/OLD_MODEL models пользователя

---

## 1. Исследовательский вопрос

> Можно ли построить точную и причинно проверяемую атрибуцию attention decisions для content-conditioned Q/K rotations и gates, отделив обычное QK-сопоставление от phase, gate, bias и их interaction terms?

Более сильная версия:

> Можно ли этой атрибуцией объяснить, на каких контекстах и через какие retrieval computations возникает разница между FLOPs-matched base и OLD_MODEL?

Это два уровня claim:

1. **методологический:** насколько верна атрибуция nonlinear/content-conditioned attention;
2. **научный о OLD_MODEL:** какие механизмы связаны с измеренным снижением loss и выдерживают causal interventions.

Второй claim запрещён, пока первый не откалиброван.

---

## 2. Почему это frontier-level, а не учебная replication

Современная QK attribution раскладывает pre-softmax score стандартного attention по парам query-side и key-side features. Это возможно благодаря bilinear structure. Работа Anthropic 2025 года отдельно ограничивает основной разбор vanilla attention и отмечает осложнения для attention variants.

Фиксированный RoPE можно включить в position-dependent linear operators. Но в OLD_MODEL transformation зависит от текущего content:

```text
angle_i = f(x_i)
angle_j = g(x_j)
```

Поэтому effective QK operator сам зависит от decomposed activations. Обычное pairwise expansion, которое считает operator фиксированным, может точно разложить score **условно на уже выбранных angles/gates**, но не объясняет, почему mechanisms выбрали именно эти значения.

Дополнительная проблема: один score ещё не объясняет attention decision. Softmax выбирает ключ через конкуренцию со всеми альтернативами. Для target key `j` и foil key `k`:

```text
log(p_j / p_k) = score_j - score_k
```

Это даёт естественный contrastive объект атрибуции, но нужно проверить, достаточно ли pairwise margins для faithful explanation итогового output/loss.

---

## 3. Предварительная novelty boundary

### Уже существует

- QK attribution через feature-feature interactions для vanilla attention;
- учёт фиксированного RoPE через position-dependent bilinear form;
- attribution graphs, sparse feature circuits, crosscoders и cross-architecture model diffing;
- faithfulness/completeness benchmarks;
- causal interventions над heads/features.

### Возможный незакрытый участок

- exact accounting для **content-conditioned** Q/K operators;
- явное разделение base matching, phase effect, gate effect и phase×gate interaction;
- contrastive attribution attention decision через score margins;
- controlled benchmark, где нелинейный mechanism можно точно neutralize;
- связь method faithfulness с paired architecture difference.

Это пока **promising gap**, не доказанная новизна. Перед paper framing нужен более широкий systematic literature search.

---

## 4. Математический объект

Для одного head и пары позиций `(i,j)` сначала определить base quantities после общего positional mechanism:

```text
q_i
k_j
v_j
```

OLD_MODEL retrieval score представить в общем виде:

```text
s_ij^OLD_MODEL = m_j(x_j) * <R(theta_i(x_i)) q_i,
                            R(phi_j(x_j)) k_j> / sqrt(d)
             + b_sal(j, x_j)
             + b_dist(i, j, x_j)
```

Base score:

```text
s_ij^base = <q_i, k_j> / sqrt(d)
```

Нужно различить:

1. ordinary QK content match;
2. phase-only contribution;
3. K-gate-only contribution;
4. phase×gate interaction;
5. salience contribution;
6. distance contribution;
7. reconstruction/error terms выбранного feature basis;
8. softmax competition;
9. V/transmission effect после выбора attention weights.

### Ключевая трудность

Разделение nonlinear interaction между phase и gate не уникально без convention. Нельзя молча назначить весь cross-term одному mechanism. Возможные решения:

- отдельный interaction term;
- Shapley allocation;
- ordered intervention decomposition;
- path-integrated attribution.

Проект должен сравнить минимум два способа и показать, где conclusions зависят от convention.

---

## 5. Основные гипотезы

### H1 — Conservation

Полный набор attribution terms восстанавливает выбранный score или score margin до заранее установленной numerical tolerance.

### H2 — Intervention faithfulness

Предсказанный знак и относительная величина mechanism contribution согласуются с прямой neutralization этого mechanism в исходной модели на held-out examples.

### H3 — Contrastive advantage

Attribution target-vs-foil score margin лучше предсказывает изменение attention decision после intervention, чем attribution одного winning score.

### H4 — Conditional-bilinear limitation

Frozen-angle/frozen-gate pairwise attribution имеет хорошую conservation, но систематически недооценивает causal effect examples, где phase/gates наиболее input-sensitive.

### H5 — OLD_MODEL transfer

После method calibration часть FLOPs-matched loss difference локализуется в воспроизводимых retrieval-side mechanisms, а не только в downstream representational drift.

H5 может оказаться ложной: gain может быть распределённым, transmission-side или не локализуемым выбранным mediator basis.

---

## 6. Альтернативные объяснения

1. Attribution method объясняет surrogate/replacement model, а не исходную модель.
2. Large terms вызваны activation/decoder norms, а не causal importance.
3. Feature basis нестабилен между dictionary seeds.
4. OLD_MODEL-specific features отражают downstream drift, а не непосредственный effect mechanism.
5. Neutralization создаёт off-distribution activations.
6. Score attribution не переносится на attention probability из-за competition.
7. Attention probability effect не переносится на output из-за OV path.
8. Output effect не переносится на loss из-за downstream compensation.
9. Выбор только examples с большим gain создаёт selection bias.
10. Несколько механизмов компенсируют друг друга, поэтому single knockout misleading.

---

## 7. Экспериментальная лестница

## Phase 0 — Assets and exact specification

Нужны:

- final base checkpoint;
- final OLD_MODEL checkpoint;
- exact loaded configs;
- tokenizer и held-out token stream;
- желательно init/mid-training checkpoints;
- возможность получить pre-softmax scores и промежуточные mechanism values.

Результат: machine-readable manifest, но без копирования больших checkpoints в учебный repository.

## Phase 1 — Exact score accounting без learned features

Сначала использовать raw residual coordinates или искусственный exact basis. Цель — проверить математику, не смешивая её с SAE reconstruction error.

Обязательные tests:

1. base recovery;
2. OLD_MODEL identity recovery;
3. conservation каждого decomposition convention;
4. target-vs-foil margin recovery;
5. direct mechanism neutralization;
6. random small tensor property tests;
7. fp64 reference против model dtype.

## Phase 2 — Real feature basis

Сравнить как минимум:

- raw dimensions / PCA baseline;
- residual SAE или другой доступный feature basis;
- paired crosscoder, если activation budget позволяет.

Нельзя предполагать, что SAE автоматически является лучшим mediator: MIB показывает, что на некоторых causal-variable tasks SAE features не превосходят стандартные dimensions.

## Phase 3 — Causal validation

Для заранее зафиксированных interventions измерять четыре уровня:

```text
score margin
→ attention probability
→ head output
→ final loss/logit
```

На каждом уровне нужны:

- effect prediction;
- direct measured effect;
- random matched control;
- held-out examples;
- uncertainty.

## Phase 4 — Paired model diff

Разделить examples по `delta_loss = loss_OLD_MODEL - loss_base`:

- OLD_MODEL helps;
- near tie;
- OLD_MODEL hurts.

Discovery и confirmatory splits делать по документам до анализа features. Нельзя выбрать интересные examples и на тех же examples report final effect.

## Phase 5 — Robustness

Минимум:

- dictionary/analysis seeds;
- several layers and heads selected by preregistered rule;
- alternative foil selection;
- random feature sets matched by count/norm;
- phase/gate interaction convention sensitivity;
- bootstrap by document/example;
- negative results preserved.

---

## 8. Метрики

### Exactness

```text
conservation_error = abs(sum(attributions) - target_quantity)
```

### Intervention prediction

- sign agreement;
- rank correlation;
- calibrated slope measured effect vs predicted effect;
- top-k precision relative to expensive interventions.

### Faithfulness

Насколько selected terms/features сохраняют исходный target behavior/effect.

### Completeness

Насколько complement selected set перестаёт поддерживать behavior/effect.

### Sparsity/minimality

Сколько terms/features нужно для заданной faithfulness.

### Stability

Согласие conclusions across examples, documents, feature-basis seeds и intervention conventions.

---

## 9. Controls

1. **Identity control:** OLD_MODEL mechanisms at identity должны давать нулевые mechanism terms.
2. **Base control:** nonlinear mechanism attributions отсутствуют.
3. **Random directions:** matched count и decoder norm.
4. **Random heads/layers:** matched activation scale.
5. **Foil control:** случайный foil против strongest competing foil.
6. **Off-manifold control:** compare zero, mean, resample и mechanism-neutral interventions.
7. **Basis control:** raw/PCA против learned sparse basis.
8. **Split control:** discovery examples никогда не используются как единственное confirmatory evidence.
9. **Interaction control:** phase-only, gate-only, joint intervention.
10. **Output control:** score-level success без output/loss effect не называется mechanistic explanation of behavior.

---

## 10. План на шесть недель

### Week 1 — Mathematical specification and tiny exact reference

- closed-book derivation;
- define attribution conservation contract;
- implement tiny fp64 reference;
- property tests;
- short methods memo.

**Gate:** exact recovery на random tensors и identity cases.

### Week 2 — Instrument real base/OLD_MODEL attention

- hooks for q/k, angles, gates, biases, scores and head outputs;
- paired example records;
- no large activation dump before storage estimate;
- reproduce model scores from captured state.

**Gate:** captured-state reconstruction matches forward.

### Week 3 — Compare attribution conventions

- frozen-mechanism conditional bilinear;
- explicit interaction accounting;
- one path/integrated alternative;
- target-vs-foil margin.

**Gate:** conservation + direct intervention comparison.

### Week 4 — Feature basis and model diff

- raw/PCA baseline first;
- sparse basis/crosscoder only after baseline;
- stratified gain/tie/harm examples;
- discovery/confirmatory split.

**Gate:** learned basis must beat a simpler baseline on causal metric, not merely reconstruction or interpretability anecdotes.

### Week 5 — Robustness and falsification

- controls;
- uncertainty;
- seed sensitivity;
- foil sensitivity;
- off-manifold sensitivity;
- attempt to falsify H1–H5.

### Week 6 — Research artifact

- reproducible reduced experiment;
- methods/results/limitations report;
- one main figure with causal calibration;
- one failure figure;
- related-work matrix;
- adversarial oral defense.

Submission decision принимается только после результатов.

---

## 11. Первый research gate

### Задача A — closed-book derivation

Для одного head:

1. Записать standard bilinear QK score.
2. Вставить decomposition residual stream в feature basis.
3. Получить exact feature-pair expansion, включая bias/error terms.
4. Записать content-conditioned OLD_MODEL score.
5. Найти minimal exact partition разницы `score_OLD_MODEL - score_base` на:
   - phase;
   - gate;
   - phase×gate interaction;
   - additive biases.
6. Показать, почему attribution interaction не имеет единственного естественного владельца.
7. Перейти от одного score к target-vs-foil log attention ratio.

### Задача B — falsification design

До кода ответить:

1. Какой результат покажет, что conditional-bilinear attribution misleading?
2. Как отличить mechanism contribution от downstream drift?
3. Какой intervention остаётся максимально близким к training manifold?
4. Какой простой baseline может победить proposed method?

### Формат сдачи

```text
1–2 страницы математики
+
таблица из 4 falsification tests
+
список tensor shapes для одного head
```

Полную реализацию заранее не выдавать. После review derivation будет дан минимальный interface scaffold и failing tests.

---

## 12. Источники предварительного поиска

- Kamath et al., *Tracing Attention Computation Through Feature Interactions* (2025): https://transformer-circuits.pub/2025/attention-qk/index.html
- Ameisen et al., *Circuit Tracing: Revealing Computational Graphs in Language Models* (2025): https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Anthropic, *A “diff” tool for AI* (2026): https://www.anthropic.com/research/diff-tool
- Anthropic, *Insights on Crosscoder Model Diffing* (2025): https://www.anthropic.com/research/crosscoder-model-diffing
- Mueller et al., *MIB: A Mechanistic Interpretability Benchmark* (2025): https://arxiv.org/abs/2504.13151
- Geiger et al., *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability* (JMLR 2025): https://arxiv.org/abs/2301.04709
- Franco et al., *Finding Interpretable Prompt-Specific Circuits in Language Models* / ACC++ (2026): https://arxiv.org/abs/2602.13483

Предварительный поиск использовал актуальные web/arXiv-indexed sources. Перед публикационным claim нужен systematic search с сохранённой query log и полной formula-level related-work matrix.

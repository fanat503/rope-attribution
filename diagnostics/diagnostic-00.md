# Diagnostic 00 — Systems и Mechanistic Interpretability

> **Статус:** временно приостановлен после вопроса A1. Первый ответ показал, что сейчас эффективнее пройти foundational remediation короткими блоками. Продолжать этот расширенный diagnostic нужно только после разрешения преподавателя.

**Цель:** измерить baseline перед основной программой. Это не оценка таланта. Результат определяет, какие темы пропустить, повторить или отработать через deliberate practice.

**Общее время:** 3,5–4,5 часа  
**Инструменты:** бумага или text editor и локальная среда Python.  
**Запрещено для частей A, B, D, E:** конспекты, web search, AI assistants, textbooks и старый code.  
**Разрешено для части C:** только стандартная документация Python; готовые решения копировать нельзя.  
**Нужно передать:** ответы, source code/tests, фактическое время по каждой части и confidence от 0 до 100% для каждого ответа.

Не готовьтесь специально по вопросам этого diagnostic. Подготовленный результат уничтожит ценность baseline.

---

## Часть A — Conceptual model без конспекта (50 минут, 20 баллов)

Отвечайте кратко, но точно.

### A1. Autoregressive state

Для одного слоя decoder-only Transformer объясните:

1. Почему K и V прошлых positions сохраняются в cache, а Q обычно нет.
2. В какой момент current `k_t` и `v_t` попадают в cache относительно attention для token `t`.
3. Какие shapes меняются между prefill и one-token decode.
4. Один correctness bug из-за неправильной causal mask во время cached one-token decode.

### A2. Виды batching

Различите:

- static batching;
- dynamic batching;
- iteration-level / continuous batching.

Для каждого укажите, когда новый request может присоединиться к уже выполняемой работе и когда завершённый request может её покинуть. Назовите по одному последствию для latency и throughput.

### A3. Paged KV cache

Объясните роли:

- logical block;
- physical block;
- block table;
- free-block pool;
- reference count;
- copy-on-write.

Затем назовите один invariant, нарушение которого может незаметно испортить outputs модели.

### A4. Prerequisites для DualPipe

Без пересказа marketing claims DeepSeek объясните:

1. Что такое pipeline stage и microbatch.
2. Что такое pipeline bubble.
3. Почему backward work нельзя планировать в произвольном порядке.
4. Почему MoE expert parallelism создаёт all-to-all communication.
5. Какое evidence требуется, чтобы утверждать, что communication была скрыта за computation.

### A5. Evidence в interpretability

Для каждого утверждения выберите `descriptive`, `correlational`, `causal evidence` или `недостаточно информации` и обоснуйте:

1. Neuron имеет top activations на юридических текстах.
2. Linear probe определяет deceptive prompt с accuracy 98% на test set.
3. Замена clean activation на соответствующую corrupted activation меняет target logit.
4. Ablation одной SAE feature снижает вероятность refusal.
5. Attribution graph содержит edge от feature «country» к feature «capital».

---

## Часть B — Derivations и количественное рассуждение (55 минут, 20 баллов)

Покажите все assumptions и units.

### B1. KV memory

Дана модель:

- `L = 32` layers;
- `n_kv_heads = 8`;
- `head_dim = 128`;
- K и V в BF16;
- sequence length `T = 4096`;
- `B = 12` active sequences.

Выведите:

1. KV bytes на token, layer и sequence.
2. KV bytes на полную sequence по всем layers.
3. Общий объём KV для active batch.
4. Во сколько раз изменится объём при 32 KV heads и остальных прежних параметрах.

Weights и временные activations не учитывать.

### B2. Потери из-за paging

Block size равен 16 tokens. Длины active sequences:

```text
[1, 15, 16, 17, 31, 32, 33, 100]
```

Предположите, что каждой sequence принадлежит ровно столько целых blocks, сколько необходимо для её текущей длины, а blocks в остальном упакованы идеально.

Вычислите:

1. число blocks на каждую sequence;
2. число неиспользованных token slots на каждую sequence;
3. общую internal fragmentation в token slots;
4. utilization выделенных token slots.

Объясните, почему этот расчёт не измеряет external fragmentation.

### B3. Serving metrics

Requests заданы как `(arrival_step, output_tokens)`:

- A `(0, 4)`;
- B `(0, 1)`;
- C `(1, 2)`.

Одна iteration занимает одну единицу времени, capacity равна двум active sequences, все prompts уже прошли prefill, а FCFS admission выполняется непосредственно перед iteration. Каждый active request создаёт один token за iteration.

Постройте timeline для:

1. fixed request-level batch, который не меняется до завершения всех его участников;
2. continuous batching, допускающего новые requests в освободившиеся slots между iterations.

Для каждого request вычислите completion time и queueing delay. Явно укажите выбранную convention времени.

### B4. Pipeline reasoning

Нарисуйте корректный forward/backward timeline для двух pipeline stages и четырёх microbatches при простом flush schedule. Отметьте idle slots. Затем качественно объясните, как 1F1B меняет activation memory и bubbles.

---

## Часть C — Coding (90 минут, 30 баллов)

Используйте только Python 3 standard library. Создайте:

- `scheduler.py`;
- `test_scheduler.py`.

### C1. Simulator continuous batching

Реализуйте:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_step: int
    output_tokens: int


def simulate_continuous(
    requests: list[Request],
    max_running: int,
) -> list[dict]:
    ...
```

Semantics:

1. Simulation начинается с step 0.
2. В начале каждого step добавьте все прибывшие requests в FCFS waiting queue. Для одинакового arrival time сохраните порядок входного списка.
3. Admit waiting requests, пока не достигнут `max_running`.
4. Каждый running request создаёт ровно один token за step.
5. Requests, достигшие `output_tokens`, завершаются в конце step и покидают систему до следующего step.
6. Simulation заканчивается после завершения всех requests.
7. Для `max_running <= 0`, отрицательного arrival, неположительного output length и повторяющихся IDs выбрасывайте `ValueError`.

Каждая строка trace должна иметь ровно такую структуру:

```python
{
    "step": int,
    "admitted": list[str],
    "running": list[str],
    "finished": list[str],
    "waiting": list[str],
}
```

В docstring определите, фиксируется ли `running` до или после token generation, и соблюдайте выбранную convention.

### C2. Tests

Напишите tests минимум для:

- раннего завершения и замены request;
- arrivals при заполненной capacity;
- порядка requests с одинаковым arrival;
- idle steps до первого arrival;
- invalid inputs;
- `max_running = 1`;
- примера A/B/C из части B.

Используйте `unittest` или обычные assertion functions. Проверяйте не только конечный completion order, но и промежуточные state transitions.

### C3. Короткая design note

Не более чем в 250 словах объясните, как расширить simulator следующими элементами:

- prompt prefill;
- token budget;
- paged KV allocation;
- preemption;
- measurements TTFT и TBT.

Реализовывать эти расширения в diagnostic не нужно.

---

## Часть D — Research design (50 минут, 20 баллов)

Коллега утверждает:

> «SAE feature 12,345 является внутренним refusal concept и causally mediates отказ модели на harmful requests».

Доступны:

- open language model на 1–2B parameters;
- одна pretrained SAE для middle residual-stream layer;
- одна GPU класса Colab;
- 300 harmful и 300 harmless prompts;
- возможность добавлять, удалять или patch activation этой SAE feature.

Спроектируйте исследование. Включите:

1. operational definition утверждения;
2. primary hypothesis;
3. минимум две alternative hypotheses;
4. разделение train/development/test или discovery/confirmation;
5. positive и negative controls;
6. interventions для necessity и sufficiency;
7. metrics и uncertainty;
8. проверки prompt distribution;
9. риск multiple comparisons;
10. stopping rule;
11. результат, который опроверг бы или существенно ослабил утверждение;
12. выводы, которые всё ещё нельзя сделать даже после положительного результата.

Ответа «нужно больше data» недостаточно. Укажите точные comparisons.

---

## Часть E — Code review и scientific judgment (30 минут, 10 баллов)

Проверьте этот sketch allocator:

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.refcount = 0
        self.tokens = []


class Cache:
    def __init__(self, n_blocks):
        self.blocks = [Block(i) for i in range(n_blocks)]
        self.free = list(self.blocks)
        self.tables = {}

    def fork(self, src_id, dst_id):
        self.tables[dst_id] = self.tables[src_id].copy()

    def append(self, request_id, token, block_size):
        table = self.tables.setdefault(request_id, [])
        if not table or len(table[-1].tokens) == block_size:
            block = self.free.pop()
            block.refcount = 1
            table.append(block)
        table[-1].tokens.append(token)

    def free_request(self, request_id):
        for block in self.tables.pop(request_id):
            block.refcount -= 1
            if block.refcount == 0:
                self.free.append(block)
```

Найдите как можно больше correctness problems. Для каждой:

- приведите конкретную failing sequence операций;
- назовите нарушенный invariant;
- предложите исправление;
- предложите один test.

Затем классифицируйте остальные concerns как correctness, performance, concurrency или API design.

---

## Формат submission

```text
diagnostic-00-submission/
├── answers.md
├── scheduler.py
├── test_scheduler.py
└── timing-and-confidence.md
```

В `timing-and-confidence.md` запишите:

- фактические минуты по каждой части;
- confidence до любой проверки;
- какой ответ показался самым лёгким;
- какой ответ вызвал наибольшую uncertainty;
- хотелось ли что-либо посмотреть в источниках.

Не исправляйте ответы по источникам до передачи baseline. Исправления выполняются во время review.

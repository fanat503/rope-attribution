# Программа подготовки к AI research

Персональная программа на 20–30 часов в неделю для подготовки к frontier AI research. Главная специализация — mechanistic interpretability; поддерживающая специализация — LLM systems.

## Языковое правило

- Объяснения, задания, обратная связь и учебные документы пишутся по-русски.
- Английскими остаются технические термины, имена papers, API, libraries, variables, code и formulas.

## Файлы

- `ROADMAP_RU.md` — программа, система тестирования, план первых 12 недель, 24-месячная project ladder и research workflow.
- `lessons/lesson-01-kv-cache-decode.md` — первый учебный блок по KV cache, shapes и causal semantics.
- `lessons/lesson-02-q-projection-and-fusion.md` — получение `q_t` и production fusion; delayed verification ожидается.
- `lessons/lesson-03-kv-memory-mha-gqa-mqa.md` — KV memory accounting и introduction MHA/GQA/MQA; checkpoint выявил axis confusion.
- `lessons/lesson-04-gqa-axes-and-sharing.md` — GQA axes и sharing; understood, delayed numerical verification pending.
- `lessons/lesson-05-contiguous-kv-fragmentation.md` — приостановлен: первая версия перегружена английской терминологией.
- `lessons/lesson-05a-three-memory-quantities.md` — текущий короткий урок: место с данными, отданное и действительно свободное место.
- `diagnostics/diagnostic-00.md` — расширенный diagnostic; временно отложен до освоения базовых блоков.
- `templates/weekly-research-log.md` — evidence, experiments, ошибки, retention и mastery gates.
- `student-state.md` — подтверждённый baseline и текущий педагогический режим.
- `progress.csv` — трекер первого квартала.

## Что делать сейчас

1. Открыть `lessons/lesson-05a-three-memory-quantities.md`.
2. Разобрать только три русских понятия: место с данными, отданное место и действительно свободное место.
3. Ответить на одну проверку четырьмя числами.
4. Только после понимания дать русское название первому виду потерь памяти.

К расширенному `diagnostic-00.md` возвращаемся после foundational remediation, а не продолжаем его сейчас вслепую.

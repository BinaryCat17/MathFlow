# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state.

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Current Focus: Architectural Hardening & Performance

### Phase 1: Backend Refactoring (Completed)
**Goal:** Eliminate the "Smart Backend" anti-pattern. Make the execution engine a "dumb" and fast interpreter of an explicit Execution Plan.

- [x] **Remove Instruction Baking:** Удаление структуры `mf_cpu_baked_instr`. Воркер работает напрямую с `mf_instruction`, что упрощает отладку и убирает лишние аллокации.
- [x] **Execution Plan (Register Plan):** Диспетчер готовит массив `mf_cpu_reg_plan` (Buffer, Generator, Scratch) заранее. Воркер лишь следует плану.
- [x] **Builtin ID System:** Замена строк (`host.index`) на перечисления (Builtin IDs) в `mf_program`. Бэкенд перестает парсить строки.
- [x] **Zero-Guess Dispatch:** Бэкенд полностью доверяет страйдам компилятора. Если компилятор ошибся — бэкенд падает, а не пытается "исправить" ситуацию.
- [x] **Kernel Signature Update:** Обновление `mf_op_func` для работы с сырыми инструкциями.

### Phase 2: Compiler Hardening & Reliability
**Goal:** Make the compilation and execution pipeline robust and predictable.

- [x] **Stride Promotion:** Замена `i8` на `i32` в `mf_instruction` для предотвращения переполнений.
- [x] **Explicit Domain Tracking:** Внедрение флага `is_spatial` в IR для надежного вычисления страйдов.
- [x] **Dual Type Masks:** Разделение `type_mask` на `input_mask` и `output_mask` для защиты от некорректного вывода типов.
- [x] **Math Kernel Consolidation:** Рефакторинг `mf_ops_math.c` для удаления дублирования в векторных операциях.
- [x] **Strict Shape Validation:** Добавление прохода валидации, который проверяет совместимость всех узлов в задаче до запуска.
- [x] **Strict DType Propagation:** Исправлен баг перезаписи типа при бродкастинге.
- [x] **Provider-based Indices:** Пространственные индексы реализованы через `Input` узлы.

### Phase 3: Logic Expansion & Infrastructure
**Goal:** Improve developer experience and support complex modular graphs.

- [ ] **Context-Aware Subgraphs:** Проброс геометрии домена внутрь подграфов через узлы `Call`.
- [ ] **String-less IR Ports:** Переход на атомарные ID (хеши) вместо строк внутри компилятора.
- [ ] **Unified Error Context:** Улучшение отчетов о крашах (вывод имен портов и координат ошибки).
- [ ] **Index Buffer Pooling:** Переиспользование буферов для виртуальных ресурсов для экономии памяти.
- [x] **Stride Model Expansion:** Поддержка N-D тензорных шагов для векторных потоков.
- [x] **Host Policy Isolation:** Хост больше не форсирует разрешение для не-пиксельных графов.
- [x] **Heterogeneous Tasks:** Поддержка графов с несколькими выходами разных форм.
- [x] **Explicit Task Geometry API:** Требование явного описания контракта ядра.

---

## Archive (Detailed Task History)

<details>
<summary>Expand for full history of completed tasks</summary>

### Milestone 13: Pure Logic Architecture (Initial Work)
- [x] **ISA & Base Cleanup:** Удаление `MF_OP_INDEX` и переход на чистый `STEP_N`.
- [x] **Virtual Resource Providers:** Входы привязываются к ресурсам с провайдерами.
- [x] **Loader Orchestration:** `mf_loader` собирает геометрию задач.
- [x] **Generator Backend:** Начальная реализация генерации индексов в бэкенде.

### Milestone 12: Intelligence & Performance
- [x] **Auto-Transient Detection:** Темпоральный анализ графа для экономии памяти.
- [x] **Instruction Slimming:** Оптимизация размера байт-кода.
- [x] **Worker Baking:** Минимизация накладных расходов в бэкенде.
- [x] **Native Math Kernels:** SmoothStep, Mix и др. в нативном C.
- [x] **Frame Arena:** Линейная аллокация временных объектов.
- [x] **IR Sanitization:** Гарантированное обнуление структур IR.
- [x] **Absolute Type Inference:** Строгий вывод типов.
- [x] **Port-Aware Resolution:** Поиск источников данных по именам портов.
- [x] **Op Resilience:** Поддержка I32 для генераторов и сохранение identity в бинарном формате.

### Milestone 11: Hardening & Memory Safety
- [x] **Protected Iterators:** Внедрен `end_ptr` в `mf_tensor_iter`.
- [x] **Atomic Kill Switch:** Глобальный флаг ошибки в `mf_engine`.
- [x] **Kernel Crash Report:** Стандартизированный формат вывода ошибок.
- [x] **Stride-Based Instruction Layout:** Массив `strides[4]` в `mf_instruction`.
- [x] **Compiler Stride Inference:** Расчет линейных шагов в `mf_pass_analyze`.
- [x] **Unified Kernel Execution:** Удаление проверок `size > 1` в пользу безусловного инкремента.
- [x] **Register Aliasing Fix:** Отслеживание максимального размера регистра в `mf_codegen.c`.
- [x] **Dynamic Index Context:** Обновление `u_FragX/Y` для каждого элемента.
- [x] **Safe Gather & NaN Protection:** Защита от NaN/Inf и OOB в Gather.
- [x] **Explicit Reduction ISA:** Разделение ADD и REDUCE_SUM.

### Foundation & Earlier
- [x] Core VM (SoA model)
- [x] Manifest-driven Pipeline (.mfapp)
- [x] SDF Rendering engine
- [x] Cross-platform Host
- [x] Instruction Fusion (FMA)
- [x] MEAN decomposition
- [x] Liveness-based Register Allocation (Buffer Aliasing)
- [x] Multi-domain Task System
</details>

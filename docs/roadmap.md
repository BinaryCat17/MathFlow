# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state.

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Active Development

### Milestone 13: Pure Logic Architecture (In Progress)

**Goal:** Achieve complete Inversion of Control (IoC) and graph purity by unifying all external data sources under the `Input` node.

#### Phase 7: Pure Logic & Shape-Driven Compilation
- [x] **ISA & Base Cleanup:** Полное удаление `MF_OP_INDEX`, `MF_NODE_INDEX` и перечисления `mf_identity`. Система переходит на чистый `STEP_N` (strides).
- [x] **Stride Inference Engine:** Переработка `mf_pass_analyze`. Реализация правил автоматического бродкастинга: сопоставление размерностей входа и домена для вычисления оптимальных страйдов.
- [x] **Heterogeneous Tasks:** Поддержка графов с несколькими выходами разных форм. Компилятор автоматически разбивает граф на задачи.
- [ ] **Context-Aware Subgraphs:** Обеспечение проброса геометрии домена внутрь подграфов через узлы `Call`.
- [x] **Explicit Task Geometry API:** Изменение API компилятора. `mf_compile` теперь требует явного описания "Контракта Ядра".
- [x] **Virtual Resource Providers:** Расширение `.mfapp`. Входы привязываются к ресурсам с провайдерами (например, `host.index`).
- [x] **Loader Orchestration:** `mf_loader` собирает геометрию задач из манифеста и передает её компилятору.
- [x] **Generator Backend:** Реализация в `mf_backend_cpu` генерации данных для виртуальных ресурсов (индексов) без аллокации буфера.

#### Phase 8: Architectural Hardening & Stability
- [x] **Strict DType Propagation:** Исправлен баг, при котором бродкастинг форм перезаписывал целевой тип операции (например, в Select).
- [x] **Host Policy Isolation:** Хост больше не форсирует разрешение `out_Color` для не-пиксельных графов.
- [ ] **Dual Type Masks:** Разделение `type_mask` на `input_mask` и `output_mask` для строгой валидации.
- [ ] **In-place Metadata:** Внедрение флага `MF_OP_FLAG_INPLACE_SAFE` для возвращения эффективной аллокации регистров без риска порчи данных в батчах.
- [ ] **String-less IR Ports:** Переход на атомарные ID или хеши для портов и узлов внутри компилятора для повышения надежности инлайнера.
- [ ] **Index Buffer Pooling:** Оптимизация бэкенда — переиспользование буферов для виртуальных ресурсов вместо постоянных аллокаций.
- [ ] **Stride Model Expansion:** Исследование перехода от `i8` линейных страйдов к полной поддержке N-D тензорных шагов для честного бродкастинга.

## Archive (Detailed Task History)

<details>
<summary>Expand for full history of completed tasks</summary>

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

### Milestone 10: Standard Library & ISA Consolidation
- [x] Explicit Import System
- [x] Default Ports mapping
- [x] ISA Category grouping
- [x] XOR and SIZE primitives

### Milestone 9: Advanced Compilation
- [x] Instruction Fusion (FMA)
- [x] MEAN decomposition
- [x] Liveness-based Register Allocation (Buffer Aliasing)
- [x] Multi-domain Task System

### Milestone 8: Compiler Consolidation
- [x] X-Macros metadata
- [x] Automated shape/type propagation

### Milestones 1-7: Foundation
- [x] Core VM (SoA model)
- [x] Manifest-driven Pipeline (.mfapp)
- [x] SDF Rendering engine
- [x] Cross-platform Host
</details>
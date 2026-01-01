# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state.

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Active Development

### Milestone 11: Hardening & Memory Safety (In Progress)

#### Phase 1: Defense & Visibility
- [x] **Protected Iterators:** Внедрить `end_ptr` в `mf_tensor_iter`. Любая попытка доступа за пределы диапазона должна вызывать `MF_LOG_FATAL`. Реализовать проверку как для линейного обхода, так и для случайного доступа.
- [x] **Atomic Kill Switch:** Реализовать глобальный атомарный флаг ошибки в `mf_engine`. При возникновении исключения в любом рабочем потоке бэкенда все остальные потоки должны завершить текущую итерацию и перейти в состояние ожидания.
- [x] **Kernel Crash Report:** Разработать стандартизированный формат вывода ошибки: ID инструкции, мнемоника Opcode, индексы регистров, значения указателей и текущие координаты домена (`Index`).

#### Phase 2: Generalized Stride Contract (STEP_N)
- [x] **Stride-Based Instruction Layout:** Расширить `mf_instruction`, добавив массив `i32 strides[4]` для каждого операнда (Dest, S1, S2, S3). Это позволит бэкенду использовать чистую арифметику указателей.
- [x] **Compiler Stride Inference:** В `mf_pass_analyze` реализовать расчет линейных шагов. Если операнд броадкастится (размер 1 в домене N), его шаг становится `0`. Если он линеен — `1`. Если это сложный вью — `N`.
- [x] **Unified Kernel Execution:** Рефакторинг макросов в `mf_kernel_utils.h`. Удалить проверки `if (size > 1)` и заменить их на безусловный инкремент `ptr += step`.
- [x] **Role Elimination:** Полностью удалить понятия "Spatial", "Uniform" и "Reduction" из бэкенда. Теперь каждый тензор — это просто поток данных с заданным шагом.

#### Phase 3: Validation & Synchronization
- [ ] **Static Binding Validation:** Реализовать проверку совместимости графа и ресурсов ПЕРЕД запуском:
    - Сравнение ожидаемых и фактических `shape` (с учетом правил броадкастинга).
    - Проверка соответствия типов данных (`dtype`).
    - Валидация прав доступа (Read-Only ресурсы не должны биндиться на Output порты).
- [ ] **Host-Compiler Schema Sync:** Автоматизировать передачу метаданных о глобальных ресурсах из `.mfapp` в контекст компилятора для корректного вывода типов.
- [ ] **Thread-Local Safety:** Изолировать временные структуры данных потоков (`scratchpad`). Гарантировать через API, что потоки имеют доступ только к своим локальным аренам памяти.

#### Phase 4: Fault Isolation
- [ ] **Hardened Gather:** Интегрировать обязательную проверку границ в операцию `MF_OP_GATHER`. При выходе индекса за пределы — немедленная активация `Kill Switch`.
- [ ] **Cross-Platform Memory Walls:** Реализовать абстракцию защиты памяти в `mf_platform`:
    - Linux: `mprotect` (PROT_READ).
    - Windows: `VirtualProtect` (PAGE_READONLY).
    Использовать для защиты буферов констант и входных данных от случайной записи.

---

## Completed Phases (Archive)

### Milestone 10: Standard Library & ISA Consolidation (Jan 2026)
- **Explicit Import System:** Added `"imports"` field and search paths (Prelude).
- **Default Ports:** Automated port mapping for single-input/single-output subgraphs.
- **Decomposition:** Moved `Mean`, `Dot`, `Length`, `Normalize`, `Mix`, and `SmoothStep` to JSON.
- **ISA Cleanup:** Grouped instructions into Atomic, Reduction, Accel, and Memory categories.
- **Core Expansion:** Added `XOR` and `SIZE` primitives.

### Milestone 9: Advanced Compilation (Jan 2026)
- **Instruction Fusion:** Implemented `FMA` detection and fusion pass.
- **Advanced Lowering:** Automated `MEAN` decomposition into `SUM/DIV`.
- **Register Allocation:** Implemented **Liveness Analysis** and register reuse (Buffer Aliasing).
- **Task System:** Full support for multi-domain execution and automated task-splitting.

### Milestone 8: Compiler Consolidation (Dec 2025)
- **Metadata-Driven:** Unified operation definitions using X-Macros.
- **Type Inference:** Automated shape/type propagation rules.

### Milestones 1-7: Foundation & Pixel Engine (2024-2025)
- **Core VM:** High-performance interpreted VM with SoA memory model.
- **Pipeline:** Explicit Manifest-driven execution (`.mfapp`).
- **Pixel Math:** SDF-based rendering engine with anti-aliasing.
- **Compiler:** Modular pass-based architecture.
- **Host:** Cross-platform SDL2 and Headless drivers.
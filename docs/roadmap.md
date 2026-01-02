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

#### Phase 3: Architectural Deep Reliability
- [ ] **Explicit Domain ISA:** Перейти от неявных страйдов к явным блокам выполнения с заданным доменом (напр. `Execute Spatial(H, W)`). Убрать `strides` из структуры инструкции.
- [ ] **Typed Accessors:** Заменить `mf_tensor_iter` (голый указатель) на типизированные аксессоры, работающие с N-мерными индексами и имеющие встроенную проверку границ относительно `shape` дескриптора.
- [ ] **Broadcast Identity:** Внедрить в ISA явную маркировку регистров (Constant, Spatial, Uniform, Reduction). Это исключит ошибки при аллокации временной памяти в бэкенде.
- [ ] **Strict Static Analysis:** Реализовать pass валидации скомпилированной программы, который проверяет совместимость типов и доменов инструкций ПЕРЕД отправкой на выполнение.

#### Phase 4: Introspection & Debugging
- [ ] **Reference Interpreter:** Создать эталонный однопоточный бэкенд-интерпретатор, который делает агрессивные проверки границ на каждом чтении/записи (Asan-style).
- [ ] **State Inspector:** Реализовать механизм дампа состояния регистров и графа в JSON/Image при активации Kill Switch.
- [ ] **Visual Debugger Bridge:** Подготовить API для передачи состояния выполнения во внешний визуализатор графа.

### Debug Notes (Milestone 11)
*   **text_demo status:** Крашится на первом кадре (Инструкция #20: `GATHER`).
*   **Ошибка:** `OUT_OF_BOUNDS`, индекс `-2147483648`. Это результат `(int)NaN`.
*   **Локация:** Подграф `render_text.json`. Регистры `D:21, S1:14, S2:21`. 
*   **Контекст:** Проблема в расчете `charcode` или `base_idx`. Нужно проверить `Clamp` узлы в `render_text.json`, возможно они не спасают от `NaN`.
*   **Инфраструктура:** Исправлены баги в `mf_tensor_iter` (скаляры больше не «шагают») и `backend_cpu` (темпы теперь всегда имеют размер батча).

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
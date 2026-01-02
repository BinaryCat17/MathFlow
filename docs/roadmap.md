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

#### Phase 3: Reliability & Stabilization (Critical Fixes)
- [x] **Register Aliasing Fix:** Исправить `mf_codegen.c`, внедрив отслеживание *максимального* требуемого размера для каждого регистра. Дескриптор в `mf_program` должен гарантировать безопасность при переиспользовании регистров (защита от Heap Corruption).
- [x] **Dynamic Index Context:** Исправить `mf_backend_cpu.c`, обеспечив обновление координат `u_FragX`, `u_FragY` для *каждого* элемента внутри чанка. Это уберет артефакты «одноцветных квадратов».
- [x] **Safe Gather & NaN Protection:** Внедрить в `mf_ops_array.c` и `mf_ops_math.c` защиту от NaN/Inf. Операция `Gather` должна проверять индекс до каста к целому числу и возвращать безопасный дефолт при некорректных значениях.
- [x] **Explicit Reduction ISA:** Разделить `MF_OP_SUM` на `MF_OP_ADD` (поэлементное) и `MF_OP_REDUCE_SUM` (схлопывание в скаляр). Убрать «магические» эвристики из бэкенда.

#### Phase 4: Architectural Deep Reliability
- [ ] **Typed Accessors (Checked Mode):** Заменить `mf_tensor_iter` на типизированные аксессоры. Добавить "Checked Mode" (включаемый при компиляции), который делает строгую проверку границ (bounds checking) на каждом чтении/записи.
- [x] **Strict Static Analysis:** Реализовать pass валидации скомпилированной программы, который проверяет совместимость типов и доменов инструкций ПЕРЕД отправкой на выполнение.
- [x] **Broadcast Identity:** Внедрить в ISA явную маркировку регистров (Constant, Spatial, Uniform). Это исключит ошибки при аллокации временной памяти в бэкенде.

### Milestone 12: Intelligence & Performance

#### Phase 5: Automatic Optimization & Slimming
- [ ] **Auto-Transient Detection:** Внедрить в `mf_engine` темпоральный анализ графа. Если ресурс не имеет зависимости от предыдущего кадра, он автоматически помечается как `Transient` и использует только один буфер вместо двух (экономия памяти и устранение лага в 1 кадр).
- [ ] **Instruction Slimming:** Убрать `strides` из структуры `mf_instruction`. Использовать `Identity` регистра для определения шага в рантайме. Это уменьшит размер байт-кода в 2 раза и улучшит Cache Locality.
- [ ] **Worker Baking:** Оптимизировать горячий цикл бэкенда. Перед запуском задачи подготавливать плоский массив указателей и шагов для всех операндов, чтобы минимизировать накладные расходы внутри `cpu_worker_job`.
- [ ] **Native Math Kernels:** Перенести `SmoothStep`, `Mix`, `Length`, `Normalize` и `Dot` из JSON-библиотек обратно в нативный C (`mf_ops_math.c`). Это даст прирост производительности в 10-50 раз для графических задач.
- [ ] **Frame Arena:** Полностью перевести аллокацию временных объектов в рантайме на линейную арену, которая сбрасывается раз в кадр.

#### Phase 6: Introspection & Debugging
- [ ] **Reference Interpreter:** Создать эталонный однопоточный бэкенд-интерпретатор для отладки сложной математики.
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
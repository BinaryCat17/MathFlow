# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state.

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Current Status: Performance & Reliability Hardening

### Phase 1: Backend Refactoring (Completed)
- [x] **Remove Instruction Baking:** Воркер работает напрямую с `mf_instruction`.
- [x] **Execution Plan (Register Plan):** Диспетчер готовит `mf_cpu_reg_plan` заранее.
- [x] **Builtin ID System:** Замена строк на перечисления (Builtin IDs) в `mf_program`.
- [x] **Zero-Guess Dispatch:** Бэкенд полностью доверяет страйдам компилятора.

### Phase 2: Compiler Hardening & Reliability (Completed)
- [x] **Stride Promotion:** Использование `i32` для страйдов.
- [x] **Explicit Domain Tracking:** Внедрение флага `is_spatial`.
- [x] **Dual Type Masks:** Разделение `input_mask` и `output_mask`.
- [x] **Strict Shape Validation:** Валидация совместимости узлов до запуска.

### Phase 3: Logic Expansion & Infrastructure (Completed)
- [x] **Domain Inheritance in Subgraphs:** Автоматическое наследование геометрии `Index`.
- [x] **Index Shape Auto-Inference:** Узлы `Index` принимают форму домена.
- [x] **Unified Error Context:** Улучшение отчетов о крашах.
- [x] **Centralized Provider Registry:** Унификация `host.*` провайдеров.

---

## Active Development

### Phase 4: Zero-Overhead Execution (Completed)
**Goal:** Полный отказ от тяжелых объектов (`mf_tensor`) и итераторов в горячих циклах исполнения.

- [x] **Flat Execution Registry:** Внедрен `reg_ptrs` и `reg_info` в `mf_exec_ctx`. Ядра больше не знают о `mf_tensor`.
- [x] **ISA Expansion (v4):** Увеличение количества страйдов в инструкции до 8 для поддержки сложных операций (`Join`, `Gather`).
- [x] **Precalculated Byte Strides:** Вычисление байтовых шагов в диспетчере (один раз на задачу).
- [x] **Linear Kernel Refactoring:** Все базовые ядра переведены на сырые указатели.
- [x] **Vector/Special Kernel Refactoring:** `Gather`, `Join`, `Reduce` избавлены от `mf_tensor_iter`.
- [x] **Universal Pre-dispatch Allocation:** Диспетчер гарантирует аллокацию всех регистров (вкл. скаляры) до запуска воркеров.

### Phase 5: Architectural Cleanup & Parallel Ops (Completed)
**Goal:** Удаление легаси-кода, упрощение макросов и внедрение быстрых параллельных алгоритмов.

- [x] **Delete Legacy Iterators:** Полное удаление `mf_tensor_iter.h` и `mf_accessor.h`.
- [x] **Kernel Macro Simplification:** Очистка `mf_kernel_utils.h` от неиспользуемых параметров (`ACC_IN`, `ACC_OUT`).
- [x] **Decouple Kernel Helpers:** Перевод хелперов разрешения форм в `mf_kernel_utils.h` с `mf_tensor` на `mf_type_info` (неиспользуемые удалены).
- [x] **Internal Macros Cleanup:** Очистка `mf_ops_internal.h` от проверок, которые теперь выполняет компилятор.
- [x] **Parallel Sync Engine:** Реализация барьерного исполнения в диспетчере и двухпроходного параллельного `CumSum`.
- [x] **Batched MatMul:** Исправлена поддержка батчей в матричном умножении.

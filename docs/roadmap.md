# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state.

---

## Current Status: Performance & Reliability Hardening

### Phase 1: Backend Refactoring (Completed)
- [x] **Remove Instruction Baking**
- [x] **Execution Plan (Register Plan)**
- [x] **Builtin ID System**
- [x] **Zero-Guess Dispatch**

### Phase 2: Compiler Hardening & Reliability (Completed)
- [x] **Stride Promotion**
- [x] **Explicit Domain Tracking**
- [x] **Dual Type Masks**
- [x] **Strict Shape Validation**

### Phase 3: Logic Expansion & Infrastructure (Completed)
- [x] **Domain Inheritance in Subgraphs**
- [x] **Index Shape Auto-Inference**
- [x] **Unified Error Context**
- [x] **Centralized Provider Registry**

---

## Active Development: Consolidation & Purity

### Phase 6: Architectural Consolidation (The 'Shrink' Phase) (Completed)
**Goal:** Сокращение объема кода при сохранении функционала. Борьба с дублированием данных и логики.
- [x] **Unified Operation Registry (X-Macros):** Создан `mf_ops_db.inc`. Удалены дубликаты в `isa` и `compiler`.
- [x] **Generic Kernel Patterns:** Унификация циклов через `MF_KERNEL_*`. Удалена ручная регистрация в модулях `ops`.
- [x] **Loader Unification:** Слияние `mf_loader` и `mf_manifest_loader`. Удалены лишние файлы.
- [x] **Deep Tensor Decoupling:** Компилятор и `mf_program` переведены на `mf_type_info`. `mf_tensor` используется только в рантайме.
- [x] **Backend Encapsulation:** Бэкенд изолирован через `mf_task`.

### Phase 7: Quality of Life & Extreme Shrink (In Progress)
**Goal:** Стандартизация, подготовка к векторизации и финальное сокращение boilerplate-кода.
- [x] **Extreme Kernel Unification:** 80% ядер (atomic ops) генерируются автоматически через один X-Macro в `mf_ops_db.inc`. Математика (`+`, `-`, `*`, `/`) и функции (`sinf`, `cosf`) перенесены в БД операций.
- [x] **Standardize Opcode Naming:** Убраны алиасы, унифицированы имена портов (`a`, `b`, `c`, `d`, `x`, `cond`, `true`, `false`).
- [x] **Unified Compiler Passes:** `mf_pass_validate`, `mf_pass_analyze` и `mf_codegen` переведены на универсальные циклы по портам. Удалено дублирование логики `s1/s2/s3/s4`.
- [x] **Improved Metadata:** В структуру `mf_op_metadata` добавлена арность (`arity`), что позволило автоматизировать проверку подключений.
- [ ] **Documentation Sync:** Обновить `docs/architecture.md` и `docs/guide_app_creation.md` согласно изменениям Фаз 6 и 7.

## Phase 8: Robustness & Compiler Maturity (Hardening) (In Progress)
**Goal:** Превращение компилятора в надежный инструмент с вменяемой диагностикой и чистой архитектурой.
- [x] **Total Diagnostic Coverage:** Искоренить все "тихие падения" (`return false` без репорта). Каждый сбой должен иметь текстовое описание и локацию в коде.
- [x] **Metadata Simplification (Wrappers):** Внедрить семантические макросы-обертки (например, `MF_MATH_BIN`) внутри `mf_ops_db.inc`. Это уберет визуальный шум (`NULL, MANUAL`) и исключит ошибки в позиционных аргументах.
- [x] **Rank Normalization Pass:** Внедрить автоматическое приведение рангов. Если операция ожидает скаляр `[]`, а получает вектор `[1]`, компилятор должен выполнять неявный *Squeeze*.
- [x] **Formal Broadcasting Logic:** Довести `mf_shape.c` до полной поддержки правил NumPy (правильное выравнивание осей по правому краю), сохраняя при этом различие между `[]` (rank 0) и `[1]` (rank 1).
- [ ] **Contract & Loader Decoupling (The "Clean Split"):**
    - **Autonomous Compiler:** Избавить компилятор от зависимости от `mf_compile_contract`. Он должен компилировать граф на основе типов/форм, указанных внутри самих узлов `Input/Output`.
    - **Dynamic Resolver in Engine:** Перенести логику сопоставления портов ядра с ресурсами манифеста из `mf_loader.c` в `mf_engine`. Движок должен сам находить нужные регистры в `mf_program` по таблице символов.
    - **Pure IO Loader:** Лоадер должен только читать файлы и парсить базовые структуры, не вмешиваясь в логику сопоставления типов и не вызывая компилятор «вручную» с хитрыми параметрами.
- [x] **Zero-Allocation Dispatch:** Перенести аллокации `reduction_scratch` и `sync_scratch` из `dispatch` в `bake`, чтобы рантайм был полностью свободен от `malloc/calloc`.
- [ ] **Automated Regression Suite:** Реализовать инструмент для массовой проверки всех существующих проектов и тестов на корректность компиляции.

## Phase 9: Structural Decoupling (Modularization)
**Goal:** Полная архитектурная изоляция компилятора от рантайма. Подготовка к разделению на два независимых проекта.
- [ ] **Header Sanitization:** Запретить модулю `compiler` импортировать что-либо из `engine` и `host`. Единственная точка соприкосновения — `ISA`.
- [ ] **Opaque Binary Interface:** Рантайм должен принимать только анонимный указатель на программу или путь к `.bin` файлу. Никаких "прямых" вызовов компилятора внутри `mf_engine`.
- [ ] **Standalone Compiler CLI:** Вынести логику компиляции в отдельную утилиту `mfc`, которая превращает `.json` в `.bin` независимо от основного приложения.
- [ ] **ISA as a Shared Library:** Выделить `base` и `isa` в минимальный набор заголовочных файлов, который распространяется как SDK.

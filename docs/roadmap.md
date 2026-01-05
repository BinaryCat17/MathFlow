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




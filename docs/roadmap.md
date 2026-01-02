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

### Phase 7: Quality of Life & SIMD Preparation (Planned)
**Goal:** Стандартизация именования и подготовка к векторизации.

- [ ] **Standardize Opcode Naming:** Привести все имена в `mf_ops_db.inc` к единому стандарту (например, убрать алиасы `SUM`/`REDUCE_SUM`).
- [ ] **Unified Error Reporting Macro:** Использовать метаданные из X-Macros для вывода имен портов в `report_crash`.
- [ ] **SIMD Hook-up:** Создание параллельной таблицы ядер для SIMD-оптимизированных операций.
- [ ] **Documentation Sync:** Обновить `docs/architecture.md` и `docs/guide_app_creation.md` согласно изменениям Фазы 6.
- [ ] **Test Coverage:** Добавить unit-тесты для нового унифицированного загрузчика.
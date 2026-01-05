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
- [x] **Documentation Sync:** Обновить `docs/architecture.md` и `docs/guide_app_creation.md` согласно изменениям Фаз 6 и 7.

## Phase 8: Robustness & Compiler Maturity (Hardening) (Completed)
**Goal:** Превращение компилятора в надежный инструмент с вменяемой диагностикой и чистой архитектурой.
- [x] **Total Diagnostic Coverage:** Искоренить все "тихие падения" (`return false` без репорта). Каждый сбой должен иметь текстовое описание и локацию в коде.
- [x] **Metadata Simplification (Wrappers):** Внедрить семантические макросы-обертки (например, `MF_MATH_BIN`) внутри `mf_ops_db.inc`. Это уберет визуальный шум (`NULL, MANUAL`) и исключит ошибки в позиционных аргументах.
- [x] **Rank Normalization Pass:** Внедрить автоматическое приведение рангов. Если операция ожидает скаляр `[]`, а получает вектор `[1]`, компилятор должен выполнять неявный *Squeeze*.
- [x] **Formal Broadcasting Logic:** Довести `mf_shape.c` до полной поддержки правил NumPy (правильное выравнивание осей по правому краю), сохраняя при этом различие между `[]` (rank 0) и `[1]` (rank 1).
- [x] **Contract & Loader Decoupling (The "Clean Split"):**
    - [x] **Autonomous Compiler:** Избавить компилятор от зависимости от `mf_compile_contract`. Он должен компилировать граф на основе типов/форм, указанных внутри самих узлов `Input/Output`.
    - [x] **Dynamic Resolver in Engine:** Перенести логику сопоставления портов ядра с ресурсами манифеста из `mf_loader.c` в `mf_engine`. Движок должен сам находить нужные регистры в `mf_program` по таблице символов.
    - [x] **Pure IO Loader:** Лоадер должен только читать файлы и парсить базовые структуры, не вмешиваясь в логику сопоставления типов и не вызывая компилятор «вручную» с хитрыми параметрами.
- [x] **Zero-Allocation Dispatch:** Перенести аллокации `reduction_scratch` и `sync_scratch` из `dispatch` в `bake`, чтобы рантайм был полностью свободен от `malloc/calloc`.

## Phase 8.5: Backend Decoupling & Brain Shift (Optimization) (Completed)
**Goal:** Перенос высокоуровневой логики из бэкенда в компилятор. Превращение бэкенда в "чистый исполнитель" без знания о конкретных опкодах.
- [x] **Compiler-Driven Segmentation:** Полностью перенести логику разделения на сегменты (барьеры) в кодогенерацию. Бэкенд получает готовый список `mf_task` и не пересчитывает их.
- [x] **Deterministic Scratchpad Sizing:** Компилятор высчитывает требуемый размер `sync_scratch` и `reduction_scratch` на этапе кодогенерации и записывает его в `mf_bin_header`.
- [x] **Dispatch Strategy Metadata:** Хардкод `if (opcode == CUMSUM)` заменен на декларативные стратегии в `mf_op_metadata`.
- [x] **Zero-Allocation Dispatch:** Рантайм полностью свободен от `malloc/calloc` в горячем цикле.
- [x] **Pre-Analyzed Liveness in Binary:** Перенести расчет списка `active_regs` для каждой задачи в компилятор. Бэкенд не должен итерироваться по инструкциям в `bake`.
- [x] **Rich Tensor Flags:** Добавить флаги (`REDUCTION`, `GENERATOR`, `PERSISTENT`) в `mf_bin_tensor_desc`, чтобы убрать анализ зависимостей из бэкенда.
- [x] **Register Binding Optimization:** Упростить `prepare_registers` в бэкенде, избавившись от лишних проверок и `switch`.
- [x] **Instruction Shrink (The 'Thin' VM):** Удалить массив `strides[5]` из `mf_instruction`. Уменьшить размер структуры до ~16 байт для резкого снижения давления на L1 Instruction Cache.
- [x] **Stride Promotion to Task Metadata:** Перенести хранение страйдов в метаданные привязки регистров внутри задачи (`mf_task`).
- [x] **Byte-Stride Pre-calculation:** Компилятор должен сразу вычислять байтовые смещения (`stride * sizeof(dtype)`), избавляя бэкенд от привязки к размерам типов.
- [ ] **Task Specialization (Fast Path):** Внедрить флаги задач (например, `MF_TASK_CONTIGUOUS`). Бэкенд должен использовать SIMD или `memcpy` для полностью непрерывных данных без проверки страйдов в цикле.
- [ ] **Linear Access Simplification:** Упростить `cpu_worker_job`, чтобы для линейных задач не выполнялась дорогостоящая логика развертки N-мерных индексов (`tile_offset`).

## Phase 9: Application Packaging & Fat Binary (The "Cartridge" Model) (Next)

- [ ] **Unified App Header:** Расширить `mf_bin_header`, добавив туда параметры хоста (Window Title, Width, Height, VSync) и настройки рантайма (количество потоков).
- [ ] **Pipeline Bundling:** Научить формат бинарника хранить несколько программ (ядер) и описание их связей (Pipeline) внутри одного файла.
- [ ] **Global Resource Declarations:** Перенести описание глобальных ресурсов из манифеста в бинарник. Рантайм должен просто выделять память согласно списку из заголовка.
- [ ] **Standalone Compiler CLI (`mfc`):** Создать утилиту, которая принимает `.mfapp` и все связанные `.json`, проводит полную валидацию и «запекает» их в один `.bin`.
- [ ] **Zero-JSON Runtime:** Полностью удалить `mf_loader.c` (парсинг JSON) из финальной сборки рантайма. Движок должен принимать только `mf_program*` или путь к `.bin`.
- [ ] **Asset Embedding (Optional):** Исследовать возможность внедрения ассетов (иконок, шрифтов) прямо в бинарный пакет для создания 100% автономных приложений.

## Phase 10: Vectorization & High-Level DSL

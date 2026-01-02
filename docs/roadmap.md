# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state.

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Active Development

### Milestone 13: Pure Logic Architecture

#### Phase 7: Logic Purity & Intrinsic Inputs
- [ ] **Index Node Elimination:** Полное удаление `MF_NODE_INDEX` из компилятора и `MF_OP_INDEX` из ISA. Перевод всех существующих графов на использование узлов `Input`.
- [ ] **Intrinsic Naming Convention:** Внедрение соглашения об именах для автоматического определения `SPATIAL` идентичности. Порты `u_FragX`, `u_FragY`, `u_FragZ` автоматически получают шаг 1 и привязываются к генераторам координат.
- [ ] **Manifest-Driven Generators:** Расширение формата `.mfapp`. Возможность явно указать тип генератора для входного порта (например, `generator: "index", axis: 0`).
- [ ] **Loader Synthesis:** Доработка `mf_loader.c` для автоматического создания виртуальных ресурсов-генераторов, если граф требует `u_Frag...` входы, но они не описаны в манифесте.
- [ ] **Subgraph Universality:** Обеспечение полной поддержки любых "координатных" входов в подграфах без нарушения их чистоты (purity).

### Milestone 13: Pure Logic Architecture

#### Phase 7: Logic Purity & Intrinsic Inputs
- [ ] **Index Node Elimination:** Полное удаление `MF_NODE_INDEX` из компилятора и `MF_OP_INDEX` из ISA. Перевод всех существующих графов на использование узлов `Input`.
- [ ] **Intrinsic Naming Convention:** Внедрение соглашения об именах для автоматического определения `SPATIAL` идентичности. Порты `u_FragX`, `u_FragY`, `u_FragZ` автоматически получают шаг 1 и привязываются к генераторам координат.
- [ ] **Manifest-Driven Generators:** Расширение формата `.mfapp`. Возможность явно указать тип генератора для входного порта (например, `generator: "index", axis: 0`).
- [ ] **Loader Synthesis:** Доработка `mf_loader.c` для автоматического создания виртуальных ресурсов-генераторов, если граф требует `u_Frag...` входы, но они не описаны в манифесте.
- [ ] **Subgraph Universality:** Обеспечение полной поддержки любых "координатных" входов в подграфах без нарушения их чистоты (purity).

---

## Completed Milestones (Summary)

*   **Milestone 12: Intelligence & Performance:** Автоматическая детекция транзиентных ресурсов, оптимизация размера инструкций, ускорение горячего цикла бэкенда и внедрение нативных математических ядер. Стабилизация системы типов и переход на Port-Aware разрешение.
*   **Milestone 11: Hardening & Memory Safety:** Внедрены защищенные итераторы, Atomic Kill Switch и детальные отчеты об ошибках в ядрах. Реализована модель STEP_N.
---
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
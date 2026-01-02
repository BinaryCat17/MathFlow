Проведя детальный анализ проекта, я вижу классическую архитектурную проблему: **«Умный Бэкенд» (Smart Backend Anti-pattern)**.

Модуль mf\_cpu\_backend.c пытается компенсировать недостатки компилятора и брать на себя задачи движка (генерацию данных), что привело к сильной связности и невозможности отладки.

Вот подробный анализ проблем и конкретный план действий по упрощению.

### ---

**1\. Анализ проблем mf\_cpu\_backend.c**

#### **А. Функция-монстр prepare\_registers**

Эта функция сейчас выполняет три несовместимые задачи внутри **горячего цикла** (в каждом потоке, для каждого чанка):

1. **Парсинг строк:** strncmp(provider, "host.index", 10\) выполняется тысячи раз. Бэкенд исполнения вообще не должен знать строковых имен — это задача этапа подготовки.  
2. **Эвристика памяти:** Она на лету решает, аллоцировать ли память (scratch) или брать существующую (buffer).  
3. **Процедурная генерация:** Она содержит математику генерации индексов. Если вы захотите добавить host.time или host.mouse, вам придется переписывать этот файл.

#### **Б. Фаза "Baking" (Запекание инструкций)**

В функции cpu\_worker\_job (строка \~300) создается массив mf\_cpu\_baked\_instr.

* **Диагноз:** Это преждевременная оптимизация. Код копирует инструкции и заменяет индексы регистров на прямые указатели. Современные CPU отлично справляются с двойной индексацией (ctx-\>registers\[inst-\>src1\]), а "запекание" удваивает потребление памяти, усложняет код воркера и делает невозможной нормальную отладку (в отладчике вы видите "baked" структуру, оторванную от контекста программы).

#### **В. Недоверие к Компилятору (Strides Inference)**

В mf\_backend\_cpu\_dispatch (строки \~380-450) есть цикл, который пытается "угадать" страйды, игнорируя то, что насчитал компилятор.

* **Симптом:** Бэкенд проверяет размеры буферов и меняет stride с 0 на 1 или наоборот.  
* **Риск:** Это скрывает баги компилятора. Бэкенд должен быть "тупым" исполнителем: если компилятор сказал "Stride 0" (скаляр), а буфер огромный — это проблема компилятора, а не повод для магии в бэкенде.

### ---

**2\. План Рефакторинга**

Я предлагаю паттерн **"Execution Plan"**. Мы разделим работу на две фазы: "Планирование" (один раз перед запуском потоков) и "Исполнение" (много раз в потоках).

#### **Шаг 1: Удаление Baking (Самое простое)**

Полностью удалите структуру mf\_cpu\_baked\_instr.  
В mf\_ops\_core.h измените сигнатуру операций, чтобы они принимали сырую инструкцию:

C

// Было  
typedef void (\*mf\_op\_func)(mf\_exec\_ctx\* ctx, const mf\_cpu\_baked\_instr\* instr);  
// Стало  
typedef void (\*mf\_op\_func)(mf\_exec\_ctx\* ctx, const mf\_instruction\* instr);

В цикле исполнения:

C

const mf\_instruction\* inst \= \&batch-\>program-\>code\[pc\];  
batch-\>op\_table\[inst-\>opcode\](\&state-\>ctx, inst);

#### **Шаг 2: Введение Плана Регистров**

Вместо того чтобы воркер анализировал mf\_program и парсил строки, Диспетчер должен подготовить простой план.

Добавьте в начало mf\_cpu\_backend.c:

C

typedef enum {  
    MF\_SRC\_BUFFER,    // Данные лежат в глобальном буфере  
    MF\_SRC\_GENERATOR, // Данные нужно сгенерировать (host.index)  
    MF\_SRC\_SCRATCH    // Временный буфер (нужна аллокация в worker arena)  
} mf\_reg\_source\_type;

// Инструкция для воркера: "Откуда взять данные для регистра N"  
typedef struct {  
    mf\_reg\_source\_type type;  
      
    // Для BUFFER  
    mf\_buffer\* buffer;  
    size\_t base\_offset;  
    size\_t stride\_bytes; // Заранее вычисленный шаг в байтах

    // Для GENERATOR  
    int gen\_axis;        // Параметр генерации  
      
    // Метаданные (копия из Program, чтобы не лазить по указателям)  
    mf\_type\_info info;     
} mf\_cpu\_reg\_plan;

// Добавить в mf\_cpu\_parallel\_batch  
mf\_cpu\_reg\_plan plans\[MF\_MAX\_REGISTERS\]; 

#### **Шаг 3: Очистка dispatch (Планирование)**

Перепишите mf\_backend\_cpu\_dispatch. Он должен заполнить массив plans.

C

// Псевдокод внутри dispatch  
for (u32 i \= 0; i \< program-\>meta.tensor\_count; \++i) {  
    mf\_cpu\_reg\_plan\* p \= \&batch.plans\[i\];  
      
    // 1\. Если это Провайдер (host.index)  
    if (batch.providers\[i\]) {  
        p-\>type \= MF\_SRC\_GENERATOR;  
        // Здесь один раз парсим строку или (лучше) проверяем Enum  
        p-\>gen\_axis \= parse\_axis(batch.providers\[i\]);   
        continue;  
    }

    // 2\. Если есть глобальный буфер  
    if (main\_state-\>registers\[i\].buffer) {  
        p-\>type \= MF\_SRC\_BUFFER;  
        p-\>buffer \= main\_state-\>registers\[i\].buffer;  
        // Строго верим страйду из инструкции\!  
        p-\>stride\_bytes \= (stride \> 0) ? mf\_dtype\_size(...) : 0;  
        continue;  
    }

    // 3\. Иначе Scratch  
    p-\>type \= MF\_SRC\_SCRATCH;  
}

#### **Шаг 4: Новый prepare\_registers (Исполнение)**

Теперь функция воркера становится линейной и тривиальной.

C

static void prepare\_registers(mf\_exec\_ctx\* ctx, const mf\_cpu\_parallel\_batch\* batch, u32 job\_offset, u32 count) {  
    for (u32 i \= 0; i \< batch-\>active\_reg\_count; \++i) {  
        u16 r \= batch-\>active\_regs\[i\];  
        const mf\_cpu\_reg\_plan\* plan \= \&batch-\>plans\[r\];  
        mf\_tensor\* t \= \&ctx-\>registers\[r\];

        t-\>info \= plan-\>info; // Быстрая копия метаданных

        switch (plan-\>type) {  
            case MF\_SRC\_BUFFER:  
                t-\>buffer \= plan-\>buffer;  
                // Никаких проверок\! Просто арифметика.  
                t-\>byte\_offset \= plan-\>base\_offset \+ (job\_offset \* plan-\>stride\_bytes);  
                break;

            case MF\_SRC\_SCRATCH:  
                // Просто аллокация, без логики  
                t-\>buffer \= alloc\_scratch\_buffer(ctx, count \* element\_size);  
                break;  
                  
            case MF\_SRC\_GENERATOR:  
                // Вызов специализированной функции БЕЗ парсинга  
                void\* data \= alloc\_scratch\_data(ctx, count \* sizeof(f32));  
                mf\_generate\_index\_chunk(data, count, job\_offset, plan-\>gen\_axis);  
                t-\>buffer \= wrap\_buffer(data);  
                break;  
        }  
    }  
}

### **3\. Рекомендация по Генераторам**

Вам нужно убрать зависимость от строк вида "host.index".

1. В mf\_program (или mf\_symbol) добавьте поле u32 flags или u16 builtin\_id.  
2. Компилятор при парсинге JSON должен распознавать provider: "host.index" и ставить флаг MF\_BUILTIN\_INDEX.  
3. Бэкенд проверяет флаг, а не делает strncmp.

### **Итог**

Применив этот план:

1. Вы удалите \~30% кода из бэкенда.  
2. **prepare\_registers** перестанет быть узким местом.  
3. Ошибки памяти станут понятными (четкое разделение Buffer vs Scratch).  
4. Отладка станет возможной, так как вы будете видеть реальные инструкции, а не "запеченные" структуры.
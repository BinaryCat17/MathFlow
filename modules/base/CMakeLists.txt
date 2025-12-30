add_library(mf_base STATIC
    include/mathflow/base/mf_utils.h
    include/mathflow/base/mf_buffer.h
    src/mf_log.c
    src/mf_memory.c
    src/mf_platform.c
    src/mf_thread_pool.c
    src/mf_utils.c
    src/mf_buffer.c
)
add_library(MathFlow::base ALIAS mf_base)

target_include_directories(mf_base 
    PUBLIC include
    PRIVATE src
)

find_package(Threads REQUIRED)
target_link_libraries(mf_base PUBLIC Threads::Threads)

if(UNIX)
    # Math library is often needed for low-level ops if used in base
    target_link_libraries(mf_base PRIVATE m)
endif()

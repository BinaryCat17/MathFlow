add_executable(mf-runner 
    src/main.c
)

target_include_directories(mf-runner PRIVATE 
    # No extra include needed, it comes from MathFlow::host_core interface
)

target_link_libraries(mf-runner 
    PRIVATE 
        MathFlow::host_core
        MathFlow::base
)

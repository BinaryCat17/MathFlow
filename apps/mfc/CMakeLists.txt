add_executable(mfc src/main.c)

target_link_libraries(mfc PRIVATE 
    mf_compiler
    mf_isa
    mf_base
)

target_include_directories(mfc PRIVATE src)

install(TARGETS mfc DESTINATION bin)

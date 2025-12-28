cmake_minimum_required(VERSION 3.25)

project(mf-window)

if(TARGET mf_host)
    add_executable(mf-window src/main.c)

    target_link_libraries(mf-window 
        PRIVATE 
            mf_host
    )
else()
    message(STATUS "Skipping mf-window (mf_host not available)")
endif()
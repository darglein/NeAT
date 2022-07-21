function(add_app TARGET_NAME)

    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    target_include_directories(${TARGET_NAME} PUBLIC ".")
    target_link_libraries(${TARGET_NAME} ${LIBS} neat)
    target_include_directories(${TARGET_NAME} PUBLIC "." ${MY_BUILDCONFIG_DIR})

    if (PR_NO_WINDOW)
        target_compile_definitions(${TARGET_NAME} PUBLIC PR_NO_WINDOW)
    endif ()


    message(STATUS "App enabled:       ${TARGET_NAME}")
endfunction()

add_executable(nikon2neat nikon2neat.cpp)
add_app(nikon2neat)

add_executable(reconstruct reconstruct.cpp)
add_app(reconstruct)


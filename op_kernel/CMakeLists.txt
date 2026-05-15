# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0 --cce-ignore-always-inline=true)
endif()

# msOpGen official style: add OpName and kernel file.
# The kernel entry function in complex_gram.cpp is `complex_gram`.
add_kernel_compile(ComplexGram ${CMAKE_CURRENT_SOURCE_DIR}/complex_gram.cpp)

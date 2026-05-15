cmake_minimum_required(VERSION 3.5.1)
project(complex_gram_caller)
add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./")

set(INC_PATH $ENV{ASCEND_HOME_PATH})
if ("${INC_PATH}" STREQUAL "")
    set(INC_PATH $ENV{ASCEND_HOME})
endif()
set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
set(LIB_PATH "${INC_PATH}/lib64")

include_directories(
    ${INC_PATH}/include
    ${CUST_PKG_PATH}/include
)
link_directories(
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_complex_gram_op main.cpp)
target_link_libraries(execute_complex_gram_op
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    stdc++
)

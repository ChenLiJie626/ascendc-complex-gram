message(STATUS "TILING SINK TASK BEGIN")
message(STATUS "TARGET: ${TARGET}")
message(STATUS "OPTION: ${OPTION}")
message(STATUS "SRC: ${SRC}")
message(STATUS "VENDOR: ${VENDOR_NAME}")

set(CMAKE_CXX_COMPILER ${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
set(CMAKE_C_COMPILER ${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc)

add_library(${TARGET} ${OPTION}
    ${SRC}
)
target_compile_definitions(${TARGET} PRIVATE DEVICE_IMPL_OP_OPTILING)
target_include_directories(${TARGET} PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
target_compile_options( ${TARGET} PRIVATE
    -fvisibility=hidden
)
target_link_libraries(${TARGET} PRIVATE
    -Wl,--whole-archive
    device_register
    -Wl,--no-whole-archive
)
target_link_directories(${TARGET} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME cust_opmaster
)
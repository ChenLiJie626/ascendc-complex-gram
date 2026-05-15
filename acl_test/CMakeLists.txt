cmake_minimum_required(VERSION 3.16)
project(complex_gram_acl_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(complex_gram_acl_run main.cpp)
# In the real Ascend environment add ACL includes/libs, for example:
# target_include_directories(complex_gram_acl_run PRIVATE $ENV{ASCEND_HOME}/include)
# target_link_libraries(complex_gram_acl_run ascendcl)

cmake_minimum_required(VERSION 3.1)
project(pv021-neural-network)

# Configure compiler warnings
if (CMAKE_CXX_COMPILER_ID MATCHES Clang OR ${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
  # using regular Clang or AppleClang or GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -pedantic")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
  # using Visual Studio C++
  set(CMAKE_CXX_FLAGS "/permissive- /W4 /EHsc")
  add_definitions("-D_CRT_SECURE_NO_WARNINGS")
endif()

# set(SOURCES src/matrix.cpp)
add_executable(main_run main.cpp)

# file(GLOB TEST_SOURCES test*.cpp)
# add_executable(hw01_tests ${TEST_SOURCES} ${SOURCES})
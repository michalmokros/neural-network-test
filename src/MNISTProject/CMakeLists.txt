cmake_minimum_required(VERSION 3.0)
project(vyuka)

if (CMAKE_CXX_COMPILER_ID MATCHES Clang OR ${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
  # using regular Clang or AppleClang or GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -pedantic -Wold-style-cast")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
  # using Visual Studio C++
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std=c++17 /W4 /WX")
  add_definitions("-D_CRT_SECURE_NO_WARNINGS")
endif()

file( GLOB APP_SOURCES *.cpp )
foreach( source_file ${APP_SOURCES} )
    # I used a simple string replace, to cut off .cpp.
	get_filename_component(exec_name ${source_file} NAME_WE)
    add_executable( ${exec_name} ${source_file} )
    # Make sure YourLib is linked to each app
endforeach( source_file ${APP_SOURCES} )
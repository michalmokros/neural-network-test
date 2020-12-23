cmake_minimum_required(VERSION 3.0)
project(neural-network)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -pedantic -Wold-style-cast")

file( GLOB APP_SOURCES *.cpp )
foreach( source_file ${APP_SOURCES} )
    # I used a simple string replace, to cut off .cpp.
	get_filename_component(exec_name ${source_file} NAME_WE)
    add_executable( ${exec_name} ${source_file} )
    # Make sure YourLib is linked to each app
endforeach( source_file ${APP_SOURCES} )
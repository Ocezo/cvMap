cmake_minimum_required(VERSION 3.20)

project( cvMap )
find_package( OpenCV REQUIRED )

add_executable( cvMap cvMap.cpp )

target_include_directories( cvMap PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( cvMap ${OpenCV_LIBS} )

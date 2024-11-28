cmake_minimum_required(VERSION 3.20)

project( cvMap )
find_package( OpenCV REQUIRED )

add_executable( cvMap cvMap.cpp )
target_link_libraries( cvMap ${OpenCV_LIBS} )

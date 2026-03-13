cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project( cvData )
find_package( OpenCV REQUIRED )

add_executable( cvData cvData.cpp )

target_include_directories( cvData PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( cvData ${OpenCV_LIBS} )

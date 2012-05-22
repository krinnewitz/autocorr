cmake_minimum_required (VERSION 2.6)
project (AC) 

FIND_PACKAGE( OpenCV REQUIRED )

add_executable (ac Main.cpp) 

TARGET_LINK_LIBRARIES( ac ${OpenCV_LIBS} )


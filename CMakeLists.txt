cmake_minimum_required (VERSION 2.6)
project (AC) 

set( CMAKE_CXX_FLAGS         "-O3 -g ")

FIND_PACKAGE( OpenCV REQUIRED )

add_executable (ac Main.cpp AutoCorr.cpp Texture.cpp ../crosscorr/CrossCorr.cpp) 

TARGET_LINK_LIBRARIES( ac ${OpenCV_LIBS} )


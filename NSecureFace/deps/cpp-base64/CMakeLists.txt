project(base64)

include_directories(include)

add_library(${PROJECT_NAME} STATIC
	include/base64.h 
	src/base64.cpp
)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_PROJECT_VERSION 0.1.1)

project(NSecureFace LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_subdirectory(deps)

find_package(OpenCV)

message("Found OpenCV => ${OpenCV_FOUND}")

if(${OpenCV_FOUND})
message("OpenCV Libraries => ${OpenCV_LIBRARIES}")
message("OpenCV Include Directory => ${OpenCV_INCLUDE_DIRS}")


set(NSECURE_FACE_HEADERS 
nsecureface.h
face-processing.hpp
)

set(NSECURE_FACE_SOURCES
nsecureface-service.cpp
face-processing.cpp
)
set(NSECURE_FACE_RESOURCES config.json)

include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${NSECURE_FACE_HEADERS}	
    deps/nlohmann-json-3.7.0/
    deps/cppfs-1.3.0/source/cppfs/include
)

add_executable(NSecureFaceService 
    ${NSECURE_FACE_HEADERS} 
    ${NSECURE_FACE_SOURCES}
    ${NSECURE_FACE_RESOURCES}
)

target_link_libraries(NSecureFaceService ${OpenCV_LIBRARIES} cppfs)
endif(${OpenCV_FOUND})

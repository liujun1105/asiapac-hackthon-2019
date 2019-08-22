project(nsecureface-service-deps)

set(CPPFS__DISABLE_TESTS ON)



set(OPTION_BUILD_TESTS OFF)
add_subdirectory(cppfs-1.3.0)
message(STATUS "Include cppfs-1.3.0 via add_subdirectory.")

set(NNG_TESTS OFF)
add_subdirectory(nng-1.1.1)
message(STATUS "Include nng-1.1.1 via add_subdirectory.")

#add_subdirectory(nlohmann-json-3.7.0)
#add_subdirectory(cpp-http-0.2.1)
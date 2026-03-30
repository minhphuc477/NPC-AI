cmake_minimum_required(VERSION 3.10)
project(test_multithreading_only)

set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)

include_directories(../include)
include_directories(../build/_deps/json-src/single_include)
include_directories(../build/_deps/json-src/include)

add_executable(test_multithreading
    "../src/SocialFabricNetwork.cpp"
    "../src/SimpleGraph.cpp"
    "../src/AmbientAwarenessSystem.cpp"
    "../src/PlayerBehaviorModeling.cpp"
    "../src/TemporalMemorySystem.cpp"
    "../src/NPCLogger.cpp"
    "test_multithreading.cpp"
)

if(WIN32)
    target_link_libraries(test_multithreading shlwapi)
    target_compile_definitions(test_multithreading PRIVATE NOMINMAX NOGDI)
endif()

cmake_minimum_required(VERSION 3.31.6)
set(PRJ_NAME WitcherPic)
project(${PRJ_NAME})

set(CUDA_PATH $ENV{CUDA_PATH})
set(WITCHER_INCLUDE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/include/types")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
# string(APPEND CMAKE_CXX_FLAGS " /EHa")
string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
	string(APPEND CMAKE_CUDA_FLAGS " -v")
	string(APPEND EXECUTABLE_OUTPUT_PATH "/Debug")
	string(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "/Debug")
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
	string(APPEND CMAKE_CUDA_FLAGS " -O3 -use_fast_math")
	string(APPEND EXECUTABLE_OUTPUT_PATH "/Release")
	string(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "/Release")
endif()
if(CMAKE_CL_64)
	string(APPEND EXECUTABLE_OUTPUT_PATH "/x64")
	string(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "/x64")
else()
	string(APPEND EXECUTABLE_OUTPUT_PATH "/x86")
	string(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "/x86")
endif()

enable_language(CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(freeImage CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

file(GLOB SRC_CPP
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)
file(GLOB SRC_CU ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
file(GLOB SRC_MODULE
	${CMAKE_CURRENT_SOURCE_DIR}/src/module/*.ixx
	${CMAKE_CURRENT_SOURCE_DIR}/src/module/*.cppm
)

add_compile_definitions(WITCHER_EXPORT)
add_library(witcherCUDA STATIC ${SRC_CU})

message(STATUS ${CMAKE_CUDA_COMPILER})
message(STATUS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
message(STATUS ${CMAKE_CXX_COMPILER})
message(STATUS ${SRC_CPP})

add_library(witcherPic STATIC)
add_executable(main main.cpp)

target_sources(witcherPic
	PRIVATE ${SRC_CPP}
	PUBLIC FILE_SET CXX_MODULES FILES ${SRC_MODULE}
)

set(WITCHER_PUBLIC_LIB
	fmt::fmt
	Eigen3::Eigen
	freeimage::FreeImage
	freeimage::FreeImagePlus)
target_include_directories(witcherPic PRIVATE ${WITCHER_INCLUDE})
target_include_directories(witcherCUDA PRIVATE ${WITCHER_INCLUDE})
target_include_directories(main PRIVATE "${PROJECT_SOURCE_DIR}/lib" "${PROJECT_SOURCE_DIR}/include/types")
# target_link_directories(witcherCUDA PRIVATE "${PROJECT_SOURCE_DIR}/lib")
target_link_libraries(witcherCUDA PRIVATE CUDA::cudart)
target_link_libraries(witcherPic PRIVATE
	witcherCUDA
	${WITCHER_PUBLIC_LIB}
)
target_link_libraries(main PRIVATE
	witcherPic
	${WITCHER_PUBLIC_LIB}
)

set_target_properties(witcherPic PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)

if(MSVC)
	target_compile_definitions(witcherPic PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_UNICODE)
endif()

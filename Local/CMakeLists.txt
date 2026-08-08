cmake_minimum_required(VERSION 3.16)

# Set policy for CUDA architectures
cmake_policy(SET CMP0104 NEW)

# Set the name of the project
project(CleanCudaExamples LANGUAGES CXX CUDA)

# Thrust/CCCL (used by RayTracerAdvancedCUDA's bvh.cuh) requires at least
# C++17. Set this project-wide so both the CXX and CUDA compilers agree.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# TODO: Actualize this number to fit your local architecture. Otherwise, kernels won't function properly and may
# even be undeterministic. You can check your GPU architecture at https://developer.nvidia.com/cuda/gpus and set the appropriate value.
set(CMAKE_CUDA_ARCHITECTURES 86)

# Gather the source files
file(GLOB_RECURSE SOURCES_1
    ${CMAKE_CURRENT_LIST_DIR}/src/1_countingGFLOPS/*.cu
)
file(GLOB_RECURSE SOURCES_2
    ${CMAKE_CURRENT_LIST_DIR}/src/2_cudaProperties/*.cu
)
# Gather the source files
file(GLOB_RECURSE SOURCES_3
    ${CMAKE_CURRENT_LIST_DIR}/src/3_variableSum/*.cu
)
file(GLOB_RECURSE SOURCES_4
    ${CMAKE_CURRENT_LIST_DIR}/src/4_matrixMul/*.cu
)
# See the diffence. This one is the serial RayTracer, for comparison only. Since we propted CMAKE for both CXX
# and CUDA, we can use .cpp files as well.
file(GLOB_RECURSE SOURCES_5
    ${CMAKE_CURRENT_LIST_DIR}/src/5_RayTracerSimpleSerial/*.cpp
)
file(GLOB_RECURSE SOURCES_6
    ${CMAKE_CURRENT_LIST_DIR}/src/6_RayTracerSimpleCUDA/*.cu
)
file(GLOB_RECURSE SOURCES_7
    ${CMAKE_CURRENT_LIST_DIR}/src/7_RayTracerAdvancedCUDA/*.cu
)



# Gather the header files
file(GLOB_RECURSE HEADERS_1
    ${CMAKE_CURRENT_LIST_DIR}/src/1_countingGFLOPS/*.cuh
)
file(GLOB_RECURSE HEADERS_2
    ${CMAKE_CURRENT_LIST_DIR}/src/2_cudaProperties/*.cuh
)
file(GLOB_RECURSE HEADERS_3
	${CMAKE_CURRENT_LIST_DIR}/src/3_variableSum/*.cuh
)
file(GLOB_RECURSE HEADERS_4
	${CMAKE_CURRENT_LIST_DIR}/src/4_matrixMul/*.cuh
)
file(GLOB_RECURSE HEADERS_5
	${CMAKE_CURRENT_LIST_DIR}/src/5_RayTracerSimpleSerial/*.hpp
)
file(GLOB_RECURSE HEADERS_6
	${CMAKE_CURRENT_LIST_DIR}/src/6_RayTracerSimpleCUDA/*.cuh
)
file(GLOB_RECURSE HEADERS_7
	${CMAKE_CURRENT_LIST_DIR}/src/7_RayTracerAdvancedCUDA/*.cuh
)

# Add the executables
add_executable(countingGFLOPS ${SOURCES_1} ${HEADERS_1})
add_executable(cudaProperties ${SOURCES_2} ${HEADERS_2})
add_executable(variableSum ${SOURCES_3} ${HEADERS_3})
add_executable(matrixMul ${SOURCES_4} ${HEADERS_4})
add_executable(RayTracerSimpleSerial ${SOURCES_5} ${HEADERS_5})
add_executable(RayTracerSimpleCUDA ${SOURCES_6} ${HEADERS_6})
add_executable(RayTracerAdvancedCUDA ${SOURCES_7} ${HEADERS_7})

# Include directories for the executables
target_include_directories(countingGFLOPS PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/1_countingGFLOPS)
target_include_directories(cudaProperties PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/2_cudaProperties)
target_include_directories(variableSum PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/3_variableSum)
target_include_directories(matrixMul PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/4_matrixMul)
target_include_directories(RayTracerSimpleSerial PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/5_RayTracerSimpleSerial)
target_include_directories(RayTracerSimpleCUDA PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/6_RayTracerSimpleCUDA)
target_include_directories(RayTracerAdvancedCUDA PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/7_RayTracerAdvancedCUDA)


# RayTracerAdvancedCUDA adds the Thrust/CCCL library, which requires C++17. We need to tell MSVC to use the correct preprocessor and C++ standard.
if(MSVC)
	target_compile_options(RayTracerAdvancedCUDA PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor,/Zc:__cplusplus>
	)
endif()

# Set the startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT countingGFLOPS)


if(MSVC_IDE)
	# Macro to preserve source files hierarchy in the IDE
	macro(GroupSources curdir)
		file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)

		foreach(child ${children})
			if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
				GroupSources(${curdir}/${child})
			else()
				string(REPLACE "/" "\\" groupname ${curdir})
				string(REPLACE "src" "Sources" groupname ${groupname})
				source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
			endif()
		endforeach()
	endmacro()

	# Run macro
	GroupSources(src)

endif()
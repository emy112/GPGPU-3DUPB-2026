cmake_minimum_required(VERSION 3.16)

# Set policy for CUDA architectures
cmake_policy(SET CMP0104 NEW)

# Set the name of the project
project(CleanCudaExamples LANGUAGES CXX CUDA)

# TODO: Actualize this number to fit your local architecture. Otherwise, kernels won't function properly and may
# even be undeterministic. You can check your GPU architecture at https://developer.nvidia.com/cuda/gpus and set the appropriate value.
set(CMAKE_CUDA_ARCHITECTURES 86)

# Gather the source files
file(GLOB_RECURSE SOURCES_1
    ${CMAKE_CURRENT_LIST_DIR}/src/1/*.cu
)
file(GLOB_RECURSE SOURCES_2
    ${CMAKE_CURRENT_LIST_DIR}/src/2/*.cu
)
# Gather the source files
file(GLOB_RECURSE SOURCES_3
    ${CMAKE_CURRENT_LIST_DIR}/src/3/*.cu
)
file(GLOB_RECURSE SOURCES_4
    ${CMAKE_CURRENT_LIST_DIR}/src/4/*.cu
)



# Gather the header files
file(GLOB_RECURSE HEADERS_1
    ${CMAKE_CURRENT_LIST_DIR}/src/1/*.cuh
)
file(GLOB_RECURSE HEADERS_2
    ${CMAKE_CURRENT_LIST_DIR}/src/2/*.cuh
)
file(GLOB_RECURSE HEADERS_3
	${CMAKE_CURRENT_LIST_DIR}/src/3/*.cuh
)
file(GLOB_RECURSE HEADERS_4
	${CMAKE_CURRENT_LIST_DIR}/src/4/*.cuh
)

# Add the executables
add_executable(countingGFLOPS ${SOURCES_1} ${HEADERS_1})
add_executable(cudaProperties ${SOURCES_2} ${HEADERS_2})
add_executable(variableSum ${SOURCES_3} ${HEADERS_3})
add_executable(matrixMul ${SOURCES_4} ${HEADERS_4})

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
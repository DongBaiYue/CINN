#########################
#  FindDPCPP.cmake
#########################

include_guard()

include(FindPackageHandleStandardArgs)

if (WITH_SYCL)
    set(DPCPP_ROOT ${WITH_SYCL})
    find_library(DPCPP_LIB NAMES sycl PATHS "${DPCPP_ROOT}/lib")
    find_package_handle_standard_args(DPCPP
        FOUND_VAR     DPCPP_FOUND
        REQUIRED_VARS DPCPP_LIB
    )
    if(NOT DPCPP_FOUND)
        return()
    endif()
    message(STATUS "Enable SYCL")
    include_directories("${DPCPP_ROOT}/include/sycl;${DPCPP_ROOT}/include")
    link_libraries(${DPCPP_LIB})
    # used in cpp file
    add_definitions(-DCINN_WITH_SYCL)
    add_definitions(-DSYCL_CXX_COMPILER="${DPCPP_ROOT}/bin/clang++")
else()
    return()
endif()
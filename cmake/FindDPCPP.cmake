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
    include_directories("${DPCPP_ROOT}/include/sycl;${DPCPP_ROOT}/include")
    link_libraries(${DPCPP_LIB})
    # link_directories("${DPCPP_ROOT}/lib")
else()
    return()
endif()
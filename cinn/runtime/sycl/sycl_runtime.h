#pragma once

#include <sycl/sycl.hpp>
#include <glog/logging.h>
#include <mutex>
#include <string>
#include <vector>
#include "cinn/common/target.h"
using namespace cinn::common;

inline const char* SYCLGetErrorString(std::error_code error_code) {
  sycl::errc error_code_value = static_cast<sycl::errc>(error_code.value());
  switch(error_code_value){
    case sycl::errc::success:
      return "SUCCESS";
    case sycl::errc::runtime:
      return "RUNTIME ERROR";
    case sycl::errc::kernel:
      return "KERNEL ERROR";
    case sycl::errc::accessor:
      return "ACCESSOR ERROR";
    case sycl::errc::nd_range:
      return "NDRANGE ERROR";
    case sycl::errc::event:
      return "EVENT ERROR";
    case sycl::errc::kernel_argument:
      return "KERNEL ARGUMNET ERROR";
    case sycl::errc::build:
      return "BUILD ERROR";
    case sycl::errc::invalid:
      return "INVALID ERROR";
    case sycl::errc::memory_allocation:
      return "MEMORY ALLOCATION";
    case sycl::errc::platform:
      return "PLATFORM ERROR";
    case sycl::errc::profiling:
      return "PROFILING ERROR";
    case sycl::errc::feature_not_supported:
      return "FEATURE NOT SUPPORTED";
    case sycl::errc::kernel_not_supported:
      return "kERNEL NOT SUPPORTED";
    case sycl::errc::backend_mismatch:
      return "BACKEND MISMATCH";
    default:
        return "";
  }
}

/*!
 * \brief Protected SYCL call
 * \param func Expression to call.
 */
#define SYCL_CALL(func)                                                       \
  {                                                                           \
    try{                                                                      \
      func;                                                                   \
    }catch(const sycl::exception &e){                                         \
      CHECK(e.code() == sycl::errc::success) << "SYCL Error, code=" << ": " << SYCLGetErrorString(e.code()) <<", message:"<< e.what();;\
    }                                                                         \
  }


/*!
 * \brief Process global SYCL workspace.
 */
class SYCLWorkspace {
 public:
  // global platform
  std::vector<sycl::platform> platforms;
  // global platform name
  std::vector<std::string> platform_names;
  // global context of this process
  std::vector<sycl::context> contexts;
  // whether the workspace it initialized.
  bool initialized_{false};
  // the device type
  std::string device_type;
  // the devices
  std::vector<sycl::device> devices;
  // the queues
  std::vector<sycl::queue> queues;
  // the events
  std::vector<std::vector<sycl::event>> events;
  // the mutex for initialization
  std::mutex mu;
  // destructor
  ~SYCLWorkspace() {
    for(auto queue : queues){
      SYCL_CALL(queue.wait_and_throw());
    }
  }
  // get the global workspace
  static SYCLWorkspace* Global();
  // Initialzie the device.
  void Init(const Target::Arch arch, const std::string& platform_name = "");
  void* malloc(size_t nbytes, int device_id=0);
  void free(void* data, int device_id=0);
  void queueSync(int queue_id=0);
  void memcpy(void* dest, const void* src, size_t nbytes, int queue_id=0);
                           
};
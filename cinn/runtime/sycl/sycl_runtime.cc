#include <cinn/runtime/sycl/sycl_runtime.h>
#include <iostream>

SYCLWorkspace* SYCLWorkspace::Global() {
  static SYCLWorkspace* inst = new SYCLWorkspace();
  return inst;
}

void SYCLWorkspace::Init(const Target::Arch arch, const std::string& platform_name) {                     
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);

  // look for matched platform
  bool have_platform = false;
  auto platforms = sycl::platform::get_platforms();
  std::string platform_key;
  switch (arch) {
    case Target::Arch::NVGPU:
        platform_key = "CUDA";
        break;
    case Target::Arch::AMDGPU:
        platform_key = "HIP";
        break;
    case Target::Arch::IntelGPU:
        platform_key = "Level-Zero";
        break;
    default:
      LOG(FATAL) << "SYCL Not supported arch!";
  }
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (const std::exception_ptr &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (const sycl::exception &e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      }
    }
  };
  sycl::property_list q_prop{sycl::property::queue::in_order()};
  for (auto &platform : platforms) {
    std::string name = platform.get_info<sycl::info::platform::name>();
    // neither NVIDIA CUDA BACKEND nor AMD HIP BACKEND nor Intel Level-Zero
    if(name.find(platform_key) == std::string::npos)
        continue;
    std::vector<sycl::device> devices = platform.get_devices(sycl::info::device_type::gpu);
    this->platforms.push_back(platform);
    this->devices.insert(this->devices.end(),devices.begin(),devices.end());
    this->platform_names.push_back(platform_name);
    for(sycl::device dev : devices){
        //create context and queue
        sycl::context ctx = sycl::context(dev,exception_handler);
        this->contexts.push_back(ctx);
        //one device one queue
        sycl::queue queue = sycl::queue(ctx, dev, q_prop); // In order queue
        this->queues.push_back(queue);
        have_platform = true;
        break;//only support single device
    }
  }
  if (!have_platform) {
    LOG(FATAL) << "No valid gpu device/platform matched given existing options ...";
    return;
  }
  this->events.resize(this->devices.size());
  initialized_ = true;
  VLOG(1) << "devices size : " << this->devices.size() << std::endl;
  std::cout << "devices size : " << this->devices.size() << std::endl;
}

void* SYCLWorkspace::malloc(size_t nbytes, int device_id){
    void* data;
    SYCL_CALL(data = sycl::malloc_device(nbytes, this->queues[device_id]))
    if(data == nullptr)
      LOG(ERROR) << "allocate sycl device memory failure!"<<std::endl;
    return data;
}

void SYCLWorkspace::free(void* data, int device_id){
  SYCL_CALL(sycl::free(data, this->queues[device_id]));
}

void SYCLWorkspace::queueSync(int queue_id) {
  SYCL_CALL(this->queues[queue_id].wait_and_throw());
}

void SYCLWorkspace::memcpy(void* dest, const void* src, size_t nbytes, int queue_id) {
  std::cout<<"SYCLWorkspace::memcpy. ly add"<<std::endl;
  SYCL_CALL(this->queues[queue_id].memcpy(dest, src, nbytes).wait());
}
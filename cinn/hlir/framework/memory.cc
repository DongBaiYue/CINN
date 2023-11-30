// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/hlir/framework/memory.h"

#ifdef CINN_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "cinn/backends/cuda_util.h"
#endif

#include <cinn/runtime/sycl/sycl_runtime.h>
namespace cinn {
namespace hlir {
namespace framework {

using common::Target;

namespace {

class X86MemoryMng : public MemoryInterface {
 public:
  void* malloc(size_t nbytes) override { return ::malloc(nbytes); }
  void free(void* data) override {
    if (!data) return;
    ::free(data);
  }
  void* aligned_alloc(size_t alignment, size_t nbytes) override { return ::aligned_alloc(alignment, nbytes); }
};

#ifdef CINN_WITH_CUDA
class CudaMemoryMng : public MemoryInterface {
 public:
  void* malloc(size_t nbytes) override {
    void* data;
    CUDA_CALL(cudaMalloc(&data, nbytes));
    return data;
  }

  void free(void* data) override { CUDA_CALL(cudaFree(data)); }
};

#endif

class SYCLMemoryMng : public MemoryInterface {
  public:
    SYCLMemoryMng(){
      sycl_workspace = SYCLWorkspace::Global();
    }
    void* malloc(size_t nbytes) override {
      return sycl_workspace->malloc(nbytes);
    }
    void free(void* data) override {
      sycl_workspace->free(data);
    }
  private:
    SYCLWorkspace *sycl_workspace;
};

}  // namespace

MemoryManager::MemoryManager() {
  Register(Target::Language::Unk, new X86MemoryMng);
  Register(Target::Language::c, new X86MemoryMng);
#ifdef CINN_WITH_CUDA
  Register(Target::Language::cuda, new CudaMemoryMng);
#endif
#ifdef CINN_WITH_SYCL
  Register(Target::Language::sycl, new SYCLMemoryMng);
#endif
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

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

#include "cinn/backends/sycl/codegen_sycl_dev.h"

#include <cinn/utils/string.h>
#include <glog/logging.h>

#include <fstream>
#include <set>
#include <unordered_set>

#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_verify.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/remove_nested_block.h"

namespace cinn {
namespace backends {

const std::string CodeGenSYCL_Dev::source_header_ =
    R"(#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
)";

const std::string &CodeGenSYCL_Dev::GetSourceHeader() { return source_header_; }

CodeGenSYCL_Dev::CodeGenSYCL_Dev(Target target) : CodeGenC(target) {}

std::string CodeGenSYCL_Dev::Compile(const ir::Module &module, bool for_syclrtc) {
  for_syclrtc_ = for_syclrtc;
  auto source  = Compile(module, OutputKind::CImpl);

  return source;
}

void CodeGenSYCL_Dev::Compile(const ir::Module &module, const Outputs &outputs) {
  LOG(FATAL) << "CINN_SYCL_codegen_NOT_IMPLEMENTED";
}

std::string CodeGenSYCL_Dev::Compile(const ir::LoweredFunc &func) {
  Print(Expr(func));
  return ss_.str();
}

std::vector<Expr> CodeGenSYCL_Dev::GenerateBufferAliasExprs(const ir::_LoweredFunc_ *op,
                                                            const std::vector<ir::Buffer> &temp_buffers) {
  std::set<ir::Buffer> temp_buffer_set(temp_buffers.begin(), temp_buffers.end());
  // prepare temp buffer alias
  std::vector<Expr> buffer_alias;
  auto tensors = ir::CollectIRNodes(op->body, [&](const Expr *x) {
    return x->as_tensor() && x->as_tensor()->buffer.defined() && temp_buffer_set.count(x->as_tensor()->buffer);
  });

  // unique tensors
  std::set<ir::Tensor> unique_tensors;
  for (auto &e : tensors) {
    unique_tensors.insert(e.as_tensor_ref());
  }

  for (auto &t : unique_tensors) {
    auto data_type     = t->type();
    auto data_ptr_type = data_type;
    data_ptr_type.set_cpp_handle();

    Var t_var(t->name, data_ptr_type);
    Var buf_var(t->buffer->name, data_ptr_type);
    buffer_alias.push_back(ir::Let::Make(t_var, buf_var));
  }

  return buffer_alias;
}

void CodeGenSYCL_Dev::Visit(const ir::_LoweredFunc_ *op) {
  // clear names valid within scope when enter a new function
  vectorized_tensor_names_.clear();

  // Print the packed function
  os() << "// CodeGenSYCL: NOTE: Auto-generated packed function\n";
  os() << "void ";
  os() << op->name;
  os() << "(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {\n";
  IncIndent();
  // read void_args
  PrintFuncArgs(op->args);
  DoIndent();
  os() << "Q.submit([&](sycl::handler &h) {\n";
  IncIndent();
  DoIndent();
  os() << "h.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) "
          "[[intel::kernel_args_restrict]]";
  if (op->cuda_axis_info.valid()) {
    os() << "[[intel::max_work_group_size(" << op->cuda_axis_info.block_dim(2) << ", "
         << op->cuda_axis_info.block_dim(1) << ", " << op->cuda_axis_info.block_dim(0) << ")]]";
  }
  os() << "\n";

  // function body
  PrintFunctionBody(op);

  os() << ");\n";
  DecIndent();
  DoIndent();
  os() << "});\n";
  DecIndent();
  os() << "}\n";
}

void CodeGenSYCL_Dev::Visit(const ir::_Var_ *op) {
  if (utils::Startswith(op->name, "threadIdx") || utils::Startswith(op->name, "blockIdx")) {
    if (utils::Startswith(op->name, "threadIdx")){
      os() << "item.get_local_id(";
    }else{
      os() << "item.get_group(";
    }
    if (utils::Endswith(op->name, "x")) {
      os() << 2;
    } else if (utils::Endswith(op->name, "y")) {
      os() << 1;
    } else if (utils::Endswith(op->name, "z")) {
      os() << 0;
    }
    os() << ")";
  } else {
    os() << op->name;
  }
}

void CodeGenSYCL_Dev::Visit(const ir::Alloc *op) {
  CHECK(op->destination.as_buffer());
  PrintTempBufferCreation(op->destination.as_buffer_ref());
}

void CodeGenSYCL_Dev::Visit(const ir::Min *op) {
  os() << "sycl::min(";
  Print(op->a());
  os() << ", ";
  Print(op->b());
  os() << ")";
}

void CodeGenSYCL_Dev::Visit(const ir::Max *op) {
  os() << "sycl::max(";
  Print(op->a());
  os() << ", ";
  Print(op->b());
  os() << ")";
}

void CodeGenSYCL_Dev::PrintFunctionBody(const ir::_LoweredFunc_ *op) {
  DoIndent();

  std::vector<Expr> new_body;

  auto alloca_temp_buffers = op->PrepareAllocTempBufferExprs();
  auto temp_buffer_alias   = GenerateBufferAliasExprs(op, op->temp_bufs);
  auto alis_var_exprs      = op->CudaAliasVarExprs();

#define APPEND_TO_NEW_BODY(field__) new_body.insert(std::end(new_body), std::begin(field__), std::end(field__));
  APPEND_TO_NEW_BODY(alloca_temp_buffers)
  APPEND_TO_NEW_BODY(temp_buffer_alias)
  APPEND_TO_NEW_BODY(alis_var_exprs)

  new_body.push_back(op->body);

  Expr func_body = ir::Block::Make(new_body);

  optim::RemoveNestedBlock(&func_body);
  // Make sure that the function's body is wrapped by a block
  if (!func_body.As<ir::Block>()) {
    func_body = ir::Block::Make({func_body});
  }
  Print(func_body);
}

void CodeGenSYCL_Dev::PrintFuncArgs(std::vector<cinn::ir::Argument> args) {
  for (int i = 0; i < args.size(); i++) {
    DoIndent();
    auto &arg = args[i];
    if (arg.is_buffer()) {
      if (arg.is_input()) os() << "const ";
      os() << GetTypeRepr(arg.buffer_arg()->dtype);
      os() << "* ";
      os() << ir::BufferGetTensorName(arg.buffer_arg().As<ir::_Buffer_>());
      os() << " = (";
      os() << GetTypeRepr(arg.buffer_arg()->dtype);
      os() << "* ";
    } else if (arg.is_var()) {
      os() << GetTypeRepr(arg.type()) << " ";
      os() << arg.name();
      os() << " = (";
      os() << GetTypeRepr(arg.type());
    } else {
      CINN_NOT_IMPLEMENTED
    }
    os() << ")(*(void **)(void_args[" << i << "]));\n";
  }
}

void CodeGenSYCL_Dev::PrintBuiltinCodes() {}

std::string CodeGenSYCL_Dev::Compile(const ir::Module &module, CodeGenC::OutputKind output_kind) {
  if (output_kind == OutputKind::CHeader) {
    LOG(FATAL) << "CINN_SYCL_codegen_NOT_IMPLEMENTED";
    // GenerateHeaderFile(module);
  } else if (output_kind == OutputKind::CImpl) {
    PrintIncludes();

    if (for_syclrtc_) {
      os() << "#ifdef __cplusplus\n"
           << "extern \"C\" {\n"
           << "#endif\n";
    }

    PrintBuiltinCodes();

    for (auto &func : module.functions()) {
      Compile(func);
    }
  } else {
    LOG(FATAL) << "Not supported OutputKind";
  }

  if (for_syclrtc_) {
    os() << "\n#ifdef __cplusplus\n"
         << "}\n"
         << "#endif\n";
  }

  return ss_.str();
}

void CodeGenSYCL_Dev::PrintIncludes() { os() << GetSourceHeader(); }

void CodeGenSYCL_Dev::PrintTempBufferCreation(const ir::Buffer &buffer) {
  CHECK_NE(buffer->type(), Void());
  auto print_gpu_memory = [&](const std::string &mark) {
    os() << mark << GetTypeRepr(buffer->dtype) << " " << buffer->name << " ";

    os() << "[ ";
    Expr buffer_size(1);
    for (int i = 0; i < buffer->shape.size(); i++) {
      buffer_size = buffer_size * buffer->shape[i];
    }
    optim::Simplify(&buffer_size);
    Print(buffer_size);
    os() << " ]";
  };
  switch (buffer->memory_type) {
    case ir::MemoryType::GPUShared:
      {
        os() << "auto " << buffer->name << " = *sycl::ext::oneapi::group_local_memory<";
        os() << GetTypeRepr(buffer->dtype);
        os() << "[ ";
        Expr buffer_size(1);
        for (int i = 0; i < buffer->shape.size(); i++) {
          buffer_size = buffer_size * buffer->shape[i];
        }
        optim::Simplify(&buffer_size);
        Print(buffer_size);
        os() << " ]>(item.get_group())";
        break;
      }
    case ir::MemoryType::GPULocal:
      print_gpu_memory("");
      break;

    default:
      LOG(FATAL) << "CUDA device codegen not support memory " << buffer->name << ", type " << buffer->memory_type;
  }
}

void CodeGenSYCL_Dev::Visit(const ir::Call *op) {
  if (op->name == "__syncthreads") {
    os() << "sycl::group_barrier(item.get_group())";
    return;
  }
  os() << op->name + "(";

  if (!op->read_args.empty()) {
    for (int i = 0; i < op->read_args.size() - 1; i++) {
      auto &arg = op->read_args[i];
      if (arg.as_tensor()) {
        os() << arg.as_tensor()->name;
        os() << ", ";
      } else {
        Print(arg);
        os() << ", ";
      }
    }
    if (op->read_args.back().as_tensor()) {
      os() << op->read_args.back().as_tensor()->name;
    } else {
      Print(op->read_args.back());
    }
  }

  if (!op->write_args.empty()) {
    os() << ", ";
    for (int i = 0; i < op->write_args.size() - 1; i++) {
      auto &arg = op->write_args[i];
      if (arg.as_tensor()) {
        os() << arg.as_tensor()->name;
        os() << ", ";
      } else {
        Print(arg);
        os() << ", ";
      }
    }
    if (op->write_args.back().as_tensor()) {
      os() << op->write_args.back().as_tensor()->name;
    } else {
      Print(op->write_args.back());
    }
  }
  // sycl need parameter nd_item
  if ((op->name.find("cinn_block_reduce") != std::string::npos) ||
      (op->name.find("cinn_warp_reduce") != std::string::npos)) {
    os() << ", item";
  }

  os() << ")";
}

void CodeGenSYCL_Dev::Visit(const ir::Let *op) {
  CHECK(op->type().valid());

  // identify vectorized tensors by checking their dtypes are customized_type
  // with customized_type::kcuda_builtin_vector_t prefix, and save their names
  if (op->type().is_customized() &&
      utils::Startswith(op->type().customized_type(), common::customized_type::kcuda_builtin_vector_t)) {
    os() << GetTypeRepr(op->type());
    if (op->type().is_cpp_handle()) {
      os() << " " << kCKeywordRestrict;
    }
    os() << " ";
    Print(op->symbol);
    vectorized_tensor_names_.insert(utils::GetStreamCnt(op->symbol));
    // skip "=0" in "half8 temp = 0;" sincethe operator= of half8 may not overloaded.
    if (op->body.As<ir::IntImm>() && op->body.As<ir::IntImm>()->value == 0) {
      return;
    }
    os() << " = ";
    Print(op->body);
  } else {
    CodeGenC::Visit(op);
  }
}

bool CodeGenSYCL_Dev::PrintBuiltinVectorAccess(const ir::LoadStoreAddrMnger *op, ir::Expr index_expr, bool is_store) {
  static constexpr char index2suffix[8] = {'x', 'y', 'z', 'w', 'v', 'u', 't', 's'};

  // addr of op should be a place of tensor and the index is simple int number
  if (!op->is_addr_tensor() || !index_expr.As<ir::IntImm>()) {
    return false;
  }
  auto *tensor = op->tensor.As<ir::_Tensor_>();
  CHECK(tensor);

  // identify vectorized tensors by their names
  if (!vectorized_tensor_names_.count(tensor->name)) {
    return false;
  }

  // the index can't exceed the range of cuda built-in vector type
  int index = index_expr.As<ir::IntImm>()->value;
  if (index < 0 || index >= 8) {
    return false;
  }
  if (is_store && tensor->type().is_cpp_handle()) {
    os() << tensor->name << "[" << index << "]";
  } else {
    os() << tensor->name << (tensor->type().is_cpp_handle() ? "->" : ".") << index2suffix[index];
  }
  return true;
}

void CodeGenSYCL_Dev::Visit(const ir::Load *op) {
  // overload this visit function to especially deal with the case when it accesses
  // element at a cuda built-in vector, others still resolve to CodeGenC
  if (!PrintBuiltinVectorAccess(op, op->index(), false)) {
    CodeGenC::Visit(op);
  }
}

void CodeGenSYCL_Dev::Visit(const ir::Store *op) {
  // overload this visit function to especially deal with the case when it accesses
  // element at a cuda built-in vector, others still resolve to CodeGenC
  if (PrintBuiltinVectorAccess(op, op->index(), true)) {
    os() << " = ";
    Print(op->value);
  } else {
    CodeGenC::Visit(op);
  }
}

}  // namespace backends
}  // namespace cinn

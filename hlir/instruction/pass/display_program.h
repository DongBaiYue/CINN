#pragma once
#include <string>

#include "hlir/instruction/pass.h"

namespace hlir {
namespace instruction {
namespace pass {

class DisplayProgram : public ModulePass {
 public:
  explicit DisplayProgram(const std::string& name) : name_(name) {}

  bool Run(Module* module) override;

  std::string_view name() const override;

 private:
  std::string name_;
};

}  // namespace pass
}  // namespace instruction
}  // namespace hlir

#include "SpvProcessor.hpp"

#include "Common.hpp"
#include "source/disassemble.h"
#include "source/latest_version_spirv_header.h"
#include "source/opt/inst_basic_block_trace_pass.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/pass.h"
#include "source/opt/types.h"
#include "source/table.h"
#include "spirv-tools/libspirv.h"
#include "spirv-tools/optimizer.hpp"
#include "spirv/unified1/spirv.hpp11"
#include <algorithm>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <string>
#include <tuple>

using namespace vkExecute::spv;

// this function only considers entrypoint function
// and the entrypoint function is not expected to have OpFunctionCall
// to other instruction or modules
std::vector<BasicBlock> vkExecute::SpvProcessor::separateBasicBlocks() {
  std::vector<BasicBlock> separatedBB;

  std::unique_ptr<spvtools::opt::IRContext> context = spvtools::BuildModule(
      spv_target_env::SPV_ENV_VULKAN_1_3,
      [&](spv_message_level_t level, const char *source,
          const spv_position_t &position, const char *message) {
        const char *errorLevels[] = {"Fatal",   "Internal error", "Error",
                                     "Warning", "Info",           "Debug"};
        std::cerr << "[" << errorLevels[level] << "] " << (source ? source : "") << " (L"
                  << position.line << ":" << position.column << "): " << message
                  << std::endl;
      },
      spvBlob.data(), spvBlob.size(), false);

  if (context == nullptr) {
    VKEXECUTE_WARN("Failed to build module");
    return separatedBB;
  }

  spvtools::opt::Pass::ProcessFunction pfn = [&context, &separatedBB](
                                                 spvtools::opt::Function *fp) {
    fp->Dump();
    std::cerr << std::endl;

    for (auto bbIt = fp->begin(); bbIt != fp->end(); bbIt++) {
      bbIt->Dump();

      std::set<uint32_t> ty_ids;
      std::set<uint32_t> consumed_ids;
      std::set<uint32_t> generated_ids;
      std::set<uint32_t> consumed_external_ids;
      std::set<uint32_t> resource_candidate_ids;

      bool newBBCreated = false;
      bbIt->ForEachInst([&ty_ids, &consumed_ids, &generated_ids, &newBBCreated,
                         &separatedBB](const spvtools::opt::Instruction *inst) {
        // Block terminators won't give new value to the basic block, hence
        // not necessary for analysis and storage NOTE: We have OpAbort
        // inside the block terminator
        if (inst->IsBlockTerminator()) {
          return;
        }

        if (inst->opcode() == ::spv::Op::OpLoopMerge ||
            inst->opcode() == ::spv::Op::OpSelectionMerge) {
          return;
        }

        if (inst->opcode() == ::spv::Op::OpLabel) {
          return;
        }

        // discard store / load instructions;
        // we only need register based description
        if (inst->opcode() == ::spv::Op::OpStore ||
            inst->opcode() == ::spv::Op::OpLoad ||
            inst->opcode() == ::spv::Op::OpAccessChain) {
          return;
        }

        if (inst->opcode() == ::spv::Op::OpVariable) {
          return;
        }

        if (!newBBCreated) {
          separatedBB.push_back(std::move(BasicBlock{}));
          newBBCreated = true;
        }

        BasicBlock &newBB = separatedBB.back();
        Instr newInst = Instr::create(inst);
        newBB.bbInstructions.push_back(newInst);

        // spv inst operands could be literial or ids
        // id could be id, typeid, resultid, memory semantics id, scope id
        // see spvIsIdType for more info

        if (inst->HasResultType()) {
          ty_ids.insert(inst->type_id());
        }

        inst->ForEachInId([&consumed_ids](const uint32_t *id_operand) {
          consumed_ids.insert(*id_operand);
        });

        if (inst->HasResultId()) {
          generated_ids.insert(inst->result_id());
        }
      });

      if (!newBBCreated) {
        std::cerr << "Skip this bb because no effective instructions present"
                  << std::endl;
        continue;
      }

      std::cerr << "=> Basic infomation of this bb:" << std::endl;

      std::set_difference(
          consumed_ids.begin(), consumed_ids.end(), generated_ids.begin(),
          generated_ids.end(),
          std::inserter(consumed_external_ids, consumed_external_ids.begin()));

      BasicBlock &newBB = separatedBB.back();
      auto &tyId2Type = newBB.associatedTypes;
      for (auto tyId : ty_ids) {
        auto spvOptType = context->get_type_mgr()->GetType(tyId);
        auto type = Type::Type::create(spvOptType, context.get());
        tyId2Type[tyId] = std::move(type);
      }

      for (auto consumedId : consumed_external_ids) {
        auto pInst = context->get_def_use_mgr()->GetDef(consumedId);

        if (pInst->HasResultType()) {
          auto tyId = pInst->type_id();
          auto spvOptType = context->get_type_mgr()->GetType(tyId);
          auto type = Type::Type::create(spvOptType, context.get());
          tyId2Type[tyId] = std::move(type);

          Register newVar;
          newVar.type = tyId;
          newVar.ssaId = consumedId;
          newBB.inputDescription.push_back(newVar);
        } else {
          // resource candidate
          resource_candidate_ids.insert(consumedId);
        }
      }

      for (auto &resCandId : resource_candidate_ids) {
        auto pInst = context->get_def_use_mgr()->GetDef(resCandId);

        if (pInst->opcode() == ::spv::Op::OpExtInstImport) {
          auto operand = pInst->GetInOperand(0);
          auto extInstSet = operand.AsString();

          newBB.resourceDescription.push_back(Resource());
          auto &newRes = newBB.resourceDescription.back();

          newRes.type = Resource::Type::ExtInstImport;
          newRes.ssaId = resCandId;
          newRes.payload = ExtInstImportResource{toU8String(extInstSet)};
        } else {
          throw std::runtime_error("Should handle this");
        }
      }

      for (auto &generatedId : generated_ids) {
        Register outVar;
        auto outInst = context->get_def_use_mgr()->GetDef(generatedId);

        assert(outInst->HasResultType() && outInst->HasResultId());
        outVar.type = outInst->type_id();
        outVar.ssaId = generatedId;

        newBB.outputDescription.push_back(outVar);
      }

      std::cerr << "   => Type Ids: " << ToString(ty_ids) << std::endl;
      std::cerr << "   => Consumed Ids: " << ToString(consumed_ids)
                << std::endl;
      std::cerr << "   => Generated Ids: " << ToString(generated_ids)
                << std::endl;
      std::cerr << "   => Consumed External Ids: "
                << ToString(consumed_external_ids) << std::endl;

      std::cerr << "   => Converted Types:" << std::endl;
      for (auto &tyPair : tyId2Type) {
        std::cerr << "     {" << tyPair.first << ", " << tyPair.second->str()
                  << "}" << std::endl;
      }

      std::cerr << "   => Resource candidate IDs: "
                << ToString(resource_candidate_ids) << std::endl;

      std::cerr << std::endl;
    }

    return false;
  };

  context->ProcessEntryPointCallTree(pfn);

  return separatedBB;
}

std::tuple<bool, std::string> vkExecute::SpvProcessor::exhaustiveInlining() {
  spvtools::Optimizer opt(spv_target_env::SPV_ENV_VULKAN_1_3);
  std::stringstream errMsgs;

  opt.SetMessageConsumer(
      [&errMsgs](spv_message_level_t level, const char *source,
                 const spv_position_t &position, const char *message) {
        const char *errorLevels[] = {"Fatal",   "Internal error", "Error",
                                     "Warning", "Info",           "Debug"};
        errMsgs << "[" << errorLevels[level] << "] " << (source ? source : "") << " (L"
                << position.line << ":" << position.column << "): " << message
                << std::endl;
      });

  // inlining requires merge return pass to patch functions with single ending
  // return
  opt.RegisterPass(spvtools::CreateMergeReturnPass());
  opt.RegisterPass(spvtools::CreateInlineExhaustivePass());

  // the inline exhaustive pass will not erase the inlined functions;
  // use this to clean those
  opt.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());

  std::vector<uint32_t> spirv;
  bool success = opt.Run(spvBlob.data(), spvBlob.size(), &spirv);

  if (!success) {
    return std::make_tuple(false, errMsgs.str());
  }

  spvBlob = spirv;
  return std::make_tuple(true, errMsgs.str());
}

std::tuple<std::string, std::string> vkExecute::SpvProcessor::disassemble() {
  std::string output = "";
  std::stringstream ss;

  // disassemble
  spv_context context = spvContextCreate(spv_target_env::SPV_ENV_VULKAN_1_3);
  spv_text text;
  spv_diagnostic diagnostic = nullptr;
  spvBinaryToText(context, spvBlob.data(), spvBlob.size(),
                  SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                      SPV_BINARY_TO_TEXT_OPTION_INDENT,
                  &text, &diagnostic);

  // dump
  if (diagnostic == nullptr) {
    output = text->str;
  } else {
    if (diagnostic->isTextSource) {
      // NOTE: This is a text position
      // NOTE: add 1 to the line as editors start at line 1, we are counting new
      // line characters to start at line 0
      ss << "error: " << diagnostic->position.line + 1 << ": "
         << diagnostic->position.column + 1 << ": " << diagnostic->error
         << "\n";
    }

    // NOTE: Assume this is a binary position
    ss << "error: ";
    if (diagnostic->position.index > 0) {
      ss << diagnostic->position.index << ": ";
    }
    ss << diagnostic->error << "\n";
  }

  // teardown
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);

  return std::make_tuple(output, ss.str());
}

std::tuple<bool, std::string>
vkExecute::SpvProcessor::assemble(std::string asmText) {
  spvtools::SpirvTools core(static_cast<spv_target_env>(SPV_ENV_VULKAN_1_3));
  std::stringstream errMsgs;

  core.SetMessageConsumer(
      [&errMsgs](spv_message_level_t level, const char *source,
                 const spv_position_t &position, const char *message) {
        const char *errorLevels[] = {"Fatal",   "Internal error", "Error",
                                     "Warning", "Info",           "Debug"};
        errMsgs << "[" << errorLevels[level] << "] " << (source ? source : "") << " (L"
                << position.line << ":" << position.column << "): " << message
                << std::endl;
      });

  if (!core.Assemble(asmText, &spvBlob,
                     SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS)) {
    spvBlob.clear();
    return std::make_tuple(false, errMsgs.str());
  }

  return std::make_tuple(true, errMsgs.str());
}

std::tuple<bool, std::string> vkExecute::SpvProcessor::validate() {
  spvtools::SpirvTools core(static_cast<spv_target_env>(SPV_ENV_VULKAN_1_3));
  std::stringstream errMsgs;

  core.SetMessageConsumer(
      [&errMsgs](spv_message_level_t level, const char *source,
                 const spv_position_t &position, const char *message) {
        const char *errorLevels[] = {"Fatal",   "Internal error", "Error",
                                     "Warning", "Info",           "Debug"};
        errMsgs << "[" << errorLevels[level] << "] " << (source ? source : "") << " (L"
                << position.line << ":" << position.column << "): " << message
                << std::endl;
      });

  if (!core.Validate(spvBlob)) {
    spvBlob.clear();
    return std::make_tuple(false, errMsgs.str());
  }

  return std::make_tuple(true, errMsgs.str());
}

std::tuple<bool, std::string> vkExecute::SpvProcessor::runPassSequence(
    std::vector<std::string> passSequence) {
  spvtools::Optimizer opt(spv_target_env::SPV_ENV_VULKAN_1_3);
  std::stringstream errMsgs;

  opt.SetMessageConsumer(
      [&errMsgs](spv_message_level_t level, const char *source,
                 const spv_position_t &position, const char *message) {
        const char *errorLevels[] = {"Fatal",   "Internal error", "Error",
                                     "Warning", "Info",           "Debug"};
        errMsgs << "[" << errorLevels[level] << "] " << (source ? source : "") << " (L"
                << position.line << ":" << position.column << "): " << message
                << std::endl;
      });

  bool regSuccess = opt.RegisterPassesFromFlags(passSequence);
  bool success = false;
  std::vector<uint32_t> spirv;
  if (regSuccess) {
    success = opt.Run(spvBlob.data(), spvBlob.size(), &spirv);
  }

  if (!success) {
    return std::make_tuple(false, errMsgs.str());
  }

  spvBlob = spirv;
  return std::make_tuple(true, errMsgs.str());
}

std::map<int, int> vkExecute::SpvProcessor::instrumentBasicBlockTrace(bool traceWithU64) {
  std::stringstream errMsgs;

  std::unique_ptr<spvtools::opt::IRContext> context = spvtools::BuildModule(
      spv_target_env::SPV_ENV_VULKAN_1_3,
      [&](spv_message_level_t level, const char *source,
          const spv_position_t &position, const char *message) {
        const char *errorLevels[] = {"Fatal",   "Internal error", "Error",
                                     "Warning", "Info",           "Debug"};
        std::cerr << "[" << errorLevels[level] << "] " << (source ? source : "") << " (L"
                  << position.line << ":" << position.column << "): " << message
                  << std::endl;
      },
      spvBlob.data(), spvBlob.size(), false);

  if (context == nullptr) {
    throw std::runtime_error("Can't build module");
  }

  auto tracePass = spvtools::opt::InstBasicBlockTracePass(traceWithU64);
  std::map<int, int> id2TraceIdx;
  uint32_t numBBLabeled = 0;

  tracePass.registerBasicBlockCorrespondenceCallback([&id2TraceIdx](const std::map<int, int> *id2TraceIdxMap) {
    id2TraceIdx = *id2TraceIdxMap;
  });
  tracePass.registerBasicBlockCountRetrievalCallback([&numBBLabeled](uint32_t numLabeled) {
    numBBLabeled = numLabeled;
  });

  
  auto status = tracePass.Run(context.get());

  assert(numBBLabeled == id2TraceIdx.size());

  spvBlob.clear();
  context->module()->ToBinary(&spvBlob, /* skip_nop = */ true);

  return id2TraceIdx;
}
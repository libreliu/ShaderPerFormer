#pragma once

#include "Common.hpp"
#include "spv/Type.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace spvtools::opt {
class Instruction;
class IRContext;
}; // namespace spvtools::opt

namespace vkExecute::spv {
class Tokenizer {
public:
  inline Tokenizer(
    bool compactTypes, bool entrypointOnly, bool convertExtInsts, bool relativeInstIdPos
  ) :
    compactTypes(compactTypes),
    entrypointOnly(entrypointOnly),
    convertExtInsts(convertExtInsts),
    relativeInstIdPos(relativeInstIdPos) {
    if (compactTypes) {
      initCommonTypeLUT();
    }
  }

  void loadSpv(BinaryBlob spv) { spvBlob = toSpvBlob(spv); }
  inline BinaryBlob exportSpv() const { return toBinaryBlob(spvBlob); }

  std::tuple<std::vector<int>, std::string> tokenize();
  std::tuple<std::vector<int>, std::vector<TraceCounter_t>, std::string> tokenizeWithTrace(
    std::map<int, int> bbIdxMap,
    std::vector<TraceCounter_t> bbTraceCounters
  );

  std::string deTokenize(const std::vector<int> &tokens);

  inline bool isCompactTypesEnabled() const {
    return compactTypes;
  }

  inline bool isEntrypointOnlyEnabled() const {
    return entrypointOnly;
  }

  inline bool isConvertExtInstsEnabled() const {
    return convertExtInsts;
  }

  inline bool isRelativeInstIdPosEnabled() const {
    return relativeInstIdPos;
  }

private:
  bool compactTypes;
  bool entrypointOnly;
  bool convertExtInsts;
  bool relativeInstIdPos;

  static const int kEntryPointFunctionIdInIdx = 1;

  // unified interface
  std::tuple<std::vector<int>,
             std::optional<std::vector<vkExecute::TraceCounter_t>>, std::string>
  doTokenize(bool withTrace,
             // Params for trace
             std::map<int, int> *bbIdxMap,
             std::vector<TraceCounter_t> *bbTraceCounters);

  // maps ResultId for inst (if any) => inst sequence
  std::unordered_map<uint64_t, int64_t> instSeqByResultId;
  std::unordered_map<uint64_t, int64_t> typeInstTokByResultId;
  void bookKeepInst(spvtools::opt::IRContext *ctx, spvtools::opt::Instruction *inst);

  // TODO: support trace information
  void tokenizeInst(spvtools::opt::IRContext *ctx, std::vector<int> &tokVec,
                    std::vector<TraceCounter_t> *tokTraceVec,
                    TraceCounter_t traceCount,
                    spvtools::opt::Instruction *inst,
                    int curInstIdx);

  void tokenizeStringLiteral(std::vector<int> &tokVec, std::string literal);
  void tokenizeInt64Literal(std::vector<int> &tokVec, int64_t integer);
  void tokenizeUInt64Literal(std::vector<int> &tokVec, uint64_t integer);
  void tokenizeDoubleLiteral(std::vector<int> &tokVec, double number);

  void initCommonTypeLUT();
  std::vector<std::unique_ptr<Type::Type>> commonTypes;

  // Byte encoded literal area - [256, 512) - 256 total
  // => integer / floating points are encoded with respect to raw string
  // => and for raw string we have a dedicated starting symbol kLiteralStart
  // Special Symbol Area: [1000, 2000) - 999 total
  // OpCode Area: [2000, 20000) - 17999 total
  // => this is to make room for KHR / Intel / NV / AMD extension things
  // Instruction Id: [20000, 40000)
  //
  // The xxxEnd pasts the valid range by 1
  enum SymbolOffsets {
    SymbolBegin = 256,

    ByteEncodedLiteralBegin = 256,
    ByteEncodedLiteralEnd = 512,

    SpecialSymbolBegin = 1000,
    SpecialSymbolEnd = 2000,

    OpCodeBegin = 2000,
    OpCodeEnd = 20000,

    IdBegin = 20000,
    IdRelativeZero = 30000,
    IdEnd = 40000,

    SymbolEnd = IdEnd
  };

  // TODO: consider what could be helpful
  enum SpecialSymbols {
    Pad = 0,
    BoS = 1,
    EoS = 2,
    Mask = 3,
    Sep = 4,
    Unk = 5,
    Cls = 6

    // LiteralStart = 100,
    // LiteralEnd = 101
  };

  static constexpr const char *specialSymbolNames[] = {
      "Pad", "Bos", "EoS", "Mask", "Sep", "Unk", "Cls"};

  std::vector<uint32_t> spvBlob;
};
}; // namespace vkExecute::spv
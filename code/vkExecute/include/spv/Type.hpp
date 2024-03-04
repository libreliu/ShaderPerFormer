#pragma once

#include "Common.hpp"
#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.hpp11"
#include <memory>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

// forward declaration
// to keep things separate
namespace spvtools::opt {
class IRContext;
class Instruction;
namespace analysis {
class Type;
}
} // namespace spvtools::opt

namespace vkExecute::spv {

class ModuleBuilder;

// Currently we maintain a minimal tree to the desire type
// and implement folding
namespace Type {

struct Void;
struct Bool;
struct Integer;
struct Float;
struct Vector;
struct Matrix;
struct Image;
struct Sampler;
struct SampledImage;
struct Array;
struct RuntimeArray;
struct Struct;
struct Opaque;
struct Pointer;
struct Function;
struct Event;
struct DeviceEvent;
struct ReserveId;
struct Queue;
struct Pipe;
struct ForwardPointer;
struct PipeStorage;
struct NamedBarrier;
struct AccelerationStructureNV;
struct CooperativeMatrixNV;
struct RayQueryKHR;
struct HitObjectNV;

// in correspondance with spvtools::opt::analysis::Type::Kind
enum Kind {
  kVoid,
  kBool,
  kInteger,
  kFloat,
  kVector,
  kMatrix,
  kImage,
  kSampler,
  kSampledImage,
  kArray,
  kRuntimeArray,
  kStruct,
  kOpaque,
  kPointer,
  kFunction,
  kEvent,
  kDeviceEvent,
  kReserveId,
  kQueue,
  kPipe,
  kForwardPointer,
  kPipeStorage,
  kNamedBarrier,
  kAccelerationStructureNV,
  kCooperativeMatrixNV,
  kRayQueryKHR,
  kHitObjectNV,
  kLast
};

struct Type {
  // Known limitations:
  // - No support for Forward pointer reference
  // - No support for variable sized array
  //   - Including OpTypeArray with variable length
  //   - Including OpTypeArray with OpSpecConstant - no specialization currently
  //   supported
  static std::unique_ptr<Type> create(const spvtools::opt::analysis::Type *type,
                                      spvtools::opt::IRContext *);

  // these are widely used
  static std::unique_ptr<Type> createInt32();
  static std::unique_ptr<Type> createUInt32();
  static std::unique_ptr<Type> createF32();

  virtual std::string str() const = 0;
  virtual Kind getKind() const = 0;
  virtual std::unique_ptr<Type> clone() const = 0;
  virtual ~Type() = default;
  virtual bool operator==(const Type &rhs) const = 0;
  inline bool operator!=(const Type &rhs) const { return !(*this == rhs); }

// clang-format off
#define DeclareCastMethod(target)                               \
  virtual target* as##target() { return nullptr; }              \
  virtual const target* as##target() const { return nullptr; }

  DeclareCastMethod(Void)
  DeclareCastMethod(Bool)
  DeclareCastMethod(Integer)
  DeclareCastMethod(Float)
  DeclareCastMethod(Vector)
  DeclareCastMethod(Matrix)
  DeclareCastMethod(Image)
  DeclareCastMethod(Sampler)
  DeclareCastMethod(SampledImage)
  DeclareCastMethod(Array)
  DeclareCastMethod(RuntimeArray)
  DeclareCastMethod(Struct)
  DeclareCastMethod(Opaque)
  DeclareCastMethod(Pointer)
  DeclareCastMethod(Function)
  DeclareCastMethod(Event)
  DeclareCastMethod(DeviceEvent)
  DeclareCastMethod(ReserveId)
  DeclareCastMethod(Queue)
  DeclareCastMethod(Pipe) 
  DeclareCastMethod(ForwardPointer)
  DeclareCastMethod(PipeStorage)
  DeclareCastMethod(NamedBarrier)
  DeclareCastMethod(AccelerationStructureNV)
  DeclareCastMethod(CooperativeMatrixNV)
  DeclareCastMethod(RayQueryKHR)
  DeclareCastMethod(HitObjectNV)
#undef DeclareCastMethod
};

// clang-format on

struct Integer : Type {
  int width;
  int signedness;

  inline Integer(int width, int signedness)
      : width(width), signedness(signedness) {}

  Integer(const Integer &) = default;

  std::string str() const override {
    return "Integer<" + std::to_string(width) + ", " +
           std::to_string(signedness) + ">";
  }
  Kind getKind() const override { return kInteger; }
  Integer *asInteger() override { return this; }
  const Integer *asInteger() const override { return this; }
  std::unique_ptr<Type> clone() const override {
    return std::make_unique<Integer>(width, signedness);
  }

  inline bool operator==(const Type &rhs) const override {
    if (rhs.getKind() != this->getKind()) {
      return false;
    } else {
      auto rhsTyped = rhs.asInteger();
      return rhsTyped->signedness == signedness && rhsTyped->width == width;
    }
  }
};

struct Float : Type {
  int width;

  inline Float(int width) : width(width) {}

  Float(const Float &) = default;

  std::string str() const override {
    return "Float<" + std::to_string(width) + ">";
  }
  Kind getKind() const override { return kFloat; }
  Float *asFloat() override { return this; }
  const Float *asFloat() const override { return this; }
  std::unique_ptr<Type> clone() const override {
    return std::make_unique<Float>(width);
  }

  inline bool operator==(const Type &rhs) const override {
    if (rhs.getKind() != this->getKind()) {
      return false;
    } else {
      auto rhsTyped = rhs.asFloat();
      return rhsTyped->width == width;
    }
  }
};

#define DefineParameterlessType(type, name)                                    \
  struct type : public Type {                                                  \
  public:                                                                      \
    type() : Type() {}                                                         \
    type(const type &) = default;                                              \
                                                                               \
    virtual std::string str() const override { return #name; }                 \
                                                                               \
    virtual Kind getKind() const override { return k##type; }                  \
    type *as##type() override { return this; }                                 \
    const type *as##type() const override { return this; }                     \
    std::unique_ptr<Type> clone() const override {                             \
      return std::make_unique<type>();                                         \
    }                                                                          \
    inline bool operator==(const Type &rhs) const override {                   \
      return rhs.getKind() == getKind();                                       \
    }                                                                          \
  }

DefineParameterlessType(Void, Void);
DefineParameterlessType(Bool, Bool);
#undef DefineParameterlessType

// composite types
struct Vector : Type {
  std::unique_ptr<Type> componentType;
  int count;

  inline Vector(int count, Type &comp) {
    this->count = count;
    componentType = comp.clone();
  }

  inline Vector(const Vector &other) {
    count = other.count;
    componentType = other.componentType->clone();
  }

  std::string str() const override {
    return "Vector<" + std::to_string(count) + ", " + componentType->str() +
           ">";
  }

  Kind getKind() const override { return kVector; }
  Vector *asVector() override { return this; }
  const Vector *asVector() const override { return this; }
  std::unique_ptr<Type> clone() const override {
    return std::make_unique<Vector>(count, *componentType.get());
  }

  inline bool operator==(const Type &rhs) const override {
    if (rhs.getKind() != this->getKind()) {
      return false;
    } else {
      auto rhsTyped = rhs.asVector();
      if (count != rhsTyped->count) {
        return false;
      }

      return *componentType == *rhsTyped->componentType;
    }
  }
};

struct Matrix : Type {
  int cols;
  std::unique_ptr<Type> colType;

  inline Matrix(int cols, Type &colType) {
    this->cols = cols;
    this->colType = colType.clone();
  }

  inline Matrix(const Matrix &other) {
    cols = other.cols;
    colType = other.colType->clone();
  }

  std::string str() const override { return "Matrix"; }
  Kind getKind() const override { return kMatrix; }
  Matrix *asMatrix() override { return this; }
  const Matrix *asMatrix() const override { return this; }
  std::unique_ptr<Type> clone() const override {
    return std::make_unique<Matrix>(cols, *colType.get());
  }

  inline bool operator==(const Type &rhs) const override {
    if (rhs.getKind() != this->getKind()) {
      return false;
    } else {
      auto rhsTyped = rhs.asMatrix();
      if (cols != rhsTyped->cols) {
        return false;
      }

      return *colType == *rhsTyped->colType;
    }
  }
};

// Pointer to self (ForwardPointer) is not considered at this moment
struct Struct : Type {
  std::vector<std::unique_ptr<Type>> members;
  std::vector<uint32_t> memberOffsets;

  inline Struct() {}
  inline Struct(const Struct &other) {
    for (auto &tyPtr : other.members) {
      members.push_back(tyPtr->clone());
    }

    memberOffsets = other.memberOffsets;
  }

  void addMember(Type &typePtr) {
    assert(memberOffsets.size() == 0 &&
           "Don't mix between unoffseted and offseted struct");
    members.push_back(typePtr.clone());
  }

  void addMember(Type &typePtr, uint32_t offset) {
    assert(memberOffsets.size() == members.size() &&
           "Don't mix between unoffseted and offseted struct");

    members.push_back(typePtr.clone());
    memberOffsets.push_back(offset);
  }

  std::string str() const override {
    std::stringstream ss;
    ss << "Struct<";

    bool first = true;
    size_t i = 0;
    for (auto &member : members) {
      if (first) {
        first = false;
      } else {
        ss << ", ";
      }

      if (memberOffsets.size() > 0) {
        ss << memberOffsets[i++] << ": ";
      }
      ss << member->str();
    }
    ss << ">";
    return ss.str();
  }
  Kind getKind() const override { return kStruct; }
  Struct *asStruct() override { return this; }
  const Struct *asStruct() const override { return this; }
  std::unique_ptr<Type> clone() const override {
    auto newStruct = std::make_unique<Struct>();
    for (auto &memPtr : members) {
      newStruct->addMember(*memPtr);
    }
    newStruct->memberOffsets = memberOffsets;

    return newStruct;
  }

  inline bool operator==(const Type &rhs) const override {
    if (rhs.getKind() != this->getKind()) {
      return false;
    } else {
      auto rhsTyped = rhs.asStruct();
      if (members.size() != rhsTyped->members.size()) {
        return false;
      }

      if (memberOffsets.size() != rhsTyped->memberOffsets.size()) {
        return false;
      } else if (memberOffsets.size() > 0) {
        assert(memberOffsets.size() == members.size());
        for (size_t i = 0; i < memberOffsets.size(); i++) {
          if (memberOffsets[i] != rhsTyped->memberOffsets[i]) {
            return false;
          }
        }
      }

      for (size_t i = 0; i < members.size(); i++) {
        if (*members[i] != *(rhsTyped->members[i])) {
          return false;
        }
      }

      return true;
    }
  }

  // See
  // https://registry.khronos.org/vulkan/specs/1.3/html/vkspec.html#interfaces-resources-layout
  // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#ShaderValidation
  void emitOffsetDecorations(ModuleBuilder *mBuilder) const;
};

struct Array : Type {
  std::unique_ptr<Type> elementType;
  int count;

  // info from annotations
  std::optional<uint32_t> arrayStride;

  inline Array(int count, Type &comp, std::optional<uint32_t> stride) {
    this->count = count;
    elementType = comp.clone();
    arrayStride = stride;
  }

  inline Array(const Array &other) {
    count = other.count;
    elementType = other.elementType->clone();
    arrayStride = other.arrayStride;
  }

  std::string str() const override {
    return "Array<" + std::to_string(count) + ", " + elementType->str() + ", " +
           (arrayStride.has_value() ? std::to_string(arrayStride.value())
                                    : "None") +
           ">";
  }

  Kind getKind() const override { return kArray; }
  Array *asArray() override { return this; }
  const Array *asArray() const override { return this; }
  std::unique_ptr<Type> clone() const override {
    return std::make_unique<Array>(count, *elementType.get(), arrayStride);
  }

  inline bool operator==(const Type &rhs) const override {
    if (rhs.getKind() != this->getKind()) {
      return false;
    } else {
      auto rhsTyped = rhs.asArray();
      if (count != rhsTyped->count) {
        return false;
      }

      if (arrayStride.has_value() != rhsTyped->arrayStride.has_value()) {
        return false;
      }

      if (arrayStride.has_value() &&
          arrayStride.value() != rhsTyped->arrayStride.value()) {
        return false;
      }

      return *elementType == *rhsTyped->elementType;
    }
  }

  void emitStrideDecorations(ModuleBuilder *mBuilder) const;
};

struct Pointer : Type {
  unsigned int storageClass;
  std::unique_ptr<Type> pointeeType;

  inline Pointer(unsigned int storageClass, Type &pointeeType) {
    this->storageClass = storageClass;
    this->pointeeType = pointeeType.clone();
  }

  inline Pointer(const Pointer &other) {
    this->storageClass = other.storageClass;
    this->pointeeType = other.pointeeType->clone();
  }

  std::string str() const override {
    return "Pointer<" + std::to_string(storageClass) + ", " +
           pointeeType->str() + ">";
  }
  Kind getKind() const override { return kPointer; }
  Pointer *asPointer() override { return this; }
  const Pointer *asPointer() const override { return this; }
  std::unique_ptr<Type> clone() const override {
    return std::make_unique<Matrix>(storageClass, *pointeeType.get());
  }

  inline bool operator==(const Type &rhs) const override {
    if (rhs.getKind() != this->getKind()) {
      return false;
    } else {
      auto rhsTyped = rhs.asPointer();
      if (storageClass != rhsTyped->storageClass) {
        return false;
      }

      return *pointeeType == *rhsTyped->pointeeType;
    }
  }
};

struct Function : Type {
  std::unique_ptr<Type> returnType;
  std::vector<std::unique_ptr<Type>> parameterTypes;

  inline Function(Type &retType) { this->returnType = retType.clone(); }

  inline Function(const Function &other) {
    for (auto &tyPtr : other.parameterTypes) {
      parameterTypes.push_back(tyPtr->clone());
    }
  }

  void addParameter(Type &typePtr) {
    parameterTypes.push_back(typePtr.clone());
  }

  std::string str() const override {
    std::stringstream ss;
    ss << "Function<";
    ss << returnType->str();
    for (auto &param : parameterTypes) {
      ss << ", ";
      ss << param->str();
    }
    ss << ">";
    return ss.str();
  }

  Kind getKind() const override { return kFunction; }
  Function *asFunction() override { return this; }
  const Function *asFunction() const override { return this; }

  std::unique_ptr<Type> clone() const override {
    auto newFunc = std::make_unique<Function>(*this->returnType);
    for (auto &param : parameterTypes) {
      newFunc->addParameter(*param);
    }
    return newFunc;
  }

  inline bool operator==(const Type &rhs) const override {
    if (rhs.getKind() != this->getKind()) {
      return false;
    } else {
      auto rhsTyped = rhs.asFunction();
      if (*returnType != *rhsTyped->returnType) {
        return false;
      }

      if (parameterTypes.size() != rhsTyped->parameterTypes.size()) {
        return false;
      }

      for (size_t i = 0; i < parameterTypes.size(); i++) {
        if (*parameterTypes[i] != *(rhsTyped->parameterTypes[i])) {
          return false;
        }
      }

      return true;
    }
  }
};

// This works for *finite* composite types
// -> Things like OpTypeRuntimeArray won't work
std::vector<std::unique_ptr<Type>> getFiniteCompositeSubTypes(const Type *type);

} // namespace Type
} // namespace vkExecute::spv

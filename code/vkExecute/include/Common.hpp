#pragma once

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define VKEXECUTE_WARN(X, ...)                                                 \
  fprintf(stderr, "vkExecute: WARN: " X "\n", ##__VA_ARGS__)
#define VKEXECUTE_LOG(X, ...) printf("vkExecute: LOG: " X "\n", ##__VA_ARGS__)

namespace vkExecute {
using BinaryBlob = std::vector<char>;
using TraceCounter_t = uint64_t;

// "Peels" off the last type of a tuple
template <class Tuple, std::size_t... I>
auto tuple_tail_impl(Tuple&& t, std::index_sequence<I...>) {
    return std::make_tuple(std::get<I>(std::forward<Tuple>(t))...);
}

template <class... Args>
auto tuple_tail(std::tuple<Args...>&& t) {
    return tuple_tail_impl(std::move(t), std::make_index_sequence<sizeof...(Args) - 1>{});
}

template <typename... Args>
auto printErrorsTuple(std::tuple<Args...> t) {
    std::string error = std::get<sizeof...(Args) - 1>(t);  // get the last argument
    std::cout << "Error: " << error << std::endl;

    // Return the rest of the tuple if have multiple arguments
    return tuple_tail(std::move(t));
}

// print errors and leave result
template <typename T> T printErrors(std::tuple<T, std::string> result) {
  std::cerr << std::get<1>(result);
  return std::get<0>(result);
}

template <typename ArrayT> int32_t argmax(ArrayT &arr) {
  if (arr.size() <= 0) {
    return -1;
  }

  int32_t curIdx = 0;
  for (int32_t i = 1; i < arr.size(); i++) {
    if (arr[i] > arr[curIdx]) {
      curIdx = i;
    }
  }

  return curIdx;
}

// TODO: use modern cpp, split out a separate sampler interface
template <typename ArrayT> int32_t sampleOnce(ArrayT &arr) {
  if (arr.size() <= 0) {
    return -1;
  }

  int32_t sampleIdx = 0;
  double cumuProb = arr[0];
  for (int32_t i = 1; i < arr.size(); i++) {
    cumuProb += arr[i];
    double replaceProb = arr[i] / cumuProb;
    if (((double)std::rand() / (RAND_MAX)) <= replaceProb) {
      sampleIdx = i;
    }
  }

  return sampleIdx;
}

inline std::u8string toU8String(std::string s) {
  return std::u8string(s.begin(), s.end());
}

inline std::string toString(std::u8string s) {
  return std::string(s.begin(), s.end());
}

inline BinaryBlob toBinaryBlob(const std::vector<uint32_t> &u32Blob) {
  BinaryBlob blob;
  blob.resize(u32Blob.size() * sizeof(uint32_t));
  memcpy(blob.data(), u32Blob.data(), blob.size());

  return blob;
}

// I know this is silly, but creating binding for vector<u32> requires separate
// work so I put this thing here
inline std::vector<uint32_t> toSpvBlob(const BinaryBlob &chrBlob) {
  if (chrBlob.size() % 4 != 0) {
    throw std::runtime_error("Illegal SPIR-V blob - not pad to multiple of 4");
  }

  std::vector<uint32_t> spvBlob;
  spvBlob.resize(chrBlob.size() / sizeof(uint32_t));
  memcpy(spvBlob.data(), chrBlob.data(), chrBlob.size());

  return spvBlob;
}

// This is only movable, not copy-constructible
template <typename CompType, int NumComp> struct ImageBlob {
public:
  using ComponentType = CompType;

  ImageBlob(const ImageBlob &other) = delete;
  ImageBlob(ImageBlob &&other) {
    d = other.d;
    w = other.w;
    h = other.h;

    other.d = nullptr;
    other.w = 0;
    other.h = 0;
  };
  ImageBlob &operator=(const ImageBlob &other) = delete;
  ImageBlob &operator=(ImageBlob &&other) {
    if (d != nullptr) {
      delete[] d;
    }

    d = other.d;
    w = other.w;
    h = other.h;

    other.d = nullptr;
    other.w = 0;
    other.h = 0;
    return *this;
  };

  ImageBlob(size_t w, size_t h) : w(w), h(h) {
    d = new CompType[w * h * NumComp];
  }

  ~ImageBlob() {
    if (d != nullptr) {
      delete[] d;
    }
  }
  CompType *data() { return d; }
  size_t width() const { return this->w; }
  size_t height() const { return this->h; }
  void setWidth(int width) { w = width; }
  void setHeight(int height) { h = height; }

  size_t compSize() const { return sizeof(CompType); }
  size_t numComp() const { return NumComp; }

private:
  size_t w, h;
  CompType *d;
};

// Only RGBA for now
using RGBAUIntImageBlob = ImageBlob<uint8_t, 4>;

// Uses VkFormat internally
struct ImageData {
  int format;
  int width;
  int height;
  BinaryBlob data;
};

template <typename T> std::string ToString(const T &destContainer) {
  std::stringstream ss;

  bool first = true;
  ss << "[";
  for (auto &elem : destContainer) {
    if (!first) {
      ss << ", ";
    } else {
      first = false;
    }
    ss << std::to_string(elem);
  }

  ss << "]";
  return ss.str();
}

template<typename T, typename U> std::string
ToString(const std::map<T, U> &mapContainer) {
  std::stringstream ss;

  bool first = true;
  ss << "[";
  for (auto &elem : mapContainer) {
    if (!first) {
      ss << ", ";
    } else {
      first = false;
    }
    ss << "(" << std::to_string(elem.first) << ", "
       << std::to_string(elem.second) << ")";
  }

  ss << "]";
  return ss.str();
}

} // namespace vkExecute
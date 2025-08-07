/*
 * Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DATAUTILS_H
#define DATAUTILS_H

#include <cstdint>
#include <string>
#include <vector>
#include <endian.h>

namespace MindStudioDebugger {
namespace  DataUtils {

inline uint64_t UnpackUint64ValueLe(const void* data)
{
    return le64toh(*reinterpret_cast<const uint64_t *>(data));
}
inline uint64_t UnpackUint64ValueBe(const void* data)
{
    return be64toh(*reinterpret_cast<const uint64_t *>(data));
}

int64_t SizeToS64(size_t v);
std::string U64ToHexString(uint64_t v);

class BFloat16 {
public:
    static constexpr uint16_t VALUE_MASK = 0x7fff;
    static constexpr uint16_t INF_VALUE = 0x7f80;
    static constexpr uint16_t NAN_VALUE = 0x7fc0;
    static constexpr uint16_t TRUE_VALUE = 0x3c00;
    static constexpr uint32_t F32_INF_VALUE = 0x7f800000;

    BFloat16() = default;
    ~BFloat16() = default;
    BFloat16(const BFloat16 &other) noexcept = default;
    BFloat16(BFloat16 &&other) noexcept = default;
    BFloat16 &operator=(const BFloat16 &other) noexcept = default;
    BFloat16 &operator=(BFloat16 &&other) noexcept = default;

    explicit BFloat16(float f32);
    explicit operator float() const;
    BFloat16 operator+(const BFloat16& other) const
        { return BFloat16(static_cast<float>(*this) + static_cast<float>(other)); }
    float operator+(const float other) const { return static_cast<float>(*this) + other; }
private:
    uint16_t value_;
};

inline float operator+(const float fp32, const BFloat16& bf16)
{
    return fp32 + static_cast<float>(bf16);
}

using ShapeBaseType = int64_t;
using TensorShape = std::vector<ShapeBaseType>;

enum DataType : int {
    DT_UNDEFINED = 0,
    DT_FLOAT = 1,
    DT_FLOAT16 = 2,
    DT_INT8 = 3,
    DT_UINT8 = 4,
    DT_INT16 = 5,
    DT_UINT16 = 6,
    DT_INT32 = 7,
    DT_INT64 = 8,
    DT_UINT32 = 9,
    DT_UINT64 = 10,
    DT_BOOL = 11,
    DT_DOUBLE = 12,
    DT_STRING = 13,
    DT_DUAL_SUB_INT8 = 14,
    DT_DUAL_SUB_UINT8 = 15,
    DT_COMPLEX64 = 16,
    DT_COMPLEX128 = 17,
    DT_QINT8 = 18,
    DT_QINT16 = 19,
    DT_QINT32 = 20,
    DT_QUINT8 = 21,
    DT_QUINT16 = 22,
    DT_RESOURCE = 23,
    DT_STRING_REF = 24,
    DT_DUAL = 25,
    DT_VARIANT = 26,
    DT_BF16 = 27,
    DT_INT4 = 28,
    DT_UINT1 = 29,
    DT_INT2 = 30,
    DT_UINT2 = 31,
    /* Add before this line */
    DT_MAX
};

enum TensorFormat : int {
    FORMAT_NCHW = 0,
    FORMAT_NHWC = 1,
    FORMAT_ND = 2,
    FORMAT_NC1HWC0 = 3,
    FORMAT_FRACTAL_Z = 4,
    FORMAT_NC1C0HWPAD = 5,
    FORMAT_NHWC1C0 = 6,
    FORMAT_FSR_NCHW = 7,
    FORMAT_FRACTAL_DECONV = 8,
    FORMAT_C1HWNC0 = 9,
    FORMAT_FRACTAL_DECONV_TRANSPOSE = 10,
    FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS = 11,
    FORMAT_NC1HWC0_C04 = 12,
    FORMAT_FRACTAL_Z_C04 = 13,
    FORMAT_CHWN = 14,
    FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15,
    FORMAT_HWCN = 16,
    FORMAT_NC1KHKWHWC0 = 17,
    FORMAT_BN_WEIGHT = 18,
    FORMAT_FILTER_HWCK = 19,
    FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20,
    FORMAT_HASHTABLE_LOOKUP_KEYS = 21,
    FORMAT_HASHTABLE_LOOKUP_VALUE = 22,
    FORMAT_HASHTABLE_LOOKUP_OUTPUT = 23,
    FORMAT_HASHTABLE_LOOKUP_HITS = 24,
    FORMAT_C1HWNCOC0 = 25,
    FORMAT_MD = 26,
    FORMAT_NDHWC = 27,
    FORMAT_FRACTAL_ZZ = 28,
    FORMAT_FRACTAL_NZ = 29,
    FORMAT_NCDHW = 30,
    FORMAT_DHWCN = 31,
    FORMAT_NDC1HWC0 = 32,
    FORMAT_FRACTAL_Z_3D = 33,
    FORMAT_CN = 34,
    FORMAT_NC = 35,
    FORMAT_DHWNC = 36,
    FORMAT_FRACTAL_Z_3D_TRANSPOSE = 37,
    FORMAT_FRACTAL_ZN_LSTM = 38,
    FORMAT_FRACTAL_Z_G = 39,
    FORMAT_RESERVED = 40,
    FORMAT_ALL = 41,
    FORMAT_NULL = 42,
    FORMAT_ND_RNN_BIAS = 43,
    FORMAT_FRACTAL_ZN_RNN = 44,
    FORMAT_YUV = 45,
    FORMAT_YUV_A = 46,
    FORMAT_NCL = 47,
    FORMAT_FRACTAL_Z_WINO = 48,
    FORMAT_C1HWC0 = 49,
    /* Add before this line */
    FORMAT_MAX
};

size_t SizeOfDType(DataType type);
std::string GetDTypeString(DataType dtype);
std::string GetFormatString(TensorFormat fmt);
std::string GetShapeString(const TensorShape& shape);

}
}

#endif
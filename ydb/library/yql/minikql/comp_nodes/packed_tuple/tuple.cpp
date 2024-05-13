#include "tuple.h"

#include <algorithm>
#include <queue>

#include <ydb/library/yql/minikql/mkql_node.h>
#include <ydb/library/yql/public/udf/udf_data_type.h>
#include <ydb/library/yql/public/udf/udf_types.h>
#include <ydb/library/yql/public/udf/udf_value.h>

#include <util/generic/bitops.h>
#include <util/generic/buffer.h>

#include "hashes_calc.h"
#include "packing.h"

namespace NKikimr {
namespace NMiniKQL {
namespace NPackedTuple {

namespace {

void bit_transpose(ui8 dst[], const ui8 *src[], const size_t row_size) {
    ui64 x;
    for (size_t ind = 0; ind < 8; ++ind) {
        x = (x << 8) | *src[ind];
    }
    x = x & 0xAA55AA55AA55AA55LL | (x & 0x00AA00AA00AA00AALL) << 7 |
        (x >> 7) & 0x00AA00AA00AA00AALL;
    x = x & 0xCCCC3333CCCC3333LL | (x & 0x0000CCCC0000CCCCLL) << 14 |
        (x >> 14) & 0x0000CCCC0000CCCCLL;
    x = x & 0xF0F0F0F00F0F0F0FLL | (x & 0x00000000F0F0F0F0LL) << 28 |
        (x >> 28) & 0x00000000F0F0F0F0LL;

    // byte reversion
    x = (x & 0xAAAAAAAAAAAAAAAA) >> 1 | (x & 0x5555555555555555) << 1;
    x = (x & 0xCCCCCCCCCCCCCCCC) >> 2 | (x & 0x3333333333333333) << 2;
    x = (x & 0xF0F0F0F0F0F0F0F0) >> 4 | (x & 0x0F0F0F0F0F0F0F0F) << 4;

    for (size_t ind = 0; ind != 8; ++ind) {
        dst[ind * row_size] = x;
        x = x >> 8;
    }
}

} // namespace

THolder<TTupleLayout>
TTupleLayout::Create(const std::vector<TColumnDesc> &columns) {

    if (NX86::HaveAVX2())
        return MakeHolder<TTupleLayoutFallback<NSimd::TSimdAVX2Traits>>(
            columns);

    if (NX86::HaveSSE42())
        return MakeHolder<TTupleLayoutFallback<NSimd::TSimdSSE42Traits>>(
            columns);

    return MakeHolder<TTupleLayoutFallback<NSimd::TSimdFallbackTraits>>(
        columns);
}

template <typename TTraits>
TTupleLayoutFallback<TTraits>::TTupleLayoutFallback(
    const std::vector<TColumnDesc> &columns)
    : TTupleLayout(columns) {

    for (ui32 i = 0, idx = 0; i < OrigColumns.size(); ++i) {
        auto &col = OrigColumns[i];

        col.OriginalIndex = idx;

        if (col.SizeType == EColumnSizeType::Variable) {
            // we cannot handle (rare) overflow strings unless we have at least
            // space for header; size of inlined strings is limited to 254
            // bytes, limit maximum inline data size
            col.DataSize = std::max<ui32>(1 + 2 * sizeof(ui32),
                                          std::min<ui32>(255, col.DataSize));
            idx += 2; // Variable-size takes two columns: one for offsets, and
                      // another for payload
        } else {
            idx += 1;
        }

        if (col.Role == EColumnRole::Key) {
            KeyColumns.push_back(col);
        } else {
            PayloadColumns.push_back(col);
        }
    }

    KeyColumnsNum = KeyColumns.size();

    auto ColumnDescLess = [](const TColumnDesc &a, const TColumnDesc &b) {
        if (a.SizeType != b.SizeType) // Fixed first
            return a.SizeType == EColumnSizeType::Fixed;

        if (a.DataSize == b.DataSize)
            // relative order of (otherwise) same key columns must be preserved
            return a.OriginalIndex < b.OriginalIndex;

        return a.DataSize < b.DataSize;
    };

    std::sort(KeyColumns.begin(), KeyColumns.end(), ColumnDescLess);
    std::sort(PayloadColumns.begin(), PayloadColumns.end(), ColumnDescLess);

    ui32 currOffset = 4; // crc32 hash in the beginning
    KeyColumnsOffset = currOffset;
    KeyColumnsFixedNum = KeyColumnsNum;
    KeyColumnsFixedEnd = 0;
    for (ui32 i = 0; i < KeyColumnsNum; ++i) {
        auto &col = KeyColumns[i];

        if (col.SizeType ==
            EColumnSizeType::Variable) { // && KeyColumnsFixedEnd == 0
            KeyColumnsFixedEnd = currOffset;
            KeyColumnsFixedNum = i;
        }

        col.ColumnIndex = i;
        col.Offset = currOffset;
        Columns.push_back(col);
        currOffset += col.DataSize;
    }

    KeyColumnsEnd = currOffset;
    if (KeyColumnsFixedEnd == 0)
        KeyColumnsFixedEnd = KeyColumnsEnd;

    KeyColumnsSize = KeyColumnsEnd - KeyColumnsOffset;
    BitmaskOffset = currOffset;

    BitmaskSize = (OrigColumns.size() + 7) / 8;

    currOffset += BitmaskSize;
    BitmaskEnd = currOffset;

    PayloadOffset = currOffset;
    for (ui32 i = 0; i < PayloadColumns.size(); ++i) {
        auto &col = PayloadColumns[i];
        col.ColumnIndex = KeyColumnsNum + i;
        col.Offset = currOffset;
        Columns.push_back(col);
        currOffset += col.DataSize;
    }

    PayloadEnd = currOffset;
    PayloadSize = PayloadEnd - PayloadOffset;

    TotalRowSize = currOffset;

    for (auto &col : Columns) {
        if (col.SizeType == EColumnSizeType::Variable) {
            VariableColumns_.push_back(col);
        } else if (IsPowerOf2(col.DataSize) && col.DataSize <= 16) {
            FixedPOTColumns_[CountTrailingZeroBits(col.DataSize)].push_back(
                col);
        } else {
            FixedNPOTColumns_.push_back(col);
        }
    }

    BlockRows_ = 256; /// TODO: dynamic configure

    std::vector<const TColumnDesc *> block_fallback;
    const bool simd_flag = false;

    const auto manage_block_packing = [&](const std::vector<TColumnDesc>
                                              &columns) {
        size_t cols_size_left = std::accumulate(
            columns.begin(), columns.end(), 0,
            [](size_t prev, const auto &col) { return prev + col.DataSize; });

        std::queue<const TColumnDesc *> next_cols;
        size_t cur_tuple_size = 0;

        for (size_t col_ind = 0;
             col_ind != columns.size() &&
             columns[col_ind].SizeType == EColumnSizeType::Fixed;) {
            cols_size_left -= columns[col_ind].DataSize;
            next_cols.push(&columns[col_ind]);
            cur_tuple_size += columns[col_ind].DataSize;

            ++col_ind;
            if (cur_tuple_size >= TSimd<ui8>::SIZE ||
                next_cols.size() == kSIMDMaxCols || col_ind == columns.size() ||
                columns[col_ind].SizeType != EColumnSizeType::Fixed) {
                const size_t tuple_size =
                    cur_tuple_size > TSimd<ui8>::SIZE
                        ? cur_tuple_size - next_cols.back()->DataSize
                        : cur_tuple_size;
                const size_t tuple_cols =
                    next_cols.size() - (cur_tuple_size > TSimd<ui8>::SIZE);

                if (!simd_flag || tuple_size < TSimd<ui8>::SIZE * 7 / 8 ||
                    tuple_size > TSimd<ui8>::SIZE ||
                    cols_size_left + tuple_size < TSimd<ui8>::SIZE) {
                    cur_tuple_size -= next_cols.front()->DataSize;
                    block_fallback.push_back(next_cols.front());
                    next_cols.pop();
                    continue;
                }

                SIMDDesc simd_desc;
                simd_desc.TupleSize = tuple_size;
                simd_desc.Cols = tuple_cols;
                simd_desc.PermMaskOffset = SIMDPermMasks_.size();
                simd_desc.TuplesPerStore =
                    std::max(1ul, TSimd<ui8>::SIZE / tuple_size);
                simd_desc.RowOffset = next_cols.front()->Offset;

                const TColumnDesc *col_descs[kSIMDMaxCols];
                ui32 col_max_size = 0;
                for (ui8 col_ind = 0; col_ind != simd_desc.Cols; ++col_ind) {
                    col_descs[col_ind] = next_cols.front();
                    col_max_size =
                        std::max(col_max_size, col_descs[col_ind]->DataSize);

                    cur_tuple_size -= next_cols.front()->DataSize;
                    next_cols.pop();
                }

                simd_desc.InnerLoopIters =
                    (TSimd<ui8>::SIZE / col_max_size) /
                    std::max(1ul, size_t(TSimd<ui8>::SIZE / TotalRowSize));

                for (ui8 col_ind = 0, offset = 0; col_ind != simd_desc.Cols;
                     ++col_ind) {
                    const auto &col_desc = col_descs[col_ind];

                    BlockFixedColsSizes_.push_back(col_desc->DataSize);

                    for (ui8 ind = 0; ind != simd_desc.InnerLoopIters; ++ind) {
                        SIMDPermMasks_.push_back(
                            SIMDPack<TTraits>::BuildTuplePerm(
                                col_desc->DataSize,
                                TotalRowSize - col_desc->DataSize, offset,
                                ind * BlockFixedColsSizes_.back() *
                                    simd_desc.TuplesPerStore,
                                true));
                    }

                    BlockColsOffsets_.push_back(offset);
                    offset += col_desc->DataSize;

                    BlockColumnsOrigInds_.push_back(col_desc->OriginalIndex);
                }
            }
        }

        while (!next_cols.empty()) {
            block_fallback.push_back(next_cols.front());
            next_cols.pop();
        }
    };

    manage_block_packing(KeyColumns);
    manage_block_packing(PayloadColumns);

    for (const auto col_desc_p : block_fallback) {
        BlockColsOffsets_.push_back(col_desc_p->Offset);
        BlockFixedColsSizes_.push_back(col_desc_p->DataSize);
        BlockColumnsOrigInds_.push_back(col_desc_p->OriginalIndex);
    }
}

// Columns (SoA) format:
//   for fixed size: packed data
//   for variable size: offset (ui32) into next column; size of colum is
//   rowCount + 1
//
// Row (AoS) format:
//   fixed size: packed data
//   variable size:
//     assumes DataSize <= 255 && DataSize >= 1 + 2*4
//     if size of payload is less than col.DataSize:
//       u8 one byte of size (0..254)
//       u8 [size] data
//       u8 [DataSize - 1 - size] padding
//     if size of payload is greater than DataSize:
//       u8 = 255
//       u32 = offset in overflow buffer
//       u32 = size
//       u8 [DataSize - 1 - 2*4] initial bytes of data
// Data is expected to be consistent with isValidBitmask (0 for fixed-size,
// empty for variable-size)
template <typename TTraits>
void TTupleLayoutFallback<TTraits>::Pack(
    const ui8 **columns, const ui8 **isValidBitmask, ui8 *res,
    std::vector<ui8, TMKQLAllocator<ui8>> &overflow, ui32 start,
    ui32 count) const {

    std::vector<ui64> bitmaskMatrix(BitmaskSize);

    for (; count--; ++start, res += TotalRowSize) {
        ui32 hash = 0;
        auto bitmaskIdx = start / 8;
        auto bitmaskShift = start % 8;

        bool anyOverflow = false;

        for (ui32 i = KeyColumnsFixedNum; i < KeyColumns.size(); ++i) {
            auto &col = KeyColumns[i];
            auto dataOffset = ReadUnaligned<ui32>(columns[col.OriginalIndex] +
                                                  sizeof(ui32) * start);
            auto nextOffset = ReadUnaligned<ui32>(columns[col.OriginalIndex] +
                                                  sizeof(ui32) * (start + 1));
            auto size = nextOffset - dataOffset;

            if (size >= col.DataSize) {
                anyOverflow = true;
                break;
            }
        }
        std::memset(res + BitmaskOffset, 0, BitmaskSize);
        for (ui32 i = 0; i < Columns.size(); ++i) {
            auto &col = Columns[i];

            res[BitmaskOffset + (i / 8)] |=
                ((isValidBitmask[col.OriginalIndex][bitmaskIdx] >>
                  bitmaskShift) &
                 1u)
                << (i % 8);
        }
        for (auto &col : FixedNPOTColumns_) {
            std::memcpy(res + col.Offset,
                        columns[col.OriginalIndex] + start * col.DataSize,
                        col.DataSize);
        }
#define PackPOTColumn(POT)                                                     \
    for (auto &col : FixedPOTColumns_[POT]) {                                  \
        std::memcpy(res + col.Offset,                                          \
                    columns[col.OriginalIndex] + start * (1u << POT),          \
                    1u << POT);                                                \
    }
        PackPOTColumn(0);
        PackPOTColumn(1);
        PackPOTColumn(2);
        PackPOTColumn(3);
        PackPOTColumn(4);
#undef PackPOTColumn
        for (auto &col : VariableColumns_) {
            auto dataOffset = ReadUnaligned<ui32>(columns[col.OriginalIndex] +
                                                  sizeof(ui32) * start);
            auto nextOffset = ReadUnaligned<ui32>(columns[col.OriginalIndex] +
                                                  sizeof(ui32) * (start + 1));
            auto size = nextOffset - dataOffset;
            auto data = columns[col.OriginalIndex + 1] + dataOffset;
            if (size >= col.DataSize) {
                res[col.Offset] = 255;

                auto prefixSize = (col.DataSize - 1 - 2 * sizeof(ui32));
                auto overflowSize = size - prefixSize;
                auto overflowOffset = overflow.size();

                overflow.resize(overflowOffset + overflowSize);

                WriteUnaligned<ui32>(res + col.Offset + 1 + 0 * sizeof(ui32),
                                     overflowOffset);
                WriteUnaligned<ui32>(res + col.Offset + 1 + 1 * sizeof(ui32),
                                     overflowSize);
                std::memcpy(res + col.Offset + 1 + 2 * sizeof(ui32), data,
                            prefixSize);
                std::memcpy(overflow.data() + overflowOffset, data + prefixSize,
                            overflowSize);
            } else {
                Y_DEBUG_ABORT_UNLESS(size < 255);
                res[col.Offset] = size;
                std::memcpy(res + col.Offset + 1, data, size);
                std::memset(res + col.Offset + 1 + size, 0,
                            col.DataSize - (size + 1));
            }
            if (anyOverflow && col.Role == EColumnRole::Key) {
                hash =
                    CalculateCRC32<TTraits, sizeof(ui32)>((ui8 *)&size, hash);
                hash = CalculateCRC32<TTraits>(data, size, hash);
            }
        }

        // isValid bitmap is NOT included into hashed data
        if (anyOverflow) {
            hash = CalculateCRC32<TTraits>(
                res + KeyColumnsOffset, KeyColumnsFixedEnd - KeyColumnsOffset,
                hash);
        } else {
            hash = CalculateCRC32<TTraits>(res + KeyColumnsOffset,
                                           KeyColumnsEnd - KeyColumnsOffset);
        }
        WriteUnaligned<ui32>(res, hash);
    }
}

template <>
void TTupleLayoutFallback<NSimd::TSimdAVX2Traits>::Pack(
    const ui8 **columns, const ui8 **isValidBitmask, ui8 *res,
    std::vector<ui8, TMKQLAllocator<ui8>> &overflow, ui32 start,
    ui32 count) const {

    std::vector<const ui8 *> block_columns;
    for (const auto col_ind : BlockColumnsOrigInds_) {
        block_columns.push_back(columns[col_ind]);
    }

    for (size_t row_ind = 0; row_ind < count; row_ind += BlockRows_) {
        const size_t cur_block_size = std::min(count - row_ind, BlockRows_);
        size_t cols_past = 0;
        for (const auto &simd_block : SIMDBlock_) {

#define CASE(i, j)                                                             \
    case i *kSIMDMaxCols + j:                                                  \
        SIMDPack<NSimd::TSimdAVX2Traits>::PackTupleOrImpl<i, j>(               \
            block_columns.data() + cols_past,                                  \
            res + start * TotalRowSize + simd_block.RowOffset, cur_block_size, \
            BlockFixedColsSizes_.data() + cols_past,                           \
            BlockColsOffsets_.data() + cols_past, TotalRowSize,                \
            SIMDPermMasks_.data() + simd_block.PermMaskOffset, start);         \
        break;

#define MULTI_8_I(C, i)                                                        \
    C(i, 0) C(i, 1) C(i, 2) C(i, 3) C(i, 4) C(i, 5) C(i, 6) C(i, 7)

#define MULTI_8(C, A)                                                          \
    C(A, 0) C(A, 1) C(A, 2) C(A, 3) C(A, 4) C(A, 5) C(A, 6) C(A, 7)

            switch (simd_block.InnerLoopIters * kSIMDMaxCols +
                    simd_block.Cols) {

                MULTI_8(MULTI_8_I, CASE)
            }

            cols_past += simd_block.Cols;
        }

        PackTupleFallbackColImpl(
            block_columns.data() + cols_past, res + start * TotalRowSize,
            BlockColsOffsets_.size() - cols_past, cur_block_size,
            BlockFixedColsSizes_.data() + cols_past,
            BlockColsOffsets_.data() + cols_past, TotalRowSize, start);

        for (ui32 cols_ind = 0; cols_ind < Columns.size(); cols_ind += 8) {
            const ui8 *bitmasks[8];
            const size_t cols = std::min(8ul, Columns.size() - cols_ind);
            for (size_t ind = 0; ind != cols; ++ind) {
                const auto &col = Columns[cols_ind + ind];
                bitmasks[ind] = isValidBitmask[col.OriginalIndex] + start / 8;
            }
            const ui8 zero = 0;
            for (size_t ind = cols; ind != 8; ++ind) {
                bitmasks[ind] =
                    (const ui8 *)&zero; // just anything dereferencing
            }

            const auto advance_masks = [&] {
                for (size_t ind = 0; ind != cols; ++ind) {
                    ++bitmasks[ind];
                }
            };

            const size_t first_full_byte = (8 - start) & 7;

            size_t block_row_ind = 0;

            const auto simple_mask_transpose = [&](const size_t until) {
                for (; block_row_ind < until; ++block_row_ind) {
                    const auto new_start = start + block_row_ind;
                    const auto start = new_start;

                    const auto new_res = res + start * TotalRowSize;
                    const auto res = new_res;

                    const auto bitmaskShift = start % 8;

                    for (size_t col_ind = 0; col_ind != cols; ++col_ind) {
                        const size_t ind = cols_ind + col_ind;
                        res[BitmaskOffset + (ind / 8)] |=
                            ((*bitmasks[col_ind] >> bitmaskShift) & 1u)
                            << (ind % 8);
                    }
                }
            };

            simple_mask_transpose(first_full_byte);
            if (first_full_byte) {
                advance_masks();
            }

            for (; block_row_ind + 7 < cur_block_size; block_row_ind += 8) {
                bit_transpose(res + (start + block_row_ind) * TotalRowSize +
                                  BitmaskOffset + cols_ind / 8,
                              bitmasks, TotalRowSize);
                advance_masks();
            }

            simple_mask_transpose(cur_block_size);
        }

        for (size_t block_row_ind = 0; block_row_ind != cur_block_size;
             ++block_row_ind) {

            const auto new_start = start + block_row_ind;
            const auto start = new_start;

            const auto new_res = res + start * TotalRowSize;
            const auto res = new_res;

            ui32 hash = 0;
            bool anyOverflow = false;

            for (ui32 i = KeyColumnsFixedNum; i < KeyColumns.size(); ++i) {
                auto &col = KeyColumns[i];
                auto dataOffset = ReadUnaligned<ui32>(
                    columns[col.OriginalIndex] + sizeof(ui32) * start);
                auto nextOffset = ReadUnaligned<ui32>(
                    columns[col.OriginalIndex] + sizeof(ui32) * (start + 1));
                auto size = nextOffset - dataOffset;

                if (size >= col.DataSize) {
                    anyOverflow = true;
                    break;
                }
            }

            for (auto &col : VariableColumns_) {
                auto dataOffset = ReadUnaligned<ui32>(
                    columns[col.OriginalIndex] + sizeof(ui32) * start);
                auto nextOffset = ReadUnaligned<ui32>(
                    columns[col.OriginalIndex] + sizeof(ui32) * (start + 1));
                auto size = nextOffset - dataOffset;
                auto data = columns[col.OriginalIndex + 1] + dataOffset;
                if (size >= col.DataSize) {
                    res[col.Offset] = 255;

                    auto prefixSize = (col.DataSize - 1 - 2 * sizeof(ui32));
                    auto overflowSize = size - prefixSize;
                    auto overflowOffset = overflow.size();

                    overflow.resize(overflowOffset + overflowSize);

                    WriteUnaligned<ui32>(res + col.Offset + 1 +
                                             0 * sizeof(ui32),
                                         overflowOffset);
                    WriteUnaligned<ui32>(
                        res + col.Offset + 1 + 1 * sizeof(ui32), overflowSize);
                    std::memcpy(res + col.Offset + 1 + 2 * sizeof(ui32), data,
                                prefixSize);
                    std::memcpy(overflow.data() + overflowOffset,
                                data + prefixSize, overflowSize);
                } else {
                    Y_DEBUG_ABORT_UNLESS(size < 255);
                    res[col.Offset] = size;
                    std::memcpy(res + col.Offset + 1, data, size);
                    std::memset(res + col.Offset + 1 + size, 0,
                                col.DataSize - (size + 1));
                }
                if (anyOverflow && col.Role == EColumnRole::Key) {
                    hash = CalculateCRC32<NSimd::TSimdAVX2Traits, sizeof(ui32)>(
                        (ui8 *)&size, hash);
                    hash = CalculateCRC32<NSimd::TSimdAVX2Traits>(data, size,
                                                                  hash);
                }
            }

            // isValid bitmap is NOT included into hashed data
            if (anyOverflow) {
                hash = CalculateCRC32<NSimd::TSimdAVX2Traits>(
                    res + KeyColumnsOffset,
                    KeyColumnsFixedEnd - KeyColumnsOffset, hash);
            } else {
                hash = CalculateCRC32<NSimd::TSimdAVX2Traits>(
                    res + KeyColumnsOffset, KeyColumnsEnd - KeyColumnsOffset);
            }
            WriteUnaligned<ui32>(res, hash);
        }

        start += cur_block_size;
    }
}

} // namespace NPackedTuple
} // namespace NMiniKQL
} // namespace NKikimr

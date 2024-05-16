#include <ydb/library/yql/minikql/mkql_runtime_version.h>
#include <ydb/library/yql/minikql/comp_nodes/ut/mkql_computation_node_ut.h>
#include <library/cpp/testing/unittest/registar.h>

#include <chrono>
#include <vector>
#include <set>
#include <random>

#include <util/system/fs.h>
#include <util/system/compiler.h>
#include <util/stream/null.h>
#include <util/system/mem_info.h>

#include <ydb/library/yql/minikql/comp_nodes/packed_tuple/tuple.h>
#include <ydb/library/yql/utils/simd/simd.h>

namespace NKikimr {
namespace NMiniKQL {
namespace NPackedTuple {

using namespace std::chrono_literals;
using random_bytes_engine = std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char>;

static bool IsVerbose = true;
#define CTEST (IsVerbose ? Cerr : Cnull)

// Function ro prevent compiler ignore the p value
static void escape(void* p) {
    __asm__ volatile("" : : "g"(p) : "memory");
}

class Benchmarker {
public:
    enum class ConverterType {
        AVX2,
        SSE42,
        Fallback
    };

public:
    Benchmarker(const std::vector<TColumnDesc>& column_descs, ConverterType type = ConverterType::Fallback)
        : col_descs_(column_descs) 
    {
        if (type == ConverterType::AVX2 && NX86::HaveAVX2()) {
            converter_ = MakeHolder<TTupleLayoutFallback<NSimd::TSimdAVX2Traits>>(column_descs);
        } else if (type == ConverterType::SSE42 && NX86::HaveSSE42()) {
            converter_ = MakeHolder<TTupleLayoutFallback<NSimd::TSimdSSE42Traits>>(column_descs);
        } else {
            converter_ = MakeHolder<TTupleLayoutFallback<NSimd::TSimdFallbackTraits>>(column_descs);
        }
    }

    void BenchPack(ui32 rows_count) {
        ui64 total_size = rows_count * converter_->TotalRowSize;
        ui32 columns_num = converter_->Columns.size();

        std::vector<std::vector<ui8>>   inputs;
        std::vector<ui8*>               column_ptrs;
        std::vector<ui8>                shared_valid_bitmask((rows_count + 7) / 8, ~0);
        std::vector<ui8*>               is_valid_bitmask_ptrs;
        
        for (const auto& col: converter_->OrigColumns) {
            if (col.SizeType == EColumnSizeType::Fixed) {
                std::vector<ui8> data(col.DataSize * rows_count, 0);
                std::generate(data.begin(), data.end(), rbe_);

                inputs.emplace_back(std::move(data));
                column_ptrs.emplace_back(inputs.back().data());
                is_valid_bitmask_ptrs.push_back(shared_valid_bitmask.data());
            } else {
                std::vector<ui32> data_offsets(1, 0);
                ui32 data_size = 0;

                for (ui32 i = 0; i < rows_count; ++i) {
                    ui32 sequence_size = 0;
                    if (rng_() % 100 < 5) { // 5% of variable byte strings bigger than DataSize
                        sequence_size = col.DataSize + rng_() % 256;
                    } else { // 95% of variable byte strings <= DataSize
                        sequence_size = std::max<ui32>(rng_() % col.DataSize, 1); // >= 1
                    }
                    data_size += sequence_size;
                    data_offsets.push_back(data_size);
                }

                std::vector<ui8> data(data_size, 0);
                std::generate(data.begin(), data.end(), rbe_);

                std::vector<ui8> byte_data_offsets(data_offsets.size() * sizeof(ui32));
                std::memcpy(byte_data_offsets.data(), data_offsets.data(), byte_data_offsets.size());

                inputs.emplace_back(std::move(byte_data_offsets));
                column_ptrs.emplace_back(inputs.back().data());
                is_valid_bitmask_ptrs.push_back(shared_valid_bitmask.data());

                inputs.emplace_back(std::move(data));
                column_ptrs.emplace_back(inputs.back().data());
                is_valid_bitmask_ptrs.push_back(nullptr);
            }
        }

        std::vector<ui8>                        res(total_size + 64, 0);
        std::vector<ui8, TMKQLAllocator<ui8>>   overflow;

        const ui8** columns = const_cast<const ui8**>(column_ptrs.data());
        const ui8** isValidBitmask = const_cast<const ui8**>(is_valid_bitmask_ptrs.data());

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        converter_->Pack(columns, isValidBitmask, res.data(), overflow, 0, rows_count);
        escape(res.data());
        escape(overflow.data());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        ui64 microseconds = std::max<ui64>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(), 1);

        auto calc_size = [&total_size] () -> std::pair<double, const char*> {
            if (total_size > 1024 * 1024 * 1024) {
                return {static_cast<double>(total_size) / (1024 * 1024 * 1024), " [GB]"};
            } else if (total_size > 1024 * 1024) {
                return {static_cast<double>(total_size) / (1024 * 1024), " [MB]"};
            } else {
                return {static_cast<double>(total_size) / 1024, " [KB]"};
            }
        };
        auto [trunc_total_size, suffix] = calc_size();

        CTEST << "------------- Benchmark for Pack -------------" << Endl;
        CTEST << "Layout: { key columns count = " << converter_->KeyColumns.size()
              << ", payload columns count = " << converter_->PayloadColumns.size() << " }" << Endl;
        CTEST << "Count of columns = " << columns_num << Endl;
        CTEST << "Count of rows = " << rows_count << Endl;
        CTEST << "TotalRowSize = " << converter_->TotalRowSize << Endl;
        CTEST << "Data size = " << trunc_total_size << suffix << Endl;
        CTEST << "Calculating speed = " << total_size / microseconds << "MB/sec" << Endl;
        CTEST << "----------------------------------------------" << Endl << Endl;
    }

// Not implemented yet
#if 0
    void BenchUnpack(ui32 rows_count) {
    }
#endif

private:
    std::vector<TColumnDesc>    col_descs_;
    THolder<TTupleLayout>       converter_;
    std::mt19937                rng_;
    random_bytes_engine         rbe_;
};

void PrintType(Benchmarker::ConverterType type) {
    if (type == Benchmarker::ConverterType::Fallback) {
        Cerr << "Fallback" << Endl;
    } else if (type == Benchmarker::ConverterType::SSE42) {
        Cerr << "SSE42" << Endl;
    } else {
        Cerr << "AVX2" << Endl;
    }
}

Y_UNIT_TEST_SUITE(BenchConversion) {

#if 0
Y_UNIT_TEST(BenchPackFixed) {

    std::mt19937 rng;
    ui64 sizes[4] = {1, 2, 4, 8};
    double key_cols_mult = 0.10; // 10% of cols are keys

    for (auto converter_type: {Benchmarker::ConverterType::Fallback, Benchmarker::ConverterType::SSE42, Benchmarker::ConverterType::AVX2}) {
        Cerr << "--------------------------------------------------" << Endl;
        PrintType(converter_type);
        for (ui64 cols_count: {8, 32, 128}) {
            for (ui64 rows_count: {10'000, 100'000, 1'000'000, 10'000'000}) {
                if (cols_count * rows_count > 1e9) {
                    continue; // too large dataset
                }
                ui64 key_cols_count = std::max<ui64>(static_cast<ui64>(key_cols_mult * cols_count), 1);
                ui64 payload_cols_count = cols_count - key_cols_count;
                std::vector<TColumnDesc> columns;

                for (ui64 i = 0; i < key_cols_count; ++i) {
                    columns.emplace_back();
                    columns.back().Role = EColumnRole::Key;
                    columns.back().DataSize = sizes[rng() % 4];
                    columns.back().SizeType = EColumnSizeType::Fixed;
                }

                for (ui64 i = 0; i < payload_cols_count; ++i) {
                    columns.emplace_back();
                    columns.back().Role = EColumnRole::Payload;
                    columns.back().DataSize = sizes[rng() % 4];
                    columns.back().SizeType = EColumnSizeType::Fixed;
                }

                Benchmarker bench(columns, converter_type);
                bench.BenchPack(rows_count);
            }
        }
    }

    UNIT_ASSERT(true);

} // Y_UNIT_TEST(BenchPackFixed)
#endif

Y_UNIT_TEST(BenchPackVariable) {
    TScopedAlloc alloc(__LOCATION__);
    double key_cols_mult = 0.10; // 10% of cols are keys

    for (auto converter_type: {Benchmarker::ConverterType::Fallback, Benchmarker::ConverterType::SSE42, Benchmarker::ConverterType::AVX2}) {
        for (ui64 seq_byte_size: {8, 32, 128}) {
            Cerr << "--------------------------------------------------" << Endl;
            PrintType(converter_type);
            Cerr << "Variable sequence size: " << seq_byte_size << Endl;
            for (ui64 cols_count: {8, 32, 128}) {
                for (ui64 rows_count: {10'000, 100'000, 1'000'000}) {
                    if (cols_count * rows_count * seq_byte_size > 1e8) {
                        continue; // too large dataset
                    }
                    ui64 key_cols_count = std::max<ui64>(static_cast<ui64>(key_cols_mult * cols_count), 1);
                    ui64 payload_cols_count = cols_count - key_cols_count;
                    std::vector<TColumnDesc> columns;

                    for (ui64 i = 0; i < key_cols_count; ++i) {
                        columns.emplace_back();
                        columns.back().Role = EColumnRole::Key;
                        columns.back().DataSize = seq_byte_size;
                        columns.back().SizeType = EColumnSizeType::Variable;
                    }

                    for (ui64 i = 0; i < payload_cols_count; ++i) {
                        columns.emplace_back();
                        columns.back().Role = EColumnRole::Payload;
                        columns.back().DataSize = seq_byte_size;
                        columns.back().SizeType = EColumnSizeType::Variable;
                    }

                    Benchmarker bench(columns, converter_type);
                    bench.BenchPack(rows_count);
                }
            }
        }
    }

    UNIT_ASSERT(true);

} // Y_UNIT_TEST(BenchPackVariable)

#if 0 // Unpack not implemented yet
Y_UNIT_TEST(BenchUnpack) {

    UNIT_ASSERT(true);
} // Y_UNIT_TEST(BenchUnpack)
#endif

} // Y_UNIT_TEST_SUITE(BenchConversion)


} // namespace NPackedTuple
} // namespace NMiniKQL
} // namespace NKikimr

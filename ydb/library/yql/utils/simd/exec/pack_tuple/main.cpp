#include <random>
#include <typeinfo>

#include <util/generic/ptr.h>
#include <util/system/cpu_id.h>
#include <util/system/types.h>

#include <ydb/library/yql/utils/simd/simd.h>

struct TPerfomancer {
    TPerfomancer() = default;

    struct TWrapWorker {
        virtual int PackTuple(bool log) = 0;
        virtual void CmpPackTupleOrAndFallback() = 0;
        virtual ~TWrapWorker() = default;
    };

    template <typename TTraits> struct TWorker : TWrapWorker {
        template <typename T>
        using TSimd = typename TTraits::template TSimd8<T>;
        TWorker() = default;

        ui8 *ShuffleMask(ui32 v[8]) {
            ui8 *det = new ui8[32];
            for (size_t i = 0; i < 32; i += 1) {
                det[i] = v[i / 4] == ui32(-1) ? ui8(-1) : 4 * v[i / 4] + i % 4;
            }
            return det;
        }

        int PackTupleImpl(bool log = true) {
            if (TTraits::Size != 32)
                return 1;
            const ui64 NTuples = 32 << 18;
            const ui64 TupleSize = sizeof(ui32) + sizeof(ui64);

            ui32 *arrUi32 __attribute__((aligned(32))) = new ui32[NTuples];
            ui64 *arrUi64 __attribute__((aligned(32))) = new ui64[NTuples];

            for (ui32 i = 0; i < NTuples; i++) {
                arrUi32[i] = 2 * i;
            }

            for (ui32 i = 0; i < NTuples; i++) {
                arrUi64[i] = 2 * i + 1;
            }

            TSimd<ui8> readReg1, readReg2, readReg1Fwd;

            TSimd<ui8> permReg11, permReg21;
            TSimd<ui8> permReg12, permReg22;

            TSimd<ui8> permIdx11(
                ShuffleMask((ui32[8]){0, 0, 0, 1, 0, 0, 0, 0}));
            TSimd<ui8> permIdx12(
                ShuffleMask((ui32[8]){2, 0, 0, 3, 0, 0, 0, 0}));
            TSimd<ui8> permIdx1f(
                ShuffleMask((ui32[8]){4, 5, 6, 7, 7, 7, 7, 7}));

            TSimd<ui8> permIdx21(
                ShuffleMask((ui32[8]){0, 0, 1, 0, 2, 3, 0, 0}));
            TSimd<ui8> permIdx22(
                ShuffleMask((ui32[8]){0, 4, 5, 0, 6, 7, 0, 0}));

            ui32 val1[8], val2[8]; // val3[8];

            using TReg = typename TTraits::TRegister;
            TSimd<ui8> blended1, blended2;

            TReg *addr1 = (TReg *)arrUi32;
            TReg *addr2 = (TReg *)arrUi64;

            std::chrono::steady_clock::time_point begin01 =
                std::chrono::steady_clock::now();

            ui64 accum1 = 0;
            ui64 accum2 = 0;
            ui64 accum3 = 0;
            ui64 accum4 = 0;

            const int blendMask = 0b00110110;

            ui32 hash1 = 0;
            ui32 hash2 = 0;
            ui32 hash3 = 0;
            ui32 hash4 = 0;

            for (ui32 i = 0; i < NTuples; i += 8) {
                readReg1 = TSimd<ui8>((ui8 *)addr1);
                for (ui32 j = 0; j < 2; j++) {

                    permReg11 = readReg1.Shuffle(permIdx11);
                    readReg2 = TSimd<ui8>((ui8 *)addr2);
                    addr2++;
                    permReg21 = readReg2.Shuffle(permIdx21);
                    blended1 = permReg11.template Blend32<blendMask>(permReg21);
                    blended1.Store((ui8 *)val1);

                    hash1 = TSimd<ui8>::CRC32u32(0, val1[0]);
                    hash2 = TSimd<ui8>::CRC32u32(0, val1[3]);

                    accum1 += hash1;
                    accum2 += hash2;

                    permReg12 = readReg1.Shuffle(permIdx12);
                    permReg22 = readReg2.Shuffle(permIdx22);
                    blended2 = permReg12.template Blend32<blendMask>(permReg22);
                    blended2.Store((ui8 *)val2);

                    hash3 = TSimd<ui8>::CRC32u32(0, val2[0]);
                    hash4 = TSimd<ui8>::CRC32u32(0, val2[3]);

                    accum3 += hash3;
                    accum4 += hash4;

                    readReg1Fwd = readReg1.Shuffle(permIdx1f);
                    readReg1Fwd.Store((ui8 *)&readReg1.Value);
                }
                addr1++;
            }

            std::chrono::steady_clock::time_point end01 =
                std::chrono::steady_clock::now();

            Cerr << "Loaded col1 ";
            readReg1.template Log<ui32>(Cerr);
            Cerr << "Loaded col2 ";
            readReg2.template Log<ui32>(Cerr);
            ;
            Cerr << "Permuted col1 ";
            permReg11.template Log<ui32>(Cerr);
            ;
            Cerr << "Permuted col2 ";
            permReg21.template Log<ui32>(Cerr);
            Cerr << "Blended ";
            blended1.template Log<ui32>(Cerr);

            ui64 microseconds =
                std::chrono::duration_cast<std::chrono::microseconds>(end01 -
                                                                      begin01)
                    .count();
            if (log) {
                Cerr << "Accum 1 2 hash: " << accum1 << " " << accum2 << " "
                     << accum3 << " " << accum4 << " " << hash1 << " " << hash2
                     << " " << hash3 << " " << hash4 << Endl;
                Cerr << "Time for stream load = " << microseconds
                     << "[microseconds]" << Endl;
                Cerr << "Data size =  "
                     << ((NTuples * TupleSize) / (1024 * 1024)) << " [MB]"
                     << Endl;
                Cerr << "Stream load/save/accum speed = "
                     << (NTuples * TupleSize * 1000 * 1000) /
                            (1024 * 1024 * (microseconds + 1))
                     << " MB/sec" << Endl;
                Cerr << Endl;
            }
            delete[] arrUi32;
            delete[] arrUi64;

            return 1;
        }

        int PackTuple(bool log = true) override { return PackTupleImpl(log); }

        template <ui8 Cols>
        static void PackTupleFallbackRowImpl(const ui8 *const (&src_cols)[Cols],
                                             ui8 *const dst_rows, size_t size,
                                             const ui8 (&col_sizes)[Cols],
                                             const size_t padding) {
            size_t tuple_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_size;
                tuple_size += col_sizes[col];
            }
            tuple_size += padding;

            for (size_t row = 0; row != size; ++row) {
                for (ui8 col = 0; col != Cols; ++col) {
                    switch (col_sizes[col] * 8) {

#define MULTY_8x4(...)                                                         \
    __VA_ARGS__(8);                                                            \
    __VA_ARGS__(16);                                                           \
    __VA_ARGS__(32);                                                           \
    __VA_ARGS__(64)

#define CASE(bits)                                                             \
    case bits:                                                                 \
        *reinterpret_cast<ui##bits *>(dst_rows + row * tuple_size +            \
                                      offsets[col]) =                          \
            *reinterpret_cast<const ui##bits *>(src_cols[col] +                \
                                                row * (bits / 8));             \
        break

                        MULTY_8x4(CASE);

#undef CASE
#undef MULTY_8x4

                    default:
                        memcpy(dst_rows + row * tuple_size + offsets[col],
                               src_cols[col] + row * col_sizes[col],
                               col_sizes[col]);
                    }
                }
            }
        }

        template <ui8 Cols>
        static void UnpackTupleFallbackRowImpl(
            const ui8 *const src_rows, ui8 *const (&dst_cols)[Cols],
            size_t size, const ui8 (&col_sizes)[Cols], const size_t padding) {
            size_t tuple_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_size;
                tuple_size += col_sizes[col];
            }
            tuple_size += padding;

            for (size_t row = 0; row != size; ++row) {
                for (ui8 col = 0; col != Cols; ++col) {
                    switch (col_sizes[col] * 8) {

#define MULTY_8x4(...)                                                         \
    __VA_ARGS__(8);                                                            \
    __VA_ARGS__(16);                                                           \
    __VA_ARGS__(32);                                                           \
    __VA_ARGS__(64)

#define CASE(bits)                                                             \
    case bits:                                                                 \
        *reinterpret_cast<ui##bits *>(dst_cols[col] + row * (bits / 8)) =      \
            *reinterpret_cast<const ui##bits *>(src_rows + row * tuple_size +  \
                                                offsets[col]);                 \
        break

                        MULTY_8x4(CASE);

#undef CASE
#undef MULTY_8x4

                    default:
                        memcpy(dst_cols[col] + row * col_sizes[col],
                               src_rows + row * tuple_size + offsets[col],
                               col_sizes[col]);
                    }
                }
            }
        }

        template <class ByteType>
        static void PackTupleFallbackTypedColImpl(const ui8 *const src_col,
                                                  ui8 *const dst_rows,
                                                  const size_t size,
                                                  const size_t tuple_size) {
            static constexpr size_t BYTES = sizeof(ByteType);
            for (size_t row = 0; row != size; ++row) {
                *reinterpret_cast<ByteType *>(dst_rows + row * tuple_size) =
                    *reinterpret_cast<const ByteType *>(src_col + row * BYTES);
            }
        }

        template <class ByteType>
        static void UnpackTupleFallbackTypedColImpl(const ui8 *const src_rows,
                                                  ui8 *const dst_col,
                                                  const size_t size,
                                                  const size_t tuple_size) {
            static constexpr size_t BYTES = sizeof(ByteType);
            for (size_t row = 0; row != size; ++row) {
                    *reinterpret_cast<ByteType *>(dst_col + row * BYTES) =
                *reinterpret_cast<const ByteType *>(src_rows + row * tuple_size);
            }
        }

        template <ui8 Cols>
        static void PackTupleFallbackColImpl(const ui8 *const (&src_cols)[Cols],
                                             ui8 *const dst_rows, size_t size,
                                             const ui8 (&col_sizes)[Cols],
                                             const size_t padding) {
            size_t tuple_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_size;
                tuple_size += col_sizes[col];
            }
            tuple_size += padding;

            for (ui8 col = 0; col != Cols; ++col) {
                switch (col_sizes[col] * 8) {

#define MULTY_8x4(...)                                                         \
    __VA_ARGS__(8);                                                            \
    __VA_ARGS__(16);                                                           \
    __VA_ARGS__(32);                                                           \
    __VA_ARGS__(64)

#define CASE(bits)                                                             \
    case bits:                                                                 \
        PackTupleFallbackTypedColImpl<ui##bits>(                               \
            src_cols[col], dst_rows + offsets[col], size, tuple_size);         \
        break

                    MULTY_8x4(CASE);

#undef CASE
#undef MULTY_8x4

                default:
                    for (size_t row = 0; row != size; ++row) {
                        memcpy(dst_rows + row * tuple_size + offsets[col],
                               src_cols[col] + row * col_sizes[col],
                               col_sizes[col]);
                    }
                }
            }
        }

        template <ui8 Cols>
        static void UnpackTupleFallbackColImpl(const ui8 *const src_rows,
                                               ui8 *const (&dst_cols)[Cols],
                                               size_t size,
                                               const ui8 (&col_sizes)[Cols],
                                               const size_t padding) {
            size_t tuple_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_size;
                tuple_size += col_sizes[col];
            }
            tuple_size += padding;

            for (ui8 col = 0; col != Cols; ++col) {
                switch (col_sizes[col] * 8) {

#define MULTY_8x4(...)                                                         \
    __VA_ARGS__(8);                                                            \
    __VA_ARGS__(16);                                                           \
    __VA_ARGS__(32);                                                           \
    __VA_ARGS__(64)

#define CASE(bits)                                                             \
    case bits:                                                                 \
        UnpackTupleFallbackTypedColImpl<ui##bits>(                               \
            src_rows + offsets[col], dst_cols[col], size, tuple_size);         \
        break

                    MULTY_8x4(CASE);

#undef CASE
#undef MULTY_8x4

                default:
                    for (size_t row = 0; row != size; ++row) {
                        memcpy(dst_cols[col] + row * col_sizes[col],
                               src_rows + row * tuple_size + offsets[col],
                               col_sizes[col]);
                    }
                }
            }
        }

        template <ui8 Cols, ui8 Block = 20>
        static void PackTupleFallbackBlockImpl(
            const ui8 *const (&src_cols)[Cols], ui8 *const dst_rows,
            size_t size, const ui8 (&col_sizes)[Cols], const size_t padding) {
            size_t tuple_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_size;
                tuple_size += col_sizes[col];
            }
            tuple_size += padding;

            const size_t block_size = size / Block;
            for (size_t block = 0; block != block_size; ++block) {
                for (ui8 col = 0; col != Cols; ++col) {
                    switch (col_sizes[col] * 8) {

#define BLOCK_LOOP(...)                                                        \
    for (size_t block_i = 0; block_i != Block; ++block_i) {                    \
        const size_t row = Block * block + block_i;                            \
        __VA_ARGS__                                                            \
    }

#define MULTY_8x4(...)                                                         \
    __VA_ARGS__(8);                                                            \
    __VA_ARGS__(16);                                                           \
    __VA_ARGS__(32);                                                           \
    __VA_ARGS__(64)

#define CASE(bits)                                                             \
    case bits:                                                                 \
        BLOCK_LOOP(*reinterpret_cast<ui##bits *>(dst_rows + row * tuple_size + \
                                                 offsets[col]) =               \
                       *reinterpret_cast<const ui##bits *>(src_cols[col] +     \
                                                           row * (bits / 8));) \
        break

                        MULTY_8x4(CASE);

#undef CASE
#undef MULTY_8x4

                    default:
                        BLOCK_LOOP(
                            memcpy(dst_rows + row * tuple_size + offsets[col],
                                   src_cols[col] + row * col_sizes[col],
                                   col_sizes[col]);)
                    }
                }
            }

            for (ui8 col = 0; col != Cols; ++col) {
                [&]<size_t... Is>(std::index_sequence<Is...>) {
                    PackTupleFallbackColImpl(
                        {src_cols[Is] + block_size * Block * col_sizes[Is]...},
                        dst_rows + block_size * Block * tuple_size,
                        size - block_size * Block, col_sizes, padding);
                }(std::make_index_sequence<Cols>{});
            }
        }

        template <ui8 Cols, ui8 Block = 20>
        static void UnpackTupleFallbackBlockImpl(
            const ui8 *const src_rows, ui8 *const (&dst_cols)[Cols], 
            size_t size, const ui8 (&col_sizes)[Cols], const size_t padding) {
            size_t tuple_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_size;
                tuple_size += col_sizes[col];
            }
            tuple_size += padding;

            const size_t block_size = size / Block;
            for (size_t block = 0; block != block_size; ++block) {
                for (ui8 col = 0; col != Cols; ++col) {
                    switch (col_sizes[col] * 8) {

#define BLOCK_LOOP(...)                                                        \
    for (size_t block_i = 0; block_i != Block; ++block_i) {                    \
        const size_t row = Block * block + block_i;                            \
        __VA_ARGS__                                                            \
    }

#define MULTY_8x4(...)                                                         \
    __VA_ARGS__(8);                                                            \
    __VA_ARGS__(16);                                                           \
    __VA_ARGS__(32);                                                           \
    __VA_ARGS__(64)

#define CASE(bits)                                                             \
    case bits:                                                                 \
        BLOCK_LOOP(                                                            \
            *reinterpret_cast<ui##bits *>(dst_cols[col] + row * (bits / 8)) =  \
                *reinterpret_cast<const ui##bits *>(                           \
                    src_rows + row * tuple_size + offsets[col]);)              \
        break

                        MULTY_8x4(CASE);

#undef CASE
#undef MULTY_8x4

                    default:
                        BLOCK_LOOP(
                            memcpy(dst_cols[col] + row * col_sizes[col],
                                   src_rows + row * tuple_size + offsets[col],
                                   col_sizes[col]);)
                    }
                }
            }

            for (ui8 col = 0; col != Cols; ++col) {
                [&]<size_t... Is>(std::index_sequence<Is...>) {
                    UnpackTupleFallbackColImpl(

                        src_rows + block_size * Block * tuple_size,
                        {dst_cols[Is] + block_size * Block * col_sizes[Is]...},
                        size - block_size * Block, col_sizes, padding);
                }(std::make_index_sequence<Cols>{});
            }
        }


        /// [8,16,32,64]-bits iters
        static const ui8 BaseIters = 4;

        /// 128-bit lane iters
        static const ui8 LaneIters = [] {
            if constexpr (std::is_same_v<TTraits, NSimd::TSimdAVX2Traits>) {
                return 1;
            }
            return 0;
        }();

        static const ui8 TransposeIters = BaseIters + LaneIters;

        /// bits reversed
        static constexpr ui8 TransposeRevInd[BaseIters][1 << BaseIters] = {
            {
                0x0,
                0x8,
                0x4,
                0xc,
                0x2,
                0xa,
                0x6,
                0xe,
                0x1,
                0x9,
                0x5,
                0xd,
                0x3,
                0xb,
                0x7,
                0xf,
            },
            {
                0x0,
                0x4,
                0x2,
                0x6,
                0x1,
                0x5,
                0x3,
                0x7,
            },
            {
                0x0,
                0x2,
                0x1,
                0x3,
            },
            {
                0x0,
                0x1,
            },
        };

        template <ui8 Cols, ui8 LogIter>
        static void Transpose(TSimd<ui8> (&regs)[2][Cols]) {
            /// iterative transposition, starting from ColSize:
            ///     ui8 -> ui16 -> ui32 -> ui64 -> 128bit-lane
            /// smth like fourier butterfly

    #define TRANSPOSE_ITER(iter, bits)                                             \
        if constexpr (LogIter <= iter) {                                           \
            constexpr bool from = iter % 2;                                        \
            constexpr bool to = from ^ 1;                                          \
                                                                                \
            constexpr ui8 log = iter - LogIter;                                    \
            constexpr ui8 exp = 1u << log;                                         \
                                                                                \
            for (ui8 col = 0; col != Cols; ++col) {                                \
                switch ((col & exp) >> log) {                                      \
                case 0: {                                                          \
                    regs[to][col] = TSimd<ui8>::UnpackLaneLo##bits(                \
                        regs[from][col & ~exp], regs[from][col | exp]);            \
                    break;                                                         \
                }                                                                  \
                case 1: {                                                          \
                    regs[to][col] = TSimd<ui8>::UnpackLaneHi##bits(                \
                        regs[from][col & ~exp], regs[from][col | exp]);            \
                    break;                                                         \
                }                                                                  \
                default:;                                                          \
                }                                                                  \
            }                                                                      \
        }

            TRANSPOSE_ITER(0, 8);
            TRANSPOSE_ITER(1, 16);
            TRANSPOSE_ITER(2, 32);
            TRANSPOSE_ITER(3, 64);
        
    #undef TRANSPOSE_ITER

            if constexpr (LaneIters == 1) {
                constexpr auto iter = BaseIters;

                constexpr bool from = iter % 2;
                constexpr bool to = from ^ 1;

                constexpr ui8 log = iter - LogIter;
                constexpr ui8 exp = 1u << log;

                for (ui8 col = 0; col != Cols; ++col) {
                    switch ((col & exp) >> log) {
                    case 0: {
                        regs[to][col] =
                            TSimd<ui8>::template PermuteLanes<0 + 2 * 16>(
                                regs[from][col & ~exp], regs[from][col | exp]);
                        break;
                    }
                    case 1: {
                        regs[to][col] =
                            TSimd<ui8>::template PermuteLanes<1 + 3 * 16>(
                                regs[from][col & ~exp], regs[from][col | exp]);
                        break;
                    }
                    }
                }
            } else if constexpr (LaneIters) {
                static_assert(!LaneIters, "Not implemented");
            }
        }

        template <ui8 ColSize>
        static void PackColSizeImpl(const ui8 *const src_cols[],
                                    ui8 *const dst_rows, const size_t size,
                                    const size_t cols_num, const size_t padded_size,
                                    const size_t start = 0) {
            static constexpr ui8 Cols = TSimd<ui8>::SIZE / ColSize;
            static constexpr ui8 LogIter = std::countr_zero(ColSize);
            static constexpr std::array<size_t, Cols> ColSizes = [] {
                std::array<size_t, Cols> offsets;
                for (size_t ind = 0; ind != Cols; ++ind) {
                    offsets[ind] = ColSize;
                }
                return offsets;
            }();

            const size_t simd_iters = size / Cols;

            TSimd<ui8> regs[2][Cols];

            for (size_t cols_group = 0; cols_group < cols_num; cols_group += Cols) {
                const ui8 *srcs[Cols];
                std::memcpy(srcs, src_cols + cols_group, sizeof(srcs));
                for (ui8 col = 0; col != Cols; ++col) {
                    srcs[col] += ColSize * start;
                }

                auto dst = dst_rows + cols_group * ColSize;
                ui8 *const end = dst + simd_iters * Cols * padded_size;

                while (dst != end) {
                    for (ui8 col = 0; col != Cols; ++col) {
                        regs[LogIter % 2][col] = TSimd<ui8>(srcs[col]);
                        srcs[col] += TSimd<ui8>::SIZE;
                    }

                    Transpose<Cols, LogIter>(regs);

                    const bool res = TransposeIters % 2;
                    for (ui8 col = 0; col != Cols; ++col) {
                        if constexpr (LaneIters) {
                            const ui8 half_ind = col % (Cols / 2);
                            const ui8 half_shift = col & (Cols / 2);
                            regs[res]
                                [TransposeRevInd[LogIter][half_ind] + half_shift]
                                    .Store(dst);
                            dst += padded_size;
                        } else {
                            regs[res][TransposeRevInd[LogIter][col]].Store(dst);
                            dst += padded_size;
                        }
                    }
                }

                PackTupleFallbackRowImpl<Cols>(srcs, dst, size - simd_iters * Cols,
                    *(ui8(*)[Cols])(&ColSizes), padded_size - Cols * ColSize);
            }
        }

        template <ui8 ColSize>
        static void
        UnpackColSizeImpl(const ui8 *const src_rows, ui8 *const dst_cols[],
                        size_t size, const size_t cols_num,
                        const size_t padded_size, const size_t start = 0) {
            static constexpr ui8 Cols = TSimd<ui8>::SIZE / ColSize;
            static constexpr ui8 LogIter = std::countr_zero(ColSize);
            static constexpr std::array<size_t, Cols> ColSizes = [] {
                std::array<size_t, Cols> offsets;
                for (size_t ind = 0; ind != Cols; ++ind) {
                    offsets[ind] = ColSize;
                }
                return offsets;
            }();

            const size_t simd_iters = size / Cols;

            TSimd<ui8> regs[2][Cols];

            for (size_t cols_group = 0; cols_group < cols_num; cols_group += Cols) {
                auto src = src_rows + cols_group * ColSize;
                const ui8 *const end = src + simd_iters * Cols * padded_size;

                ui8 *dsts[Cols];
                std::memcpy(dsts, dst_cols + cols_group, sizeof(dsts));
                for (ui8 col = 0; col != Cols; ++col) {
                    dsts[col] += ColSize * start;
                }

                while (src != end) {
                    for (ui8 iter = 0; iter != Cols; ++iter) {
                        regs[LogIter % 2][iter] = TSimd<ui8>(src);
                        src += padded_size;
                    }

                    Transpose<Cols, LogIter>(regs);

                    const bool res = TransposeIters % 2;
                    for (ui8 col = 0; col != Cols; ++col) {
                        if constexpr (LaneIters) {
                            const ui8 half_ind = col % (Cols / 2);
                            const ui8 half_shift = col & (Cols / 2);
                            regs[res]
                                [TransposeRevInd[LogIter][half_ind] + half_shift]
                                    .Store(dsts[col]);
                            dsts[col] += TSimd<ui8>::SIZE;
                        } else {
                            regs[res][TransposeRevInd[LogIter][col]].Store(
                                dsts[col]);
                            dsts[col] += TSimd<ui8>::SIZE;
                        }
                    }
                }

                UnpackTupleFallbackRowImpl<Cols>(
                    src, dsts, size - simd_iters * Cols, *(ui8(*)[Cols])(&ColSizes),
                    padded_size - Cols * ColSize);
            }
        }

        template <size_t Cols>
        static void PackColSize(const ui8 *const (&src_cols)[Cols],
                                ui8 *const dst_rows_, size_t size,
                                const ui8 (&col_sizes)[Cols],
                                const size_t padding) {
            size_t padded_size = 0;
            for (ui8 col = 0; col != Cols; ++col) {
                padded_size += col_sizes[col];
            }
            padded_size += padding;

            size_t cols_num;
            const size_t block_size = 256;
            for (size_t row = 0; row < size; row += block_size) {
                auto dst_rows = dst_rows_ + padded_size * row;

                size_t pad = 0;
                for (ui8 col = 0; col != Cols;) {

                    switch (col_sizes[col]) {
                    case 1:
                        cols_num = TSimd<ui8>::SIZE / 1;
                        PackColSizeImpl<1>(&src_cols[col], dst_rows + pad, block_size, cols_num, padded_size, row);
                        col += cols_num;
                        pad += cols_num * 1;
                        break;
                    case 2:
                        cols_num = TSimd<ui8>::SIZE / 2;
                        PackColSizeImpl<2>(&src_cols[col], dst_rows + pad, block_size, cols_num, padded_size, row);
                        col += cols_num;
                        pad += cols_num * 2;
                        break;
                    case 4:
                        cols_num = TSimd<ui8>::SIZE / 4;
                        PackColSizeImpl<4>(&src_cols[col], dst_rows + pad, block_size, cols_num, padded_size, row);
                        col += cols_num;
                        pad += cols_num * 4;
                        break;
                    case 8:
                        cols_num = TSimd<ui8>::SIZE / 8;
                        PackColSizeImpl<8>(&src_cols[col], dst_rows + pad, block_size, cols_num, padded_size, row);
                        col += cols_num;
                        pad += cols_num * 8;
                        break;
                    default:
                        throw std::runtime_error("What? Unexpected pack switch case " +
                                                std::to_string(col_sizes[col]));
                    }
                }
            }
        }

        template <ui8 Cols>
        static void UnpackColSize(const ui8 *const src_rows_, ui8 *const (&dst_cols)[Cols],
                                  size_t size, const ui8 (&col_sizes)[Cols], const size_t padding) {
            size_t padded_size = 0;
            for (ui8 col = 0; col != Cols; ++col) {
                padded_size += col_sizes[col];
            }
            padded_size += padding;

            size_t cols_num;
            const size_t block_size = 256;
            for (size_t row = 0; row < size; row += block_size) {
                auto src_rows = src_rows_ + padded_size * row;

                size_t pad = 0;

                for (ui8 col = 0; col != Cols;) {
                    switch (col_sizes[col]) {
                    case 1:
                        cols_num = TSimd<ui8>::SIZE / 1;
                        UnpackColSizeImpl<1>(src_rows + pad, &dst_cols[col], block_size, cols_num, padded_size, row);
                        col += cols_num;
                        break;
                    case 2:
                        cols_num = TSimd<ui8>::SIZE / 2;
                        UnpackColSizeImpl<2>(src_rows + pad, &dst_cols[col], block_size, cols_num, padded_size, row);
                        col += cols_num;
                        pad += cols_num * 2;
                        break;
                    case 4:
                        cols_num = TSimd<ui8>::SIZE / 4;
                        UnpackColSizeImpl<4>(src_rows + pad, &dst_cols[col], block_size, cols_num, padded_size, row);
                        col += cols_num;
                        pad += cols_num * 4;
                        break;
                    case 8:
                        cols_num = TSimd<ui8>::SIZE / 8;
                        UnpackColSizeImpl<8>(src_rows + pad, &dst_cols[col], block_size, cols_num, padded_size, row);
                        col += cols_num;
                        pad += cols_num * 8;
                        break;
                    default:
                        throw std::runtime_error("What? Unexpected pack switch case " +
                                                std::to_string(col_sizes[col]));
                    }
                }
            }
        }

        template <size_t Padding, class... Types> struct PackBenchCase {
            static constexpr size_t Iters = 2;
            static constexpr size_t Memory = 1ul << 30;
            static constexpr size_t TupleSize = (sizeof(Types) + ...) + Padding;
            static constexpr size_t Cols = sizeof...(Types);
            static constexpr size_t NTuples = Memory / TupleSize;

            static constexpr size_t StoresPerLoad =
                // std::min(Cols, 
                (TSimd<ui8>::SIZE / std::max({sizeof(Types)...})) /
                                   std::max(1ul, TSimd<ui8>::SIZE / TupleSize);
            //    );
            using PackTupleFs =
                std::pair<void (*)(const ui8 *const (&)[Cols], ui8 *const,
                                   size_t, const ui8 (&)[Cols], const size_t),
                          void (*)(const ui8 *const, ui8 *const (&)[Cols],
                                   size_t, const ui8 (&)[Cols], const size_t)>;

            static constexpr std::pair<PackTupleFs, const char *> Runs[] = {
                {{&PackTupleFallbackRowImpl<Cols>,
                  &UnpackTupleFallbackRowImpl<Cols>},
                 "fallback (row)"},
                {{&PackTupleFallbackColImpl<Cols>,
                  &UnpackTupleFallbackColImpl<Cols>},
                 "fallback (col)"},
                {{&PackTupleFallbackBlockImpl<Cols>,
                  &UnpackTupleFallbackBlockImpl<Cols>},
                 "fallback (block)"},
                {{&PackColSize<Cols>,
                  &UnpackColSize<Cols>},
                 "simd (typed cols)"},
            };

            static void Run() {
#define INDEXED_TYPES(...)                                                     \
    [&]<size_t... Is>(std::index_sequence<Is...>) {                            \
        __VA_ARGS__                                                            \
    }(std::make_index_sequence<Cols>{})

#define INDEXED_TYPES_B [&]<size_t... Is>(std::index_sequence<Is...>) {

#define INDEXED_TYPES_E                                                        \
    }                                                                          \
    (std::make_index_sequence<Cols>{})

                Cerr << "\n - cols byte alignment";
                (..., (Cerr << ' ' << sizeof(Types)));
                Cerr << '\n';

                const ui8 col_sizes[Cols] = {sizeof(Types)...};
                size_t offsets[Cols];
                for (ui8 col = 0, offset = 0; col != Cols; ++col) {
                    offsets[col] = offset;
                    offset += col_sizes[col];
                }

                std::unique_ptr<ui8[]> srcs[sizeof...(Types)];
                ui8 *src_cols[Cols];

                INDEXED_TYPES((srcs[Is].reset(new (std::align_val_t(64))
                                                  ui8[NTuples * sizeof(Types)]),
                               ...);
                              ((src_cols[Is] = srcs[Is].get()), ...););

                std::unique_ptr<ui8[]> dst{new (std::align_val_t(64))
                                               ui8[NTuples * TupleSize]};
                ui8 *dst_rows = dst.get();

                for (const auto &[funcs, name] : Runs) {
                    Cerr << "\nrunning: " << name << '\n';
                    for (ui8 iters = 0; iters != Iters; ++iters) {

                        for (size_t ind = 0; ind < NTuples; ind++) {
                            INDEXED_TYPES(((reinterpret_cast<Types *>(
                                                srcs[Is].get())[ind] = Cols * ind + Is),
                                        ...););
                        }

                        dst[0] = 1;
                        for (size_t ind = 1; ind != NTuples * TupleSize;
                             ++ind) {
                            dst[ind] = dst[ind - 1] * 127 + 1;
                        }

                        std::chrono::steady_clock::time_point begin_ts =
                            std::chrono::steady_clock::now();

                        std::invoke(funcs.first, src_cols, dst.get(), NTuples,
                                    col_sizes, Padding);

                        std::chrono::steady_clock::time_point end_ts =
                            std::chrono::steady_clock::now();
                        ui64 us =
                            std::chrono::duration_cast<
                                std::chrono::microseconds>(end_ts - begin_ts)
                                .count();
                        size_t mbs =
                            (NTuples * (TupleSize - Padding) * 1000ul * 1000ul) /
                            (us * 1024ul * 1024ul + 1);

                        size_t fails = 0;
                        for (size_t row = 0; row < NTuples; row += 1) {
                            bool flag = true;
                            INDEXED_TYPES(flag =
                                              (flag && ... &&
                                               (reinterpret_cast<Types *>(
                                                    dst_rows + row * TupleSize +
                                                    offsets[Is])[0] ==
                                                Types(Cols * row + Is))););
                            if (flag) {
                                continue;
                            }
                            ++fails;

                            // INDEXED_TYPES(
                            //     (..., (Cerr << row << ": "
                            //                 << size_t(reinterpret_cast<Types *>(
                            //                        dst_rows + row * TupleSize +
                            //                        offsets[Is])[0])
                            //                 << " =?= " << (Cols * row + Is)
                            //                 << '\n'));

                            // );
                            // break;
                        }

                        Cerr << "> " << mbs << " mib/s, fails: " << fails << '\n';

                        begin_ts = std::chrono::steady_clock::now();

                        std::invoke(funcs.second, dst.get(), src_cols, NTuples,
                                    col_sizes, Padding);

                        end_ts = std::chrono::steady_clock::now();
                        us = std::chrono::duration_cast<
                                 std::chrono::microseconds>(end_ts - begin_ts)
                                 .count();
                        mbs = (NTuples * (TupleSize - Padding) * 1000ul *
                               1000ul) /
                              (us * 1024ul * 1024ul + 1);

                        fails = 0;
                        for (size_t row = 0; row < NTuples; row += 1) {
                            bool flag = true;
                            INDEXED_TYPES(flag =
                                              (flag && ... &&
                                               (reinterpret_cast<Types *>(
                                                    dst_rows + row * TupleSize +
                                                    offsets[Is])[0] ==
                                                Types(Cols * row + Is))););
                            if (flag) {
                                continue;
                            }
                            ++fails;

                            // INDEXED_TYPES(
                            //     (..., (Cerr << row << ": "
                            //                 << size_t(reinterpret_cast<Types *>(
                            //                        dst_rows + row * TupleSize +
                            //                        offsets[Is])[0])
                            //                 << " =?= " << (Cols * row + Is)
                            //                 << '\n'));

                            // );
                            // break;
                        }

                        Cerr << "< " << mbs << " mib/s, fails: " << fails << '\n';
                    }
                }

#undef INDEXED_TYPES
            }
        };

        void CmpPackTupleOrAndFallback() override {
            Cerr << " --- Compare simd-or and fallback copy --- \n";

            static constexpr size_t Padding = 128;

            PackBenchCase<0, ui64, ui64, ui64, ui64>::Run();
            PackBenchCase<0, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32>::Run();
            PackBenchCase<0, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16>::Run();

            PackBenchCase<Padding, ui64, ui64, ui64, ui64>::Run();
            PackBenchCase<Padding, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32>::Run();
            
            PackBenchCase<0
                , ui64, ui64, ui64, ui64
                , ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32
                , ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16
                , ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32
                , ui64, ui64, ui64, ui64
            >::Run();
        }

        ~TWorker() = default;
    };

    template <typename TTraits> THolder<TWrapWorker> Create() const {
        return MakeHolder<TWorker<TTraits>>();
    };

    struct BinaryStats {
        static constexpr size_t SAMPLES = 50;

        size_t mean;
        size_t std;
    };

    template <class TOffset>
    static BinaryStats CollectBinaryStats(const TOffset *const src_offset,
                                          const size_t length) {
        std::mt19937 gen;
        BinaryStats result;
        result.mean = (src_offset[length] - src_offset[0]) / length;

        uint64_t sample_sqr_sum = 0;
        if (length <= BinaryStats::SAMPLES) {
            const auto *end = src_offset + length;
            for (auto *p = src_offset; p != end; ++p) {
                const size_t len = *(p + 1) - *(p);
                const uint64_t sample_val =
                    len < result.mean ? result.mean - len : len - result.mean;
                sample_sqr_sum += sample_val * sample_val;
            }
            sample_sqr_sum = (sample_sqr_sum + length - 1) / length;
        } else {
            struct {
                size_t ind;
                TOffset val;
            } visited[BinaryStats::SAMPLES];

            uint64_t visited_hash = 0;
            size_t visited_num = 0;

            for (size_t iter = 0; iter != BinaryStats::SAMPLES; ++iter) {
                const size_t last = length - iter - 1;
                const size_t ind =
                    std::uniform_int_distribution<size_t>{0, last}(gen);
                const uint64_t hash = 1ull << (ind % 64);

                TOffset sample_val = src_offset[ind + 1] - src_offset[ind];

                if (visited_hash & hash) {
                    const auto last_it =
                        std::find_if(visited, visited + visited_num,
                                     [&](auto el) { return el.ind == last; });
                    TOffset last_val;
                    if (last_it == visited + visited_num) {
                        last_val = src_offset[last + 1] - src_offset[last];
                    } else {
                        last_val = last_it->val;
                    }

                    const auto it =
                        std::find_if(visited, visited + visited_num,
                                     [&](auto el) { return el.ind == ind; });
                    if (it == visited + visited_num) {
                        it->ind = ind;
                        ++visited_num;
                    } else {
                        sample_val = it->val;
                    }
                    it->val = last_val;
                } else {
                    visited[visited_num].ind = ind;
                    visited[visited_num].val = sample_val;
                    ++visited_num;
                    visited_hash |= hash;
                }

                sample_val = sample_val < result.mean
                                 ? result.mean - sample_val
                                 : sample_val - result.mean;
                sample_sqr_sum +=
                    static_cast<uint64_t>(sample_val) * sample_val;
            }

            sample_sqr_sum = (sample_sqr_sum + BinaryStats::SAMPLES - 1) /
                             BinaryStats::SAMPLES;
        }
        result.std = std::ceil(std::sqrt(sample_sqr_sum));

        return result;
    }
};

int main() {
    if (!NX86::HaveAVX2())
        return 0;

    TPerfomancer tp;
    auto worker = tp.Create<NSimd::TSimdAVX2Traits>();

    bool fine = true;
    // fine &= worker->PackTuple(false);

    worker->CmpPackTupleOrAndFallback();

    return !fine;
}
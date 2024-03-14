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

    template<typename TTraits>
    struct TWorker : TWrapWorker {
        template<typename T>
        using TSimd = typename TTraits::template TSimd8<T>;
        TWorker() = default;

        ui8* ShuffleMask(ui32 v[8]) {
            ui8* det = new ui8[32];
            for (size_t i = 0; i < 32; i += 1) {
                det[i] = v[i / 4] == ui32(-1) ? ui8(-1) : 4 * v[i / 4] + i % 4;
            }
            return det;
        }

        int PackTupleImpl(bool log = true) {
            if (TTraits::Size != 32)
                return 1;
            const ui64 NTuples = 32 << 18;
            const ui64 TupleSize =  sizeof(ui32) + sizeof(ui64);

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

            TSimd<ui8> permIdx11(ShuffleMask((ui32[8]) {0, 0, 0, 1, 0, 0, 0, 0}));
            TSimd<ui8> permIdx12(ShuffleMask((ui32[8]) {2, 0, 0, 3, 0, 0, 0, 0}));
            TSimd<ui8> permIdx1f(ShuffleMask((ui32[8]) {4, 5, 6, 7, 7, 7, 7, 7}));

            TSimd<ui8> permIdx21(ShuffleMask((ui32[8]) {0, 0, 1, 0, 2, 3, 0, 0}));
            TSimd<ui8> permIdx22(ShuffleMask((ui32[8]) {0, 4, 5, 0, 6, 7, 0, 0}));

            ui32 val1[8], val2[8]; // val3[8];

            using TReg = typename TTraits::TRegister;
            TSimd<ui8> blended1, blended2;

            TReg *addr1 = (TReg*) arrUi32;
            TReg *addr2 = (TReg*) arrUi64;

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
                readReg1 = TSimd<ui8>((ui8*) addr1);
                for (ui32 j = 0; j < 2; j++) {

                    permReg11 = readReg1.Shuffle(permIdx11);
                    readReg2 = TSimd<ui8>((ui8*) addr2);
                    addr2++;
                    permReg21 = readReg2.Shuffle(permIdx21);
                    blended1 = permReg11.template Blend32<blendMask>(permReg21);
                    blended1.Store((ui8*) val1);

                    hash1 = TSimd<ui8>::CRC32u32(0, val1[0]);
                    hash2 = TSimd<ui8>::CRC32u32(0, val1[3]);

                    accum1 += hash1;
                    accum2 += hash2;

                    permReg12 = readReg1.Shuffle(permIdx12);
                    permReg22 = readReg2.Shuffle(permIdx22);
                    blended2 = permReg12.template Blend32<blendMask>(permReg22);
                    blended2.Store((ui8*) val2);

                    hash3 = TSimd<ui8>::CRC32u32(0, val2[0]);
                    hash4 = TSimd<ui8>::CRC32u32(0, val2[3]);

                    accum3 += hash3;
                    accum4 += hash4;

                    readReg1Fwd = readReg1.Shuffle(permIdx1f);
                    readReg1Fwd.Store((ui8*) &readReg1.Value);

                }
                addr1++;
            }


            std::chrono::steady_clock::time_point end01 =
                std::chrono::steady_clock::now();

            Cerr << "Loaded col1 ";
            readReg1.template Log<ui32>(Cerr);
            Cerr << "Loaded col2 ";
            readReg2.template Log<ui32>(Cerr);;
            Cerr << "Permuted col1 ";
            permReg11.template Log<ui32>(Cerr);;
            Cerr << "Permuted col2 ";
            permReg21.template Log<ui32>(Cerr);
            Cerr << "Blended ";
            blended1.template Log<ui32>(Cerr);

            ui64 microseconds =
                std::chrono::duration_cast<std::chrono::microseconds>(end01 - begin01).count();
            if (log) {
                Cerr << "Accum 1 2 hash: " << accum1 << " " << accum2 << " "  << accum3 << " " << accum4 << " "
                << hash1 << " " << hash2 << " " << hash3 << " " << hash4 << Endl;
                Cerr << "Time for stream load = " << microseconds << "[microseconds]"
                    << Endl;
                Cerr << "Data size =  " << ((NTuples * TupleSize) / (1024 * 1024))
                    << " [MB]" << Endl;
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

        int PackTuple(bool log = true) override {
            return PackTupleImpl(log);
        }

        template <ui8 Cols>
        static void PackTupleFallbackRowImpl(const ui8 *const (&src_cols)[Cols],
                                             ui8 *const dst_rows, size_t size,
                                             const ui8 (&col_sizes)[Cols]) {
            size_t tuple_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_size;
                tuple_size += col_sizes[col];
            }

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

        template <ui8 Cols>
        static void PackTupleFallbackColImpl(const ui8 *const (&src_cols)[Cols],
                                             ui8 *const dst_rows, size_t size,
                                             const ui8 (&col_sizes)[Cols]) {
            size_t tuple_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_size;
                tuple_size += col_sizes[col];
            }

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

        template <ui8 Cols, ui8 Block = 16>
        static void
        PackTupleFallbackBlockImpl(const ui8 *const (&src_cols)[Cols],
                                   ui8 *const dst_rows, size_t size,
                                   const ui8 (&col_sizes)[Cols]) {
            size_t tuple_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_size;
                tuple_size += col_sizes[col];
            }

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
                      size - block_size * Block, col_sizes);
                }(std::make_index_sequence<Cols>{});
            }
        }

        static TSimd<ui8> BuildTuplePerm(ui8 col_size, ui8 col_pad, ui8 offset,
                                         ui8 ind) {
            ui8 perm[TSimd<ui8>::SIZE];
            std::memset(perm, 0x80, TSimd<ui8>::SIZE);

            const ui8 size = col_size + col_pad;
            ui8 iters = TSimd<ui8>::SIZE / size;

            while (iters--) {
                for (ui8 it = col_size; it; --it, ++offset, ++ind) {
                    perm[offset] = ind;
                }
                offset += col_pad;
            }

            return TSimd<ui8>{perm};
        }

        template <ui8 TupleSize>
        static TSimd<ui8> TupleOr(TSimd<ui8> (&vec)[TupleSize]) {
            return TupleOrImpl<TupleSize>(vec);
        }

        template <ui8 TupleSize>
        static TSimd<ui8> TupleOrImpl(TSimd<ui8> vec[]) {
            static constexpr ui8 Left = TupleSize / 2;
            static constexpr ui8 Right = TupleSize - Left;

            return TupleOrImpl<Left>(vec) | TupleOrImpl<Right>(vec + Left);
        }

        template <> TSimd<ui8> TupleOrImpl<1>(TSimd<ui8> vec[]) {
            return vec[0];
        }

        template <> TSimd<ui8> TupleOrImpl<2>(TSimd<ui8> vec[]) {
            return vec[0] | vec[1];
        }

        template <ui8 StoresPerLoad, ui8 Cols>
        static void PackTupleOrImpl(const ui8 *const (&src_cols)[Cols],
                                    ui8 *const dst_rows, size_t size,
                                    const ui8 (&col_sizes)[Cols]) {
            // ui8 type used for sizes as a reminder,
            // that tuple is small and shoud fit into one reg
            ui8 tuple_size = 0;
            for (ui8 col = 0; col != Cols; ++col) {
                tuple_size += col_sizes[col];
            };

            const ui8 tuples_per_store = TSimd<ui8>::SIZE / tuple_size;
            ui8 col_store_sizes[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                col_store_sizes[col] = col_sizes[col] * tuples_per_store;
            };

            TSimd<ui8> perms[Cols][StoresPerLoad];
            for (ui8 col = 0, offset = 0; col != Cols; ++col) {
                for (ui8 ind = 0; ind != StoresPerLoad; ++ind) {
                    perms[col][ind] = BuildTuplePerm(
                        col_sizes[col], ui8(tuple_size - col_sizes[col]),
                        offset, ind * col_store_sizes[col]);
                }
                offset += col_sizes[col];
            }

            TSimd<ui8> src_regs[Cols];
            TSimd<ui8> perm_regs[Cols];

            const ui8 *srcs[Cols];
            std::memcpy(srcs, src_cols, sizeof(srcs));

            const size_t simd_iters = size / (tuples_per_store * StoresPerLoad);
            ui8 *const end = dst_rows + simd_iters * tuples_per_store *
                                            StoresPerLoad * tuple_size;

            TSimd<ui8> tmp;

            auto dst = dst_rows;
            while (dst != end) {
                for (ui8 col = 0; col != Cols; ++col) {
                    src_regs[col] = TSimd<ui8>(srcs[col]);
                    srcs[col] += col_store_sizes[col] * StoresPerLoad;
                }

                for (ui8 iter = 0; iter != StoresPerLoad; ++iter) {
                    // shuffling each col bytes to the right positions
                    // then blending them together with 'or'
                    for (ui8 col = 0; col != Cols; ++col) {
                      perm_regs[col] = src_regs[col].Shuffle(perms[col][iter]);
                    }

                    // tmp |= TupleOr(perm_regs);
                    TupleOr(perm_regs).Store(dst);
                    dst += tuple_size * tuples_per_store;
                }
            }

            PackTupleFallbackRowImpl(
                srcs, dst, size - simd_iters * tuples_per_store * StoresPerLoad,
                col_sizes);
        }

        template <class... Types> struct PackBenchCase {
            static constexpr size_t Iters = 2;
            static constexpr size_t Memory = 1ul << 28;
            static constexpr size_t TupleSize = (sizeof(Types) + ...);
            static constexpr size_t Cols = sizeof...(Types);
            static constexpr size_t NTuples = Memory / TupleSize;

            static constexpr size_t StoresPerLoad =
                std::min(Cols, TSimd<ui8>::SIZE / std::max({sizeof(Types)...}) /
                                   (TSimd<ui8>::SIZE / TupleSize));
            using PackTupleF = void (*)(const ui8 *const (&)[Cols], ui8 *const,
                                        size_t, const ui8 (&)[Cols]);

            static constexpr std::pair<PackTupleF, const char *> Runs[] = {
                {&PackTupleFallbackRowImpl<Cols>, "fallback (row)"},
                {&PackTupleFallbackColImpl<Cols>, "fallback (col)"},
                {&PackTupleFallbackBlockImpl<Cols>, "fallback (block)"},
                {&PackTupleOrImpl<StoresPerLoad, Cols>, "or"},
            };

            static void Run() {
#define INDEXED_TYPES(...)                                                     \
    [&]<size_t... Is>(std::index_sequence<Is...>) {                            \
      __VA_ARGS__                                                              \
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
                const ui8 *src_cols[Cols];

                INDEXED_TYPES((srcs[Is].reset(new (std::align_val_t(32))
                                                  ui8[NTuples * sizeof(Types)]),
                               ...);
                              ((src_cols[Is] = srcs[Is].get()), ...););

                for (size_t ind = 0; ind < NTuples; ind++) {
                    INDEXED_TYPES(((reinterpret_cast<Types *>(
                                        srcs[Is].get())[ind] = Cols * ind + Is),
                                   ...););
                }

                std::unique_ptr<ui8[]> dst{new (std::align_val_t(32))
                                               ui8[NTuples * TupleSize]};
                ui8 *dst_rows = dst.get();

                for (const auto &[func, name] : Runs) {
                    Cerr << "\nrunning: " << name << '\n';
                    for (ui8 iters = 0; iters != Iters; ++iters) {

                      dst[0] = 1;
                      for (size_t ind = 1; ind != NTuples * TupleSize; ++ind) {
                        dst[ind] = dst[ind - 1] * 127 + 1;
                      }

                      const std::chrono::steady_clock::time_point begin_ts =
                          std::chrono::steady_clock::now();

                      std::invoke(func, src_cols, dst.get(), NTuples,
                                  col_sizes);

                      const std::chrono::steady_clock::time_point end_ts =
                          std::chrono::steady_clock::now();
                      const ui64 us =
                          std::chrono::duration_cast<std::chrono::microseconds>(
                              end_ts - begin_ts)
                              .count();
                      const size_t mbs =
                          (NTuples * TupleSize * 1000ul * 1000ul) /
                          (us * 1024ul * 1024ul + 1);

                      size_t fails = 0;
                      for (size_t row = 0; row < NTuples; row += 1) {
                        bool flag = true;
                        INDEXED_TYPES(flag = (flag && ... &&
                                              (reinterpret_cast<Types *>(
                                                   dst_rows + row * TupleSize +
                                                   offsets[Is])[0] ==
                                               Types(Cols * row + Is))););
                        if (flag) {
                          continue;
                        }
                        ++fails;

                        INDEXED_TYPES(
                            (...,
                             (Cerr << row << ": "
                                   << size_t(reinterpret_cast<Types *>(
                                          dst_rows + row * TupleSize +
                                          offsets[Is])[0])
                                   << " =?= " << (Cols * row + Is) << '\n'));

                        );
                      }

                      Cerr << mbs << " mib/s, fails: " << fails << '\n';
                    }
                }

#undef INDEXED_TYPES
            }
        };

        void CmpPackTupleOrAndFallback() override {
            Cerr << " --- Compare simd-or and fallback copy --- \n";

            PackBenchCase<ui64>::Run();
            PackBenchCase<ui32, ui32>::Run();
            PackBenchCase<ui16, ui16, ui16, ui16>::Run();
            PackBenchCase<ui8, ui8, ui8, ui8, ui8, ui8, ui8, ui8>::Run();
            PackBenchCase<ui8, ui8, ui8, ui8, ui8, ui8>::Run();
            PackBenchCase<ui8, ui8, ui8, ui8>::Run();
            PackBenchCase<ui32, ui32, ui32, ui32, ui32, ui32, ui32,
                          ui32>::Run();
            PackBenchCase<ui32, ui32, ui32, ui32, ui32, ui32>::Run();
            PackBenchCase<ui32, ui32, ui32, ui32>::Run();
            PackBenchCase<ui32, ui32, ui32>::Run();
            PackBenchCase<ui64, ui64, ui64, ui64>::Run();
            PackBenchCase<ui8, ui16, ui32>::Run();
            PackBenchCase<ui8, ui16, ui8, ui8, ui16, ui8>::Run();
        }

        ~TWorker() = default;
    };

    template<typename TTraits>
    THolder<TWrapWorker> Create() const {
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

    template <class TOffset, bool NullTerminated = false,
              bool StorePrefixes = false>
    static void PackBinaryImpl(ui8 *dst, ui8 *const dst_buffer, const ui8 *src,
                               const TOffset *src_offsets, size_t size,
                               size_t max_len, size_t padding) {
        assert(max_len >= sizeof(TOffset));

        TOffset dst_buf_offset = 0;
        src += src_offsets[0];

        for (size_t ind = 0; ind != size; ++ind) {
            const TOffset bin_size = src_offsets[ind + 1] - src_offsets[ind];

            *reinterpret_cast<TOffset *>(dst) = bin_size;
            dst += sizeof(bin_size);

            if (bin_size + NullTerminated <= max_len) {
                std::memcpy(dst, src, bin_size);
                if constexpr (NullTerminated) {
                    dst[bin_size] = '\0';
                }
            } else {
                *reinterpret_cast<TOffset *>(dst) = dst_buf_offset;
                std::memcpy(dst_buffer + dst_buf_offset, src, bin_size);
                dst_buf_offset += bin_size;

                if constexpr (NullTerminated) {
                    dst_buffer[dst_buf_offset] = '\0';
                    ++dst_buf_offset;
                }

                if constexpr (StorePrefixes) {
                    std::memcpy(dst + sizeof(TOffset), src,
                                max_len - sizeof(TOffset));
                }
            }

            dst += max_len + padding;
            src += bin_size;
        }
    }

    template <class TOffset, bool NullTerminated = false>
    static void UnpackBinaryImpl(ui8 *dst, TOffset *const dst_offsets,
                                 const ui8 *src, const ui8 *src_buffer,
                                 size_t size, size_t max_len, size_t padding) {
        assert(max_len >= sizeof(TOffset));

        TOffset src_buf_offset = 0;
        dst_offsets[0] = 0;

        for (size_t ind = 0; ind != size; ++ind) {
            const size_t bin_size = *reinterpret_cast<const TOffset *>(src);
            src += sizeof(TOffset);

            dst_offsets[ind + 1] = dst_offsets[ind] + bin_size;

            if (bin_size + NullTerminated <= max_len) {
                std::memcpy(dst, src, bin_size);
            } else {
                assert(*reinterpret_cast<const TOffset *>(src) ==
                       src_buf_offset);
                std::memcpy(dst, src_buffer + src_buf_offset, bin_size);
                src_buf_offset += bin_size + NullTerminated;
            }

            src += max_len + padding;
            dst += bin_size;
        }
    }

    template <template <class T> class Distribution, auto... Args,
              class TOffset = uint32_t>
    static void PackBinaryTest(const char *case_name) {
        std::mt19937 gen(1);
        Distribution<TOffset> dist{Args...};

        static constexpr size_t MEM = 1ul << 30;
        static constexpr size_t MAXSIZ = MEM / sizeof(TOffset) - 1;

        ui8 *src = new ui8[MEM];
        std::memset(src, 2, MEM);

        TOffset *src_offsets = new TOffset[MAXSIZ + 1];
        src_offsets[0] = 0;

        size_t size;
        for (size = 0; size != MAXSIZ; ++size) {
            const size_t str_siz = dist(gen);
            const size_t next_offset = src_offsets[size] + str_siz;

            if (next_offset > MEM) {
                break;
            }
            src_offsets[size + 1] = next_offset;
        }

        const auto stats = CollectBinaryStats(src_offsets, size);
        const size_t maxlen =
            std::max(sizeof(TOffset), stats.mean + size_t(1. * stats.std));
        const size_t padding = 4;
        const size_t row_len = sizeof(TOffset) + maxlen + padding;

        Cerr << "stats for: " << case_name << '\n';
        Cerr << "µ = " << stats.mean << ", σ = " << stats.std
             << "; len = " << maxlen << '\n';

        ui8 *dst = new ui8[size * row_len];
        std::memset(dst, 5, size * row_len);
        ui8 *dst_buf = new ui8[MEM];
        std::memset(dst_buf, 7, MEM);

        std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();

        PackBinaryImpl(dst, dst_buf, src, src_offsets, size, maxlen, padding);

        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();

        ui64 us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                .count();
        const size_t mbs_pack =
            (src_offsets[size] * 1000ul * 1000ul) / (us * 1024ul * 1024ul + 1);

        ui8 *src_copy = new ui8[MEM];
        std::memset(src_copy, 24, MEM);

        TOffset *src_offsets_copy = new TOffset[MAXSIZ + 1];
        std::memset(src_offsets_copy, 24, sizeof(TOffset) * MAXSIZ);

        begin = std::chrono::steady_clock::now();

        UnpackBinaryImpl(src_copy, src_offsets_copy, dst, dst_buf, size, maxlen,
                         padding);

        end = std::chrono::steady_clock::now();

        us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                 .count();
        const size_t mbs_unpack =
            (src_offsets[size] * 1000ul * 1000ul) / (us * 1024ul * 1024ul + 1);

        const bool failed = std::memcmp(src_offsets, src_offsets_copy,
                                        sizeof(TOffset) * (size + 1)) ||
                            std::memcmp(src, src_copy, src_offsets[size]);

        Cerr << "mb/s: " << mbs_pack << " (pack), " << mbs_unpack
             << " (unpack), " << (failed ? "failed" : "ok") << '\n';

        delete[] src;
        delete[] src_offsets;
        delete[] dst;
        delete[] dst_buf;
        delete[] src_copy;
        delete[] src_offsets_copy;

        Cerr << '\n';
    }
};

int main() {
    if (!NX86::HaveAVX2())
        return 0;

    TPerfomancer::PackBinaryTest<std::uniform_int_distribution, 8, 32>("uni(8, 32)");
    TPerfomancer::PackBinaryTest<std::uniform_int_distribution, 16, 128>("uni(16, 128)");
    TPerfomancer::PackBinaryTest<std::uniform_int_distribution, 32, 256>("uni(32, 256)");

    TPerfomancer tp;
    auto worker = tp.Create<NSimd::TSimdAVX2Traits>();

    bool fine = true;
    fine &= worker->PackTuple(false);

    worker->CmpPackTupleOrAndFallback();

    return !fine;
}
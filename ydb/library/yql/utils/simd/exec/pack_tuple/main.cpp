#include <util/generic/ptr.h>
#include <util/system/cpu_id.h>
#include <util/system/types.h>

#include <ydb/library/yql/utils/simd/simd.h>

struct TPerfomancer {
    TPerfomancer() = default;

    struct TWrapWorker {
        virtual int PackTuple(bool log) = 0;
        virtual void CmpPackTupleOrAndNaive() = 0;
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
        static void
        PackTupleNaiveImpl(const ui8 *const (&src_cols)[Cols],
                          ui8 *const dst_rows, size_t size,
                          const ui8 (&col_sizes)[Cols]) {
            size_t tuple_total_size = 0;
            size_t offsets[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                offsets[col] = tuple_total_size;
                tuple_total_size += col_sizes[col];
            }

            for (size_t row = 0; row != size; ++row) {
                for (ui8 col = 0; col != Cols; ++col) {
                    memcpy(dst_rows + row * tuple_total_size + offsets[col],
                           src_cols[col] + row * col_sizes[col],
                           col_sizes[col]);
                }
            }
        }

        static TSimd<ui8>
        BuildTuplePerm(ui8 col_size, ui8 col_pad, ui8 offset, ui8 ind) {
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
        static void
        PackTupleOrImpl(const ui8 *const (&src_cols)[Cols],
                           ui8 *const dst_rows, size_t size, const ui8 (&col_sizes)[Cols]) {
            // ui8 type used for sizes as a reminder, 
            // that tuple is small and shoud fit into one reg
            ui8 tuple_total_size = 0;
            for (ui8 col = 0; col != Cols; ++col) {
                tuple_total_size += col_sizes[col];
            };

            const ui8 tuples_per_store = TSimd<ui8>::SIZE / tuple_total_size;
            ui8 col_store_sizes[Cols];
            for (ui8 col = 0; col != Cols; ++col) {
                col_store_sizes[col] = col_sizes[col] * tuples_per_store;
            };

            TSimd<ui8> perms[Cols][StoresPerLoad];
            for (ui8 col = 0, offset = 0; col != Cols; ++col) {    
                for (ui8 ind = 0; ind != StoresPerLoad; ++ind) {
                    perms[col][ind] = BuildTuplePerm( col_sizes[col],
                                     ui8(tuple_total_size - col_sizes[col]),
                                    offset,  ind * col_store_sizes[col]);
                }
                offset += col_sizes[col];
            }

            TSimd<ui8> src_regs[Cols];
            TSimd<ui8> perm_regs[Cols];

            const ui8 *srcs[Cols];
            std::memcpy(srcs, src_cols, sizeof(srcs));

            const size_t simd_iters = size / (tuples_per_store * StoresPerLoad);
            ui8 *const end = dst_rows + simd_iters * tuples_per_store * StoresPerLoad * tuple_total_size;
            
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
                    dst += tuple_total_size * tuples_per_store;
                }
            }

            PackTupleNaiveImpl(srcs, dst,
                              size - simd_iters * tuples_per_store * StoresPerLoad, col_sizes);
        }

        template <class... Types> struct PackBenchCase {
            static constexpr size_t Iters = 1; // value > 1 affects inlining
            static constexpr size_t Memory = 1 << 28;
            static constexpr size_t TupleSize = (sizeof(Types) + ...);
            static constexpr size_t Cols = sizeof...(Types);
            static constexpr size_t NTuples = Memory / TupleSize;

            static constexpr size_t StoresPerLoad =
                std::min(Cols, TSimd<ui8>::SIZE / std::max({sizeof(Types)...}) /
                                   (TSimd<ui8>::SIZE / TupleSize));
            using PackTupleF = void (*)(const ui8 *const (&)[Cols], ui8 *const,
                                        size_t, const ui8 (&)[Cols]);

            static constexpr std::pair<PackTupleF, const char *> Runs[] = {
                // order affects naive (memcpy) performance
                {&PackTupleNaiveImpl<Cols>, "naive"},
                {&PackTupleOrImpl<StoresPerLoad, Cols>, "or"},
            };

            static void Run() {
#define INDEXED_TYPES(...)                                                     \
    [&]<class... ATypes, size_t... Is>(const std::tuple<ATypes...> &,          \
                                       std::index_sequence<Is...>) {           \
      __VA_ARGS__                                                              \
    }(std::tuple<Types...>{}, std::make_index_sequence<Cols>{})

#define INDEXED_TYPES_B                                                        \
    [&]<class... ATypes, size_t... Is>(const std::tuple<ATypes...> &,          \
                                       std::index_sequence<Is...>) {

#define INDEXED_TYPES_E                                                        \
    }                                                                          \
    (std::tuple<Types...>{}, std::make_index_sequence<Cols>{})

                Cerr << "\n - cols byte alignment";
                (..., (Cerr << ' ' << sizeof(Types)));
                Cerr << '\n';

                const ui8 col_sizes[Cols] = {sizeof(Types) ...};
                size_t offsets[Cols];
                for (ui8 col = 0, offset = 0; col != Cols; ++col) {
                    offsets[col] = offset;
                    offset += col_sizes[col];
                }

                std::unique_ptr<ui8[]> srcs[sizeof...(Types)];
                const ui8* src_cols[Cols];

                INDEXED_TYPES((srcs[Is].reset(new (std::align_val_t(
                                   32)) ui8[NTuples * sizeof(ATypes)]),
                               ...);
                               ((src_cols[Is] = srcs[Is].get()), ...);
                               );

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

        void CmpPackTupleOrAndNaive() override {
            Cerr << " --- Compare simd-or and naive copy --- \n";

            PackBenchCase<ui64>::Run();
            PackBenchCase<ui32, ui32>::Run();
            PackBenchCase<ui32, ui32, ui32>::Run();
            PackBenchCase<ui32, ui32, ui32, ui32>::Run();
            PackBenchCase<ui8, ui16, ui32>::Run();
            PackBenchCase<ui8, ui16, ui8, ui8>::Run();
            PackBenchCase<ui16, ui16, ui16, ui16>::Run();
        }

        ~TWorker() = default;
    };

    template<typename TTraits>
    THolder<TWrapWorker> Create() const {
        return MakeHolder<TWorker<TTraits>>();
    };
};

int main() {
    if (!NX86::HaveAVX2())
        return 0;

    TPerfomancer tp;
    auto worker = tp.Create<NSimd::TSimdAVX2Traits>();

    bool fine = true;
    fine &= worker->PackTuple(false);

    worker->CmpPackTupleOrAndNaive();

    return !fine;
}
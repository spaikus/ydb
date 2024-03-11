#pragma once

#include <util/system/cpu_id.h>
#include <util/system/types.h>
#include <vector>
#include <stdlib.h>

#include "simd_avx2.h"
#include "simd_sse42.h"
#include "simd_fallback.h"

namespace NSimd {

template<int RegisterSize, typename TBaseRegister, template<typename> typename TSimd>
struct TSimdTraits {
    using TRegister = TBaseRegister;
    template<typename T>
    using TSimd8 = TSimd<T>;
    using TSimdI8 = TSimd8<i8>;
    static constexpr int Size = RegisterSize;
};

using TSimdAVX2Traits = TSimdTraits<32, __m256i, NSimd::NAVX2::TSimd8>;
using TSimdSSE42Traits = TSimdTraits<16, __m128i, NSimd::NSSE42::TSimd8>;
using TSimdFallbackTraits = TSimdTraits<8, ui64, NSimd::NFallback::TSimd8>;


template<typename TFactory>
auto SelectSimdTraits(const TFactory& factory) {
    if (NX86::HaveAVX2()) {
        return factory.template Create<TSimdAVX2Traits>();
    } else {
        return factory.template Create<TSimdSSE42Traits>();
    }
}

// Creates unpack mask for Simd register content. dataSize - value in bytes to unpack, stripeSize - distance between content parts.
// when needOffset is true, first data part starts at stipeSize bytes in result register
template<typename TTraits>
auto CreateUnpackMask(ui32 dataSize, ui32 stripeSize, bool needOffset) {

    using TSimdI8 = typename TTraits::template TSimd8<i8>;
    i8 indexes[TTraits::Size];

    bool insideStripe = needOffset;
    ui32 stripeOffset = 0;
    ui32 currOffset = 0;
    ui32 dataOffset = 0;
    ui32 currDataSize = 0;

    while ( currOffset < TTraits::Size) {
        if (insideStripe) {
            if (stripeOffset >= stripeSize) {
                insideStripe = false;
                currDataSize = 0;
                stripeOffset = 0;
            } else {
                indexes[currOffset++] = -1;
                stripeOffset++;
            }
        } else {
            indexes[currOffset++] = dataOffset++;
            currDataSize++;
            if (currDataSize >= dataSize) {
                insideStripe = true;
                currDataSize = 0;
                stripeOffset = 0;
            }
        }
    }

    return TSimdI8(indexes);
}


// Creates mask to advance register content for N bytes. When N is negative, move data to lower bytes.
template<typename TTraits> auto AdvanceBytesMask(const int N) {
    i8 positions[TTraits::Size];
    if (N < 0) {
        for (int i = 0; i < TTraits::Size; i += 1) {
            positions[i] = -N + i > (TTraits::Size - 1) ? -1 : -N + i;
        }
    } else {
        for (int i = 0; i < TTraits::Size; i += 1) {
            positions[i] = -N + i < 0 ? -1 : -N + i;
        }
    }
    return typename TTraits::TSimdI8(positions);
}


// Prepare unpack mask to merge two columns in one register. col1Bytes, col2Bytes - size of data in columns.
template<typename TTraits>
void PrepareMergeMasks( ui32 col1Bytes, ui32 col2Bytes, typename TTraits::TSimdI8& unpackMask1, typename TTraits::TSimdI8& unpackMask2) {
    unpackMask1 = CreateUnpackMask<TTraits>(col1Bytes, col2Bytes, false);
    unpackMask2 = CreateUnpackMask<TTraits>(col2Bytes, col1Bytes, true);
}

using AVX2Trait = NSimd::NAVX2::TSimd8<i8>;

using SSE42Trait = NSimd::NSSE42::TSimd8<i8>;

using FallbackTrait = NSimd::NFallback::FallbackTrait<i8>;

inline void FallbackMergeColumns(i8 *result, i8 *const data[4], size_t sizes[4],
                                 size_t length, size_t from) {
    if (length < from) {
        return;
    }

    const i8 *srcs[4];
    i8 *dst;
    size_t col_sizes[4];

    [&]<size_t... Is>(std::index_sequence<Is...>) {
      const size_t pack_size = (... + sizes[Is]);
      (..., (srcs[Is] = data[Is] + from * sizes[Is]));
      dst = result + from * pack_size;
      (..., (col_sizes[Is] = sizes[Is]));
    }(std::make_index_sequence<4>{});

    // merge_columns/Fallback_algo/fallback_algo.cpp
    void FallbackMergeBlockImpl(const i8 *const(&src_cols)[4],
                                i8 *const dst_rows, size_t size,
                                const size_t(&col_sizes)[4]);

    FallbackMergeBlockImpl(srcs, dst, length - from, col_sizes);
}

struct Perfomancer {

    Perfomancer() = default;

    struct Interface {

        virtual ~Interface() = default;

        inline virtual void MergeColumns(i8* result, i8* const data[4], size_t sizes[4], size_t length) = 0;

    };

    template <typename Trait>
    class Algo : public Interface {
    public:
        Algo() {}

        void MergeColumns(i8* result, i8* const data[4], size_t sizes[4], size_t length) override {
            std::vector<Trait> reg(20);
            std::vector<Trait> mask(19);

            int pack = (sizes[0] + sizes[1] + sizes[2] + sizes[3]);
            int block = Trait::SIZE / pack * pack;

            PrepareMask(sizes, mask);

            //const size_t stores = std::min(4ul, Trait::SIZE / sizes[3]);
            size_t i = 0;

            for (; i * sizes[0] + Trait::SIZE < length * sizes[0]; i += Trait::SIZE / pack * 4) {
                Iteration(sizes, data, result, i, pack * i, block, reg, mask);
            }
            FallbackMergeColumns(result, data, sizes, length, i);
        }

        ~Algo() = default;
    
    private:
        Trait CreateBlendMask(size_t size1, size_t size2, bool shift) {
            i8 result[Trait::SIZE];

            size_t cnt = 0;

            if (shift) {
                for (size_t i = 0; i < size2; ++i) {
                    result[cnt++] = 0xFF;
                }
            }

            while (cnt + size1 + size2 <= Trait::SIZE) {

                for (size_t i = 0; i < size1; ++i) {
                    result[cnt++] = 0x00;
                }

                if (cnt + size2 > Trait::SIZE) break;

                for (size_t i = 0; i < size2; ++i) {
                    result[cnt++] = 0xFF;
                }
            }

            Trait reg;
            reg.SetMask(result);
            return reg;
        }

        Trait CreateShuffleToBlendMask(size_t size1, size_t size2, size_t order, bool shift) {
            size_t cnt = 0;
            i8 result[Trait::SIZE];

            while (cnt < Trait::SIZE) {

                //0000000123456700008910111200000...
                if (shift) {
                    if (cnt % (size1 + size2) < size2) {
                        result[cnt++] = 0x80;
                    } else {
                        result[cnt++] = order++;
                    }
                } else {
                    if (cnt % (size1 + size2) < size1) {
                        result[cnt++] = order++;
                    } else {
                        result[cnt++] = 0x80;
                    }
                }
            }

            Trait reg(result);
            return reg;
        }

        void PrepareMasks(size_t sizes[4], std::vector<Trait>& mask) {

            int pack = sizes[0] + sizes[1] + sizes[2] + sizes[3];

            mask[0] = CreateShuffleToBlendMask(sizes[0], sizes[1], 0, false);
            mask[1] = CreateShuffleToBlendMask(sizes[0], sizes[1], Trait::SIZE / (sizes[0] + sizes[1]) * sizes[0], false);

            mask[2] = CreateShuffleToBlendMask(sizes[1], sizes[0], 0, true);
            mask[3] = CreateShuffleToBlendMask(sizes[1], sizes[0], Trait::SIZE / (sizes[0] + sizes[1]) * sizes[1], true);

            mask[4] = CreateShuffleToBlendMask(sizes[2], sizes[3], 0, false);
            mask[5] = CreateShuffleToBlendMask(sizes[2], sizes[3], Trait::SIZE / (sizes[2] + sizes[3]) * sizes[2], false);

            mask[6] = CreateShuffleToBlendMask(sizes[3], sizes[2], 0, true);
            mask[7] = CreateShuffleToBlendMask(sizes[3], sizes[2], Trait::SIZE / (sizes[2] + sizes[3]) * sizes[2], true);

            mask[8] = CreateShuffleToBlendMask(sizes[0] + sizes[1], sizes[2] + sizes[3], 0, false);
            mask[9] = CreateShuffleToBlendMask(sizes[0] + sizes[1], sizes[2] + sizes[3], Trait::SIZE / pack * (sizes[0] + sizes[1]), false);
            
            mask[10] = CreateShuffleToBlendMask(sizes[2] + sizes[3], sizes[0] + sizes[1], 0, true);
            mask[11] = CreateShuffleToBlendMask(sizes[2] + sizes[3], sizes[0] + sizes[1], Trait::SIZE / pack * (sizes[2] + sizes[3]), true);

            mask[12] = CreateBlendMask(sizes[0], sizes[1], false);
            mask[13] = CreateBlendMask(sizes[2], sizes[3], false);
            mask[14] = CreateBlendMask(sizes[0] + sizes[1], sizes[2] + sizes[3], false);
        }
        Trait M1(int order, int start, size_t sizes[4]) {
            i8 res[Trait::SIZE];
            int pack = sizes[0] + sizes[1] + sizes[2] + sizes[3];
            int cnt = 0;
            int lb = 0;
            for (int i = 0; i < order; ++i) {
                lb += sizes[i];
            }
            int ub = lb + sizes[order];
            while (cnt < Trait::SIZE) {
                if ((cnt % pack >= lb) && (cnt % pack < ub)) {
                    res[cnt++] = start++;
                } else {
                    res[cnt++] = 0x80;
                }
            }
            Trait reg(res);
            return reg;
        }
        void M2(size_t sizes[4], std::vector<Trait>& masks) {
            i8 res1[Trait::SIZE];
            i8 res2[Trait::SIZE];
            i8 res3[Trait::SIZE];
            int pack = sizes[0] + sizes[1] + sizes[2] + sizes[3];
            int cnt = 0;
            for (int i = 0; i < Trait::SIZE; ++i) {
                res1[i] = 0xFF;
                res2[i] = 0xFF;
                res3[i] = 0xFF;
            }
            while (cnt < Trait::SIZE) {
                if (cnt % pack < int(sizes[0])) {
                    res1[cnt] = 0x00;
                    res2[cnt] = 0x00;
                    res3[cnt] = 0x00;
                }
                if (cnt % pack < int(sizes[0] + sizes[1])) {
                    res2[cnt] = 0x00;
                    res3[cnt] = 0x00;
                }
                if (cnt % pack < int(sizes[0] + sizes[1] + sizes[2])) {
                    res3[cnt] = 0x00;
                }
                cnt++;
            }
            Trait r1;
            r1.SetMask(res1);
            Trait r2;
            r2.SetMask(res2);
            Trait r3;
            r3.SetMask(res3);
            masks[16] = r1;
            masks[17] = r2;
            masks[18] = r3;
        }

        void PrepareMask(size_t sizes[4], std::vector<Trait>& masks) {
            size_t pack = sizes[0] + sizes[1] + sizes[2] + sizes[3];
            int cnt = Trait::SIZE / pack;
            M2(sizes, masks);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    masks[i * 4 + j] = M1(i, j * cnt * sizes[i], sizes);
                }
            }
        }


        void Iteration(size_t sizes[4], i8* const data[4], i8* result, int ind, int addr, int step, std::vector<Trait>& reg, std::vector<Trait>& mask) {
            reg[16].Get(&data[0][ind * sizes[0]]);
            reg[17].Get(&data[1][ind * sizes[1]]);
            reg[18].Get(&data[2][ind * sizes[2]]);
            reg[19].Get(&data[3][ind * sizes[3]]);
            
            for (int i = 0; i < 16; ++i) {
                reg[i] = reg[16 + i / 4].template Shuffle(mask[i]);
            }
            reg[16] = reg[0].Blend(reg[4], mask[16]);
            reg[17] = reg[8].Blend(reg[12], mask[18]);
            reg[18] = reg[16].Blend(reg[17], mask[17]); //ok
            reg[0] = reg[1].Blend(reg[5], mask[16]);
            reg[4] = reg[9].Blend(reg[13], mask[18]);
            reg[8] = reg[0].Blend(reg[4], mask[17]); //ok

            reg[1] = reg[2].Blend(reg[6], mask[16]);
            reg[5] = reg[10].Blend(reg[14], mask[18]);
            reg[9] = reg[1].Blend(reg[5], mask[17]); //ok
            
            reg[2] = reg[3].Blend(reg[7], mask[16]);
            reg[6] = reg[11].Blend(reg[15], mask[18]);
            reg[10] = reg[2].Blend(reg[6], mask[17]); //ok
            
            reg[18].Store(&result[addr]);
            reg[8].Store(&result[addr + step]);
            reg[9].Store(&result[addr + 2 * step]);
            reg[10].Store(&result[addr + 3 * step]);
        }


        // void Iteration(size_t sizes[4], i8* const data[4], i8* result, int ind, int addr, int step, std::vector<Trait>& reg, std::vector<Trait>& mask) {
        //     reg[0].Get(&data[0][ind * sizes[0]]);
        //     reg[1].Get(&data[1][ind * sizes[1]]);
        //     reg[2].Get(&data[2][ind * sizes[2]]);
        //     reg[3].Get(&data[3][ind * sizes[3]]);

        //     reg[4] = reg[0].template Shuffle<false>(mask[0]);

        //     reg[5] = reg[0].template Shuffle<false>(mask[1]);

        //     reg[6] = reg[1].template Shuffle<false>(mask[2]);
        //     reg[7] = reg[1].template Shuffle<false>(mask[3]);

        //     reg[8] = reg[2].template Shuffle<false>(mask[4]);
        //     reg[9] = reg[2].template Shuffle<false>(mask[5]);

        //     reg[10] = reg[3].template Shuffle<false>(mask[6]);
        //     reg[11] = reg[3].template Shuffle<false>(mask[7]);

        //     reg[12] = reg[4].Blend(reg[6], mask[12]);  //12121212
        //     reg[13] = reg[5].Blend(reg[7], mask[12]);  //12121212
        //     reg[14] = reg[8].Blend(reg[10], mask[13]); //34343434
        //     reg[15] = reg[9].Blend(reg[11], mask[13]); //34343434

        //     reg[0] = reg[12].template Shuffle<false>(mask[8]);
        //     reg[1] = reg[12].template Shuffle<false>(mask[9]);

        //     reg[2] = reg[13].template Shuffle<false>(mask[8]);
        //     reg[3] = reg[13].template Shuffle<false>(mask[9]);

        //     reg[4] = reg[14].template Shuffle<false>(mask[10]);
        //     reg[5] = reg[14].template Shuffle<false>(mask[11]);

        //     reg[6] = reg[15].template Shuffle<false>(mask[10]);
        //     reg[7] = reg[15].template Shuffle<false>(mask[11]);

        //     reg[8] = reg[0].Blend(reg[4], mask[14]);
        //     reg[9] = reg[1].Blend(reg[5], mask[14]);
        //     reg[10] = reg[2].Blend(reg[6], mask[14]);
        //     reg[11] = reg[3].Blend(reg[7], mask[14]);

        //     reg[8].Store(&result[addr]);
        //     reg[9].Store(&result[addr + step]);
        //     reg[10].Store(&result[addr + 2 * step]);
        //     reg[12].Store(&result[addr + 3 * step]);
        // }

        void MergeEnds(i8* result, i8* const data[4], size_t sizes[4], size_t length, size_t ind, int addr) {

            while (ind < length) {
                for (int i = 0; i < 4; ++i) {
                    memcpy(&result[addr], &data[i][ind * sizes[i]], sizes[i]);
                    addr += sizes[i];
                }
                ind++;
            }
        }
    };

    template <typename Trait>
    inline THolder<Interface> Create() {
        return MakeHolder<Interface>();
    }

};

template <> class Perfomancer::Algo<FallbackTrait> : public Interface {
    void MergeColumns(i8 *result, i8 *const data[4], size_t sizes[4],
                      size_t length) override {
        FallbackMergeColumns(result, data, sizes, length, 0);
    }
};

template<>
THolder<Perfomancer::Interface> Perfomancer::Create<AVX2Trait>();

template<>
THolder<Perfomancer::Interface> Perfomancer::Create<SSE42Trait>();

template<>
THolder<Perfomancer::Interface> Perfomancer::Create<FallbackTrait>();

template <typename TFactory>
auto ChooseTrait(TFactory& factory) {
    
    if (NX86::HaveAVX2()) {
        return factory.template Create<AVX2Trait>();
    
    } else if (NX86::HaveSSE42()) {
        return factory.template Create<SSE42Trait>();
    
    }
    
    return factory.template Create<FallbackTrait>();
}


}
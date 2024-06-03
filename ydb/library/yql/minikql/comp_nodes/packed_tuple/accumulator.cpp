#include "accumulator.h"

#include <cstdlib>
#include <ydb/library/yql/utils/simd/simd.h>

namespace NKikimr {
namespace NMiniKQL {
namespace NPackedTuple {

THolder<TAccumulator> TAccumulator::Create(TTupleLayout* layout, ui32 nBuckets) {
    if (NX86::HaveAVX2())
    {
        return MakeHolder<TAccumulatorImpl<NSimd::TSimdAVX2Traits>>(layout, nBuckets);
    }

    if (NX86::HaveSSE42())
    {
        return MakeHolder<TAccumulatorImpl<NSimd::TSimdSSE42Traits>>(layout, nBuckets);
    }

    return MakeHolder<TAccumulatorImpl<NSimd::TSimdFallbackTraits>>(layout, nBuckets);
}

// -----------------------------------------------------------------------

template <typename TTraits>
TAccumulatorImpl<TTraits>::TAccumulatorImpl(TTupleLayout* layout, ui32 nBuckets)
    : NBuckets_(nBuckets)
    , Layout_(layout)
    , SecondLevelAccum_(NBuckets_, nullptr)
    // std::max for case when TotalRowSize extremly big, so 32KB wouldn't be enough
    , FirstLevelMemLimit_(std::max<ui32>((Layout_->TotalRowSize + sizeof(ui32) + TTraits::Size) * NBuckets_, 32000 /* 32KB */))
    , FirstLevelBucketSize_(FirstLevelMemLimit_ / NBuckets_)
    , SecondLevelBucketSizes_(NBuckets_, 0)
    // std::max for case when TotalRowSize extremly big, so 4KB for L2 bucket wouldn't be enough
    , MinimalSecondLevelBucketSize_(std::max<ui32>(4000 /* 4KB */, 4 * (Layout_->TotalRowSize + TTraits::Size))) {

    Y_ASSERT(nBuckets > 0);
    Y_ASSERT(FirstLevelBucketSize_ > sizeof(ui32) + TTraits::Size);

    FirstLevelBucketSize_ = (FirstLevelBucketSize_ / TTraits::Size + 1) * TTraits::Size; // multiple of register width
    FirstLevelMemLimit_ = FirstLevelBucketSize_ * NBuckets_;
    FirstLevelAccum_ = static_cast<ui8*>(std::aligned_alloc(TTraits::Size, FirstLevelMemLimit_));
    std::memset(FirstLevelAccum_, 0, FirstLevelMemLimit_);
}

template <typename TTraits>
TAccumulatorImpl<TTraits>::~TAccumulatorImpl() {
    std::free(FirstLevelAccum_);
    for (ui32 i = 0; i < NBuckets_; ++i) {
        std::free(SecondLevelAccum_[i]);
    }
}

// -----------------------------------------------------------------------

template <typename TTraits>
void TAccumulatorImpl<TTraits>::AddData(const ui8* data, ui32 nItems) {
    using TSimd8 = typename TTraits:: template TSimd8<ui8>;

    ui32 tuplesPerFirstLevelBucket = std::min<ui32>(
        (FirstLevelBucketSize_ - sizeof(ui32) - TTraits::Size) / Layout_->TotalRowSize,
        255 /* 0xFF */);
    const ui8* end = data + nItems * Layout_->TotalRowSize;

    for (ui32 i = 0; i < nItems; ++i) {
        const ui8* tuple = data + i * Layout_->TotalRowSize;
        ui32 hash        = *reinterpret_cast<const ui32*>(tuple);
        ui32 bucketId    = hash & (NBuckets_ - 1);

        ui32 firstLevelBucketOffset = bucketId * FirstLevelBucketSize_;
        ui8* firstLevelBucketAddr   = FirstLevelAccum_ + firstLevelBucketOffset;

        ui32 nTuplesTotal       = *reinterpret_cast<ui32*>(firstLevelBucketAddr);
        ui32 nTuplesFirstLevel  = nTuplesTotal & 0xFF;
        ui32 nTuplesSecondLevel = nTuplesTotal >> 8;

        ui8* storeAddr{nullptr};

        if (nTuplesFirstLevel == tuplesPerFirstLevelBucket) {
            ui32 tuplesPerSecondLevelBucket = SecondLevelBucketSizes_[bucketId] == 0
                                              ? 0
                                              : (SecondLevelBucketSizes_[bucketId] - TTraits::Size) / Layout_->TotalRowSize;

            if (nTuplesSecondLevel == tuplesPerSecondLevelBucket) [[unlikely]] {
                ui32 newSize = SecondLevelBucketSizes_[bucketId] == 0
                               ? MinimalSecondLevelBucketSize_
                               : SecondLevelBucketSizes_[bucketId] * GrowthRate_;
                newSize = (newSize / TTraits::Size + 1) * TTraits::Size; // multiple of register width
                ui8* newBucket = static_cast<ui8*>(std::aligned_alloc(TTraits::Size, newSize));

                if (SecondLevelAccum_[bucketId] != nullptr) [[likely]] {
                    if constexpr (TTraits::Size > 8) { // SSE and AVX case
                        for (ui32 offset = 0; offset < SecondLevelBucketSizes_[bucketId]; offset += TTraits::Size) {
                            auto reg = TSimd8::LoadStream(SecondLevelAccum_[bucketId] + offset);
                            reg.StoreStream(newBucket + offset);
                        }
                    } else { // no SIMD case
                        std::memcpy(newBucket, SecondLevelAccum_[bucketId], SecondLevelBucketSizes_[bucketId]);
                    }
                    std::free(SecondLevelAccum_[bucketId]);
                }

                SecondLevelAccum_[bucketId] = newBucket;
                SecondLevelBucketSizes_[bucketId] = newSize;
            }

            storeAddr = SecondLevelAccum_[bucketId] + nTuplesSecondLevel * Layout_->TotalRowSize;
            nTuplesSecondLevel++;
        } else {
            storeAddr = firstLevelBucketAddr + sizeof(ui32) + nTuplesFirstLevel * Layout_->TotalRowSize;
            nTuplesFirstLevel++;
        }

        if constexpr (TTraits::Size > 8) { // SSE and AVX case
            // max for the case when the row size < register width
            ui32 bound = std::max<ui32>(Layout_->TotalRowSize, TTraits::Size);
            ui32 offset = 0;

            for (; offset < bound - TTraits::Size
                ; offset += TTraits::Size, tuple += TTraits::Size, storeAddr += TTraits::Size) {
                auto reg = TSimd8::Load(tuple);
                reg.Store(storeAddr);
            }

            // tail of the data copy via memcpy to avoid wrong memory access
            if (end - tuple < TTraits::Size) [[unlikely]] {
                std::memcpy(storeAddr, tuple, bound - offset);
            } else [[likely]] {
                auto reg = TSimd8::Load(tuple);
                reg.Store(storeAddr);
            }
        } else { // no SIMD case
            std::memcpy(storeAddr, tuple, Layout_->TotalRowSize);
        }

        nTuplesTotal = (nTuplesSecondLevel << 8) | nTuplesFirstLevel;
        *reinterpret_cast<ui32*>(firstLevelBucketAddr) = nTuplesTotal;
    }
}

template __attribute__((target("avx2"))) void
TAccumulatorImpl<NSimd::TSimdAVX2Traits>::AddData(const ui8* data, ui32 nItems);

template __attribute__((target("sse4.2"))) void
TAccumulatorImpl<NSimd::TSimdSSE42Traits>::AddData(const ui8* data, ui32 nItems);

// -----------------------------------------------------------------------

template <typename TTraits>
TAccumulator::BucketInfo TAccumulatorImpl<TTraits>::GetBucket(ui32 bucket) const {
    TAccumulator::BucketInfo result;
    ui8* bucketPtr = FirstLevelAccum_ + bucket * FirstLevelBucketSize_;

    result.FirstLevelBucket     = bucketPtr + sizeof(ui32);
    result.SecondLevelBucket    = SecondLevelAccum_[bucket];

    ui32 nTuplesTotal           = *reinterpret_cast<ui32*>(bucketPtr);
    result.FirstLevelElements   = nTuplesTotal & 0xFF;
    result.SecondLevelElements  = nTuplesTotal >> 8;
    result.Layout               = Layout_;

    return result;
}


} // namespace NPackedTuple
} // namespace NMiniKQL
} // namespace NKikimr

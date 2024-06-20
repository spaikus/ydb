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
    , FirstLevelMemLimit_(std::max<ui32>((Layout_->TotalRowSize + sizeof(ui32) + sizeof(ui8) + TTraits::Size) * NBuckets_, 32000 /* 32KB */))
    , FirstLevelBucketSize_(FirstLevelMemLimit_ / NBuckets_)
    , TuplesPerFirstLevelBucket_(std::min<ui32>((FirstLevelBucketSize_ - sizeof(ui32) - sizeof(ui8) - TTraits::Size) / Layout_->TotalRowSize, 255 /* 0xFF */))
    , SecondLevelBucketSizes_(NBuckets_, 0)
    , TuplesPerSecondLevelBucket_(NBuckets_, 0)
    // std::max for case when TotalRowSize extremly big, so 4KB for L2 bucket wouldn't be enough
    , MinimalSecondLevelBucketSize_(std::max<ui32>(4000 /* 4KB */, 4 * (Layout_->TotalRowSize + TTraits::Size))) {

    Y_ASSERT(nBuckets > 0);
    Y_ASSERT((nBuckets & (nBuckets - 1)) == 0);
    Y_ASSERT(FirstLevelBucketSize_ > sizeof(ui32) + TTraits::Size);

    FirstLevelBucketSize_ = (FirstLevelBucketSize_ / TTraits::Size + 1) * TTraits::Size; // multiple of register width
    FirstLevelMemLimit_ = FirstLevelBucketSize_ * NBuckets_;
    FirstLevelAccum_ = static_cast<ui8*>(std::aligned_alloc(TTraits::Size, FirstLevelMemLimit_));
    std::memset(FirstLevelAccum_, 0, FirstLevelMemLimit_);
    // Add info about offset after padding
    // [cnt|offset|...|values]
    for (ui32 i = 0; i < NBuckets_; ++i) {
        ui8* ptr = FirstLevelAccum_ + i * FirstLevelBucketSize_;
        auto ptrAfterOffset = reinterpret_cast<std::uintptr_t>(ptr + sizeof(ui32) + sizeof(ui8));
        // What value we need to add to get aligned address
        *(ptr + sizeof(ui32)) = (ui8)(TTraits::Size - ptrAfterOffset % TTraits::Size);
    }
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

    const auto layoutTotalRowSize = Layout_->TotalRowSize;
    const ui32 bound = std::max<ui32>(layoutTotalRowSize, TTraits::Size);
    const ui8* const end = data + nItems * Layout_->TotalRowSize;

    for (ui32 i = 0; i < nItems; ++i) {
        const ui8* tuple = data + i * layoutTotalRowSize;
        const ui32 bucketId = *reinterpret_cast<const ui32*>(tuple) & (NBuckets_ - 1);

        ui8* firstLevelBucketAddr = FirstLevelAccum_ + bucketId * FirstLevelBucketSize_;

        ui32 nCounters = *reinterpret_cast<ui32*>(firstLevelBucketAddr);
        ui32 nBucketTuplesL1 = nCounters & (0xFF);
        ui32 nBucketTuplesL2 = nCounters >> 8;

        ui8  nBucketDataOffset = *(firstLevelBucketAddr + sizeof(ui32));
        ui8* firstLevelDataAddr = firstLevelBucketAddr + sizeof(ui32) + sizeof(ui8) + nBucketDataOffset;
        ui8* storeAddr{nullptr};

        if (nBucketTuplesL1 >= TuplesPerFirstLevelBucket_) {
            const ui32 tuplesPerSecondLevelBucket = TuplesPerSecondLevelBucket_[bucketId];

            if (nBucketTuplesL2 + nBucketTuplesL1 >= tuplesPerSecondLevelBucket) [[unlikely]] {
                ui32 newSize =
                    tuplesPerSecondLevelBucket == 0
                        ? MinimalSecondLevelBucketSize_
                        : std::max({ui32(SecondLevelBucketSizes_[bucketId] * GrowthRate_),
                                    ui32(1 + nBucketTuplesL2 + ui64(nItems - i) * layoutTotalRowSize * AddReserveMul_ / NBuckets_)});

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
                TuplesPerSecondLevelBucket_[bucketId] = (SecondLevelBucketSizes_[bucketId] - TTraits::Size) / layoutTotalRowSize;
            }

            storeAddr = SecondLevelAccum_[bucketId] + nBucketTuplesL2 * layoutTotalRowSize;
            std::memcpy(storeAddr, firstLevelDataAddr, layoutTotalRowSize * nBucketTuplesL1);

            nBucketTuplesL2 += nBucketTuplesL1;
            nBucketTuplesL1 = 0;
        }

        storeAddr = firstLevelDataAddr + nBucketTuplesL1 * layoutTotalRowSize;

        if constexpr (TTraits::Size > 8) { // SSE and AVX case
            // max for the case when the row size < register width
            ui32 offset = 0;

            for (; offset < bound - TTraits::Size; offset += TTraits::Size, tuple += TTraits::Size, storeAddr += TTraits::Size) {
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
            std::memcpy(storeAddr, tuple, layoutTotalRowSize);
        }

        nBucketTuplesL1++;

        nCounters = (nBucketTuplesL2 << 8) | nBucketTuplesL1;
        *reinterpret_cast<ui32*>(firstLevelBucketAddr) = nCounters;
    }

    TotalTuples_ += nItems;
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
    const ui32 nCounters = *reinterpret_cast<ui32*>(bucketPtr);
    const ui32 nBucketTuplesL1 = nCounters & (0xFF);
    const ui32 nBucketTuplesL2 = nCounters >> 8;
    const ui8  nBucketDataOffset = *(bucketPtr + sizeof(ui32));

    result.FirstLevelBucket     = bucketPtr + sizeof(ui32) + sizeof(ui8) + nBucketDataOffset;
    result.SecondLevelBucket    = SecondLevelAccum_[bucket];

    result.FirstLevelElements   = nBucketTuplesL1;
    result.SecondLevelElements  = nBucketTuplesL2;
    result.Layout = Layout_;

    return result;
}


} // namespace NPackedTuple
} // namespace NMiniKQL
} // namespace NKikimr

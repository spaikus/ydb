#include "accumulator.h"

#include <cstdlib>
#include <ydb/library/yql/utils/simd/simd.h>

namespace NKikimr {
namespace NMiniKQL {
namespace NPackedTuple {


TAccumulator::TAccumulator(TTupleLayout* layout, ui32 nBuckets)
    : NBuckets_(nBuckets)
    , Layout_(layout)
    , SecondLevelAccum_(NBuckets_, nullptr)
    // std::max for case when TotalRowSize extremly big, so 32KB wouldn't be enough
    , FirstLevelMemLimit_(std::max<ui32>((Layout_->TotalRowSize + sizeof(ui32)) * NBuckets_, 32000 /* 32KB */))
    , FirstLevelBucketSize_(FirstLevelMemLimit_ / NBuckets_)
    , SecondLevelBucketSizes_(NBuckets_, 0)
    // std::max for case when TotalRowSize extremly big, so 4KB for L2 bucket wouldn't be enough
    , MinimalSecondLevelBucketSize_(std::max<ui32>(4000 /* 4KB */, 4 * Layout_->TotalRowSize)) {

    Y_ASSERT(nBuckets > 0);
    FirstLevelMemLimit_ = (FirstLevelMemLimit_ / 64 + 1) * 64; // multiple of 64
    FirstLevelAccum_ = static_cast<ui8*>(std::aligned_alloc(64 /* cache line size */, FirstLevelMemLimit_));
    std::memset(FirstLevelAccum_, 0, FirstLevelMemLimit_);
}

TAccumulator::~TAccumulator() {
    std::free(FirstLevelAccum_);
    for (ui32 i = 0; i < NBuckets_; ++i) {
        std::free(SecondLevelAccum_[i]);
    }
}

void TAccumulator::AddData(const ui8* data, ui32 nItems) {
    ui32 tuplesPerFirstLevelBucket = std::min<ui32>((FirstLevelBucketSize_ - sizeof(ui32)) / Layout_->TotalRowSize, 255 /* 0xFF */);
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
            if (nTuplesSecondLevel == SecondLevelBucketSizes_[bucketId] / Layout_->TotalRowSize) [[unlikely]] {
                ui32 newSize = SecondLevelBucketSizes_[bucketId] == 0
                               ? MinimalSecondLevelBucketSize_
                               : SecondLevelBucketSizes_[bucketId] * GrowthRate_;
                newSize = (newSize / 64 + 1) * 64; // multiple of 64
                ui8* newBucket = static_cast<ui8*>(std::aligned_alloc(64 /* cache line size */, newSize));
                if (SecondLevelAccum_[bucketId] != nullptr) [[unlikely]] {
                    std::memcpy(newBucket, SecondLevelAccum_[bucketId], SecondLevelBucketSizes_[bucketId]);
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

        std::memcpy(storeAddr, tuple, Layout_->TotalRowSize);
        nTuplesTotal = (nTuplesSecondLevel << 8) | nTuplesFirstLevel;
        *reinterpret_cast<ui32*>(firstLevelBucketAddr) = nTuplesTotal;
    }
}

TAccumulator::BucketInfo TAccumulator::GetBucket(ui32 bucket) {
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

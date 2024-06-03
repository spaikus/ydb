#pragma once

#include <ydb/library/yql/minikql/mkql_node.h>
#include <ydb/library/yql/public/udf/udf_value.h>
#include <ydb/library/yql/public/udf/udf_types.h>
#include <ydb/library/yql/public/udf/udf_data_type.h>

#include <util/generic/buffer.h>

#include "tuple.h"

namespace NKikimr {
namespace NMiniKQL {
namespace NPackedTuple {

/*
* Class TAccumulator is used to split provided vector of tuples to buckets using provided key hashes.
*/
class TAccumulator {   
public:
    struct BucketInfo {
        const ui8*      FirstLevelBucket;       // Pointer to start of first level bucket
        ui32            FirstLevelElements;     // Count of elements in first level bucket
        const ui8*      SecondLevelBucket;      // Pointer to start of second level bucket, possibly equals to nullptr
        ui32            SecondLevelElements;    // Count of elements in second level bucket
        TTupleLayout*   Layout;                 // Layout for packed row (tuple)
    };

public:
    virtual ~TAccumulator() {};

    // Creates new accumulator for nBuckets for given layout
    static THolder<TAccumulator> Create(TTupleLayout* layout, ui32 nBuckets = 64);

    // Adds new nItems of data in TTupleLayout representation to accumulator 
    virtual void AddData(const ui8* data, ui32 nItems) = 0;

    // Returns bucket info
    virtual BucketInfo GetBucket(ui32 bucket) const = 0;
};


template <typename TTrait>
class TAccumulatorImpl: public TAccumulator {   
public:
    TAccumulatorImpl(TTupleLayout* layout, ui32 nBuckets = 64);
    ~TAccumulatorImpl();

    void AddData(const ui8* data, ui32 nItems) override;

    BucketInfo GetBucket(ui32 bucket) const override;

private:
    ui32 NBuckets_{0};                                                  // Number of buckets
    TTupleLayout* Layout_;                                              // Tuple layout
    ui8* FirstLevelAccum_{nullptr};                                     // First level small accumulator.  Should fit into L1 cache
    std::vector<ui8*, TMKQLAllocator<ui8*>> SecondLevelAccum_;          // Second level accumulator data
    ui32 FirstLevelMemLimit_{0};                                        // Memory limit for first level accumulator
    ui32 FirstLevelBucketSize_{0};                                      // Fixed bucket size of level 1 accumulator
    std::vector<ui32, TMKQLAllocator<ui32>> SecondLevelBucketSizes_;    // Fixed bucket sizes for level 2 accumulator
    ui32 MinimalSecondLevelBucketSize_{0};                              // Fixed minimum initial size for second level bucket
    static constexpr double GrowthRate_{1.5};                           // Growth rate for second level buckets
};


} // namespace NPackedTuple
} // namespace NMiniKQL
} // namespace NKikimr

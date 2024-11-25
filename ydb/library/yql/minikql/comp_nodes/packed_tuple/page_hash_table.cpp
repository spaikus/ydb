#include "page_hash_table.h"

#include <ydb/library/yql/utils/simd/simd.h>
#include <arrow/util/bit_util.h>

namespace NKikimr {
namespace NMiniKQL {
namespace NPackedTuple {


// -----------------------------------------------------------------
THolder<TPageHashTable> TPageHashTable::Create(const TTupleLayout* layout) {
    if (NX86::HaveAVX2()) {
        return MakeHolder<TPageHashTableImpl<NSimd::TSimdAVX2Traits>>(layout);
    }

    if (NX86::HaveSSE42()) {
        return MakeHolder<TPageHashTableImpl<NSimd::TSimdSSE42Traits>>(layout);
    }

    return MakeHolder<TPageHashTableImpl<NSimd::TSimdFallbackTraits>>(layout);
}


// -----------------------------------------------------------------
template <typename TTraits>
TPageHashTableImpl<TTraits>::TPageHashTableImpl(const TTupleLayout* layout)
    : KeySize_(layout->KeyColumnsSize)
    , KeyOffset_(layout->KeyColumnsOffset)
    , PayloadSize_(layout->PayloadSize)
    , TupleSize_(layout->TotalRowSize)
    , PageHeaderSize_(SlotsCount_ + 1 /* One byte is used for filled slots count */)
    , PageSize_(PageHeaderSize_ + SlotSize_ * SlotsCount_)
    , Layout_(layout)
    , OriginalData_(nullptr) {
}


// -----------------------------------------------------------------
template <typename TTraits>
void TPageHashTableImpl<TTraits>::Build(const ui8* data, ui32 nItems) {
    OriginalData_ = data;
    TotalSlots_ = (7 * (nItems + 7)) / 3;
    TotalPages_ = (TotalSlots_ * SlotSize_ + PageSize_ - 1) / PageSize_ + 7;
    Y_ASSERT(TotalPages_ > 0);
    PageIndexOffset_ = 32 /* size of crc32 in bits */ - arrow::BitUtil::NumRequiredBits(TotalPages_ - 1);

    Data_.resize(TotalPages_ * PageSize_, 0);

    ui8 isEmbedded = static_cast<ui8>(TupleSize_ <= SlotSize_); // this flag used to dispatch where to store tuple
    static const void* embeddedDispatch[] = {&&inderected, &&embedded};

    for (ui32 i = 0; i < nItems; i++) {
        const ui8* tuple = OriginalData_ + i * TupleSize_;
        ui32 hash = ReadUnaligned<ui32>(tuple);
        ui64 pageNum = (hash >> PageIndexOffset_) % TotalPages_;

        ui64 pagePos = pageNum * PageSize_;
        ui8* pageAddr = Data_.data() + pagePos;
        ui8 filledSlotsCounter = ReadUnaligned<ui8>(pageAddr);

        while (filledSlotsCounter >= SlotsCount_) { // while page is completely filled
            pageAddr += PageSize_;

            if (pageAddr >= Data_.end()) { // cyclic buffer
                pageAddr = Data_.data();
            }

            filledSlotsCounter = ReadUnaligned<ui8>(pageAddr);
        }

        // found page with empty slot
        pageAddr[filledSlotsCounter + 1] = (hash >> PROBE_BYTE_INDEX) & 0xFF;
        pageAddr[0] = filledSlotsCounter + 1;

        goto *embeddedDispatch[isEmbedded]; // it is expected that goto will be faster than if statement in for-loop

    inderected:
        // put index to original buffer into slot
        WriteUnaligned<ui32>(pageAddr + PageHeaderSize_ + SlotSize_ * filledSlotsCounter, i);
        continue;

    embedded:
        // copy tuple from original buffer into slot
        std::memcpy(pageAddr + PageHeaderSize_ + SlotSize_ * filledSlotsCounter, tuple, TupleSize_);
        continue;
    }
}

template __attribute__((target("avx2"))) void
TPageHashTableImpl<NSimd::TSimdAVX2Traits>::Build(const ui8* data, ui32 nItems);

template __attribute__((target("sse4.2"))) void
TPageHashTableImpl<NSimd::TSimdSSE42Traits>::Build(const ui8* data, ui32 nItems);


// -----------------------------------------------------------------
template <typename TTraits>
void TPageHashTableImpl<TTraits>::Apply(ui32 hash, const ui8* key, OnMatchCallback* onMatch) {
    Y_ASSERT(OriginalData_ != nullptr); // Build was called before Apply
    using TSimd8 = typename TTraits:: template TSimd8<ui8>;

    static const void* keySizeDispatch[] = {&&keySize1, &&keySize2, &&keySize4, &&keySize8, &&bigKeySize};
    ui32 keySizeToCmp; // this variable used to dispatch how to load as compare keys
    switch (KeySize_) {
        case 1: {
            keySizeToCmp = 0;
            break;
        }
        case 2: {
            keySizeToCmp = 1;
            break;
        }
        case 4: {
            keySizeToCmp = 2;
            break;
        }
        case 8: {
            keySizeToCmp = 3;
            break;
        }
        default: {
            keySizeToCmp = 4;
            break;
        }
    }

    static const void* embeddedDispatch[] = {&&inderected, &&embedded};
    ui8 isEmbedded = static_cast<ui8>(TupleSize_ <= SlotSize_); // this flag used to dispatch where to store tuple

    ui64 pageNum = (hash >> PageIndexOffset_) % TotalPages_;
    ui64 pagePos = pageNum * PageSize_;
    ui8* pageAddr = Data_.data() + pagePos;
    ui8 filledSlotsCounter = ReadUnaligned<ui8>(pageAddr);

    while (true) { // Check all possible pages in collisions cycle
        TSimd8 headerReg = TSimd8(pageAddr + 1);                                          // [<Probe byte 1> ... <Probe byte K>]
        TSimd8 hashPartReg = TSimd8(static_cast<ui8>((hash >> PROBE_BYTE_INDEX) & 0xFF)); // [< Hash byte  > ... < Hash byte  >]
        ui32 foundBitMask = (headerReg == hashPartReg).ToBitMask(); // get all possible positions where desired tuples could be stored

        ui32 nextSetPos = 0;
        while (foundBitMask != 0) { // Check all possible matches in single page
            ui32 moveMask = foundBitMask & -foundBitMask; // find least significant not null bit
            nextSetPos = __builtin_ctzl(foundBitMask); // find position of least significant not null bit
            foundBitMask ^= moveMask; // remove least significant not null bit

            if (nextSetPos >= filledSlotsCounter) {
                break;
            }

            const ui8* tupleAddr;
            const ui8* keyAddr;
            goto *embeddedDispatch[isEmbedded]; // it is expected that goto will be faster than if statement in for-loop

            // get address of tuple if it is embedded or if it is lays in original buffer
            inderected: {
                ui32 index = ReadUnaligned<ui32>(pageAddr + PageHeaderSize_ + SlotSize_ * nextSetPos);
                tupleAddr = OriginalData_ + index * TupleSize_;
                goto embeddedDispatchEnd;
            }
            embedded: {
                tupleAddr = pageAddr + PageHeaderSize_ + SlotSize_ * nextSetPos;
            }

            embeddedDispatchEnd:;
            keyAddr = tupleAddr + KeyOffset_;
            goto *keySizeDispatch[keySizeToCmp]; // it is expected that goto will be faster than if statement in while-loop

            // compare keys
            keySize1: {
                if (ReadUnaligned<ui8>(keyAddr) == ReadUnaligned<ui8>(key)) {
                    onMatch(tupleAddr);
                }
                continue;
            }
            keySize2: {
                if (ReadUnaligned<ui16>(keyAddr) == ReadUnaligned<ui16>(key)) {
                    onMatch(tupleAddr);
                }
                continue;
            }
            keySize4: {
                if (ReadUnaligned<ui32>(keyAddr) == ReadUnaligned<ui32>(key)) {
                    onMatch(tupleAddr);
                }
                continue;
            }
            keySize8: {
                if (ReadUnaligned<ui64>(keyAddr) == ReadUnaligned<ui64>(key)) {
                    onMatch(tupleAddr);
                }
                continue;
            }
            // FIXME: WORKS ONLY WITH FIXED SIZE KEY COLUMNS
            bigKeySize: {
                if (std::equal(keyAddr, keyAddr + KeySize_, key)) {
                    onMatch(tupleAddr);
                }
                continue;
            }
        }

        if (filledSlotsCounter < SlotsCount_) { // Page has empty slots, i.e. there will definitely be no further occurrences
            break;
        }

        pageAddr += PageSize_;
        if (pageAddr >= Data_.end()) { // cyclic buffer
            pageAddr = Data_.data();
        }
    }
}

template __attribute__((target("avx2"))) void
TPageHashTableImpl<NSimd::TSimdAVX2Traits>::Apply(ui32 hash, const ui8* key, OnMatchCallback* onMatch);

template __attribute__((target("sse4.2"))) void
TPageHashTableImpl<NSimd::TSimdSSE42Traits>::Apply(ui32 hash, const ui8* key, OnMatchCallback* onMatch);


// -----------------------------------------------------------------
template <typename TTraits>
void TPageHashTableImpl<TTraits>::Clear() {
    OriginalData_ = nullptr;
    Data_.clear();
}


}
}
}

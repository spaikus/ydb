#include "mkql_computation_node_ut.h"

#include <ydb/library/yql/minikql/computation/mkql_block_reader.h>
#include <ydb/library/yql/minikql/computation/mkql_block_builder.h>
#include <ydb/library/yql/minikql/computation/mkql_block_impl.h>
#include <ydb/library/yql/minikql/computation/mkql_computation_node_holders.h>

#include <ydb/library/yql/minikql/arrow/arrow_defs.h>
#include <ydb/library/yql/minikql/arrow/arrow_util.h>
#include <ydb/library/yql/minikql/computation/mkql_computation_node_codegen.h>  // Y_IGNORE
#include <ydb/library/yql/minikql/mkql_node_builder.h>
#include <ydb/library/yql/minikql/mkql_node_cast.h>

#include <ydb/library/yql/parser/pg_wrapper/interface/arrow.h>

#include <ydb/library/yql/utils/simd/simd.h>

#include <arrow/scalar.h>
#include <arrow/array.h>
#include <arrow/datum.h>
#include <arrow/array/builder_primitive.h>

#include <chrono>
#include <numeric>

#include <ydb/library/yql/minikql/mkql_type_builder.h>
#include <ydb/library/yql/minikql/mkql_program_builder.cpp> // TODO: need for WideFromBlocks2, ValidateBlockFlowType

namespace NKikimr {
namespace NMiniKQL {

namespace {

size_t COUNT_OF_BLOCKS  = 1'000;
size_t BLOCK_SIZE       = 1'000; // For now 1K block_size is limit. TODO: Allocate more than one page in FromBlocks2 node

class TTestBlockFlowWrapper: public TStatefulWideFlowCodegeneratorNode<TTestBlockFlowWrapper> {
using TBaseComputation = TStatefulWideFlowCodegeneratorNode<TTestBlockFlowWrapper>;
using TArrowBlocks = std::vector<std::vector<std::shared_ptr<arrow::ArrayData>>>;

public:
    TTestBlockFlowWrapper(TComputationMutables& mutables, size_t blockSize, size_t blockCount, TArrowBlocks&& blocks)
        : TBaseComputation(mutables, nullptr, EValueRepresentation::Embedded)
        , BlockSize(blockSize)
        , BlockCount(blockCount)
        , Blocks(std::move(blocks))
    {
        mutables.CurValueIndex += 6U;
    }

    EFetchResult DoCalculate(NUdf::TUnboxedValue& state, TComputationContext& ctx, NUdf::TUnboxedValue*const* output) const {
        return DoCalculateImpl(state, ctx, *output[0], *output[1], *output[2], *output[3], *output[4], *output[5]);
    }

private:
    EFetchResult DoCalculateImpl(NUdf::TUnboxedValue& state, TComputationContext& ctx,
                                 /* 4 columns --> */ NUdf::TUnboxedValue& val1, NUdf::TUnboxedValue& val2, NUdf::TUnboxedValue& val3, NUdf::TUnboxedValue& val4,
                                 NUdf::TUnboxedValue& val5, NUdf::TUnboxedValue& val6) const {
        if (!state.HasValue()) {
            state = NUdf::TUnboxedValue::Zero();
        }

        auto index = state.Get<ui64>();
        if (index >= BlockCount) {
            return EFetchResult::Finish;
        }

        val1 = ctx.HolderFactory.CreateArrowBlock(std::move(Blocks[0][index]));
        val2 = ctx.HolderFactory.CreateArrowBlock(std::move(Blocks[1][index]));
        val3 = ctx.HolderFactory.CreateArrowBlock(std::move(Blocks[2][index]));
        val4 = ctx.HolderFactory.CreateArrowBlock(std::move(Blocks[3][index]));
        val5 = ctx.HolderFactory.CreateArrowBlock(arrow::Datum(std::make_shared<arrow::UInt64Scalar>(index)));
        val6 = ctx.HolderFactory.CreateArrowBlock(arrow::Datum(std::make_shared<arrow::UInt64Scalar>(BlockSize)));

        state = NUdf::TUnboxedValuePod(++index);
        return EFetchResult::One;
    }

    void RegisterDependencies() const final {
    }

    const size_t BlockSize;
    const size_t BlockCount;
    const TArrowBlocks Blocks;
};

IComputationNode* WrapTestBlockFlow(TCallable& callable, const TComputationNodeFactoryContext& ctx) {
    MKQL_ENSURE(callable.GetInputsCount() == 0, "Expected no args");
    std::vector<std::vector<std::shared_ptr<arrow::ArrayData>>> blocks(4);
    arrow::MemoryPool* pool = arrow::default_memory_pool(); // TComputationContext not allowed here, so ise default mempool

    for (auto& block: blocks)
    {
        block.resize(BLOCK_SIZE);
    }

    for (size_t pos = 0; pos < 4; ++pos)
    {
        for (size_t i = 0; i < COUNT_OF_BLOCKS; ++i)
        {
            arrow::UInt64Builder builder(pool);
            ARROW_OK(builder.Reserve(BLOCK_SIZE));
            for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                builder.UnsafeAppend(pos + 1);
            }
            ARROW_OK(builder.FinishInternal(&blocks[pos][i]));
        }
    }

    return new TTestBlockFlowWrapper(ctx.Mutables, BLOCK_SIZE, COUNT_OF_BLOCKS, std::move(blocks));
}

} // namespace

// -----------------------------------------------------------------------------------------------------------

namespace {

class TTestFromBlocks2Wrapper : public TStatefulWideFlowCodegeneratorNode<TTestFromBlocks2Wrapper> {
using TBaseComputation = TStatefulWideFlowCodegeneratorNode<TTestFromBlocks2Wrapper>;
public:
    TTestFromBlocks2Wrapper(TComputationMutables& mutables, IComputationWideFlowNode* flow, TVector<TType*>&& types)
        : TBaseComputation(mutables, flow, EValueRepresentation::Boxed)
        , Flow_(flow)
        , Types_(std::move(types))
        , WideFieldsIndex_(mutables.IncrementWideFieldsIndex(Types_.size() + 1U))
    {}

    EFetchResult DoCalculate(NUdf::TUnboxedValue& state, TComputationContext& ctx, NUdf::TUnboxedValue*const* output) const
    {
        auto& s = GetState(state, ctx);
        const auto fields = ctx.WideFields.data() + WideFieldsIndex_;
        if (s.Current_ == s.ColumnsLength_) do {
            if (const auto result = Flow_->FetchValues(ctx, fields); result != EFetchResult::One)
            {
                return result;
            }

            s.Current_ = 0;
            s.ColumnsLength_ = GetBlockCount(*fields[Types_.size()]);
            s.Transpose();
        } while (!s.ColumnsLength_);

        s.Get(output);
        s.Current_++;

        return EFetchResult::One;
    }

private:
    struct TState : public TComputationValue<TState> {
        size_t ColumnsLength_ = 0;
        size_t Current_ = 0;
        size_t ColumnsCount_ = 0;
        const TVector<TType*>& Types_;
        TUnboxedValueVector Values_;
        NKikimr::TAlignedPagePool& Pool_;
        i8* Buffer_ = nullptr;
        size_t Sizes_[4]; // TODO: Change 4 for some N

        TState(TMemoryUsageInfo* memInfo, const TVector<TType*>& types, TComputationContext& ctx)
            : TComputationValue(memInfo)
            , ColumnsCount_(types.size() - 1U)
            , Types_(types)
            , Values_(Types_.size() + 1U)
            , Pool_(ctx.HolderFactory.GetPagePool())
            , Buffer_(reinterpret_cast<i8*>(Pool_.GetBlock(NKikimr::TAlignedPagePool::POOL_PAGE_SIZE)))
        {
            Y_ASSERT(ColumnsCount_ == 4 && "More than 4 columns not supported yet");
        }

        ~TState() {
            Pool_.ReturnBlock(Buffer_, NKikimr::TAlignedPagePool::POOL_PAGE_SIZE);
        }

        void Transpose() {
            i8* data[4]; // TODO: Change 4 for some N

            for (size_t i = 0; i < ColumnsCount_; ++i) {
                auto array = TArrowBlock::From(Values_[i]).GetDatum().array();
                Sizes_[i] = CalcMaxBlockItemSize(Types_[i]);
                data[i] = const_cast<i8*>(array->template GetValues<const i8>(1));
            }

            // TODO: Change 4 for some N
            Y_ASSERT(std::accumulate(Sizes_, Sizes_ + 4, 0) * ColumnsLength_ <= NKikimr::TAlignedPagePool::POOL_PAGE_SIZE && "Total size of block should be less than 64KB");

            NSimd::Perfomancer performancer;
            auto worker = NSimd::ChooseTrait(performancer);
            worker->MergeColumns(Buffer_, data, Sizes_, ColumnsLength_);
        }

        void Get(NUdf::TUnboxedValue*const* output) const {
            auto total_row_size = std::accumulate(Sizes_, Sizes_ + 4, 0); // TODO: Change 4 for some N
            auto offset_buffer = Buffer_ + Current_ * total_row_size;

            for (size_t i = 0; i < ColumnsCount_; ++i) {
                if (const auto out = output[i]) {
                    auto dataType = static_cast<TDataType*>(Types_[i]);
                    switch (dataType->GetSchemeType()) {
                        case NUdf::TDataType<bool>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<bool*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<i8>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<i8*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<ui8>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<ui8*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<i16>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<i16*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<ui16>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<ui16*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<i32>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<i32*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<ui32>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<ui32*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<float>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<float*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<i64>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<i64*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<ui64>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<ui64*>(offset_buffer));
                            break;
                        }
                        case NUdf::TDataType<double>::Id:
                        {
                            *out = NUdf::TUnboxedValuePod(*reinterpret_cast<double*>(offset_buffer));
                            break;
                        }
                        default:
                        {
                            // TODO: Support all types conversion
                            Y_ASSERT(false && "Type not supported yet");
                            break;
                        }
                    }
                    offset_buffer += Sizes_[i];
                }
            }
        }
    };

    void RegisterDependencies() const final {
        FlowDependsOn(Flow_);
    }

    void MakeState(TComputationContext& ctx, NUdf::TUnboxedValue& state) const {
        state = ctx.HolderFactory.Create<TState>(Types_, ctx);
    }

    TState& GetState(NUdf::TUnboxedValue& state, TComputationContext& ctx) const {
        if (!state.HasValue()) {
            MakeState(ctx, state);

            const auto s = static_cast<TState*>(state.AsBoxed().Get());
            auto**const fields = ctx.WideFields.data() + WideFieldsIndex_;
            for (size_t i = 0; i <= Types_.size(); ++i) {
                fields[i] = &s->Values_[i];
            }
            return *s;
        }
        return *static_cast<TState*>(state.AsBoxed().Get());
    }

    IComputationWideFlowNode* const Flow_;
    const TVector<TType*> Types_;
    const size_t WideFieldsIndex_;
};

IComputationNode* WrapTestFromBlocks2(TCallable& callable, const TComputationNodeFactoryContext& ctx) {
    MKQL_ENSURE(callable.GetInputsCount() == 1, "Expected 1 args, got " << callable.GetInputsCount());
    const auto flowType = AS_TYPE(TFlowType, callable.GetInput(0).GetStaticType());
    const auto wideComponents = GetWideComponents(flowType);
    MKQL_ENSURE(wideComponents.size() > 0, "Expected at least one column");
    TVector<TType*> items;
    for (ui32 i = 0; i < wideComponents.size() - 1; ++i) {
        const auto blockType = AS_TYPE(TBlockType, wideComponents[i]);
        items.push_back(blockType->GetItemType());
    }

    const auto wideFlow = dynamic_cast<IComputationWideFlowNode*>(LocateNode(ctx.NodeLocator, callable, 0));
    MKQL_ENSURE(wideFlow != nullptr, "Expected wide flow node");
    return new TTestFromBlocks2Wrapper(ctx.Mutables, wideFlow, std::move(items));
}

} // namespace

// -----------------------------------------------------------------------------------------------------------

namespace {

TComputationNodeFactory GetNodeFactory() {
    return [](TCallable& callable, const TComputationNodeFactoryContext& ctx) -> IComputationNode* {
        if (callable.GetType()->GetName() == "TestBlockFlow") {
            return WrapTestBlockFlow(callable, ctx);
        }
        else if (callable.GetType()->GetName() == "TestFromBlocks2") {
            return WrapTestFromBlocks2(callable, ctx);
        }
        return GetBuiltinFactory()(callable, ctx);
    };

}

template<bool LLVM>
TRuntimeNode MakeFlow(TSetup<LLVM>& setup) {
    TProgramBuilder& pb = *setup.PgmBuilder;
    TCallableBuilder callableBuilder(*setup.Env, "TestBlockFlow",
                                     pb.NewFlowType(
                                         pb.NewMultiType({
                                             pb.NewBlockType(pb.NewDataType(NUdf::EDataSlot::Uint64), TBlockType::EShape::Many),
                                             pb.NewBlockType(pb.NewDataType(NUdf::EDataSlot::Uint64), TBlockType::EShape::Many),
                                             pb.NewBlockType(pb.NewDataType(NUdf::EDataSlot::Uint64), TBlockType::EShape::Many),
                                             pb.NewBlockType(pb.NewDataType(NUdf::EDataSlot::Uint64), TBlockType::EShape::Many),
                                             pb.NewBlockType(pb.NewDataType(NUdf::EDataSlot::Uint64), TBlockType::EShape::Scalar),
                                             pb.NewBlockType(pb.NewDataType(NUdf::EDataSlot::Uint64), TBlockType::EShape::Scalar),
                                             })));
    return TRuntimeNode(callableBuilder.Build(), false);
}

template<bool LLVM>
TRuntimeNode WideFromBlocks2(TRuntimeNode flow, TSetup<LLVM>& setup) {
    TProgramBuilder& pb = *setup.PgmBuilder;
    auto outputItems = ValidateBlockFlowType(flow.GetStaticType());
    outputItems.pop_back();
    TType* outputMultiType = pb.NewMultiType(outputItems);
    TCallableBuilder callableBuilder(*setup.Env, "TestFromBlocks2", pb.NewFlowType(outputMultiType));
    callableBuilder.Add(flow);
    return TRuntimeNode(callableBuilder.Build(), false);
}

} // namespace

// -----------------------------------------------------------------------------------------------------------

Y_UNIT_TEST_SUITE(TMiniKQLWideTakeSkipBlocks) {
    Y_UNIT_TEST_LLVM(TestSIMDFromBlocks) {
        auto simd_fromblocks_bench_run = [] () -> std::pair<double, double> /* prepare_time, exec_time */ {
            auto begin1 = std::chrono::steady_clock::now();
            TSetup<LLVM> setup(GetNodeFactory());
            TProgramBuilder& pb = *setup.PgmBuilder;

            const auto ui64Type  = pb.NewDataType(NUdf::TDataType<ui64>::Id);
            const auto tupleType = pb.NewTupleType({ui64Type, ui64Type, ui64Type, ui64Type});
            const auto flow      = MakeFlow(setup);
            const auto plain     = WideFromBlocks2(flow, setup); // <--- use new simd node

            const auto wideFlow = pb.NarrowMap(plain, [&](TRuntimeNode::TList items) -> TRuntimeNode {
                return pb.NewTuple(tupleType, {items[0], items[1], items[2], items[3]});
            });

            const auto pgmReturn = pb.ForwardList(wideFlow);
            const auto graph = setup.BuildGraph(pgmReturn);

            auto end1 = std::chrono::steady_clock::now();
            double prepare_time = std::chrono::duration<double>(end1 - begin1).count();

            auto begin2 = std::chrono::steady_clock::now();
            const auto iterator = graph->GetValue().GetListIterator();
            NUdf::TUnboxedValue item;
            while (iterator.Next(item))
            {
                auto first  = item.GetElement(0).Get<ui64>();
                auto second = item.GetElement(1).Get<ui64>();
                auto third  = item.GetElement(2).Get<ui64>();
                auto fourth = item.GetElement(3).Get<ui64>();

                UNIT_ASSERT(first == 1 && second == 2 && third == 3 && fourth == 4);
            }
            auto end2 = std::chrono::steady_clock::now();
            double calculation_time = std::chrono::duration<double>(end2 - begin2).count();

            return std::make_pair(prepare_time, calculation_time);
        };

        auto [warmup_prep_time, warmup_exec_time] = simd_fromblocks_bench_run(); // warmup run
        UNIT_ASSERT(!!warmup_prep_time && !!warmup_exec_time);
        auto [prep_time, exec_time] = simd_fromblocks_bench_run();
        Cerr << "\n------------------- SIMD --------------------\n";
        Cerr << "Count of uint64_t elements: " << COUNT_OF_BLOCKS * BLOCK_SIZE * 4 /* columns */ << Endl;
        Cerr << "Prepare stage time: " << prep_time * 1000.0 << "[ms]" << Endl;
        Cerr << "Calculation stage time: " << exec_time * 1000.0 << "[ms]" << Endl;
        Cerr << "Speed: " << static_cast<double>(COUNT_OF_BLOCKS * BLOCK_SIZE * 4 * 8) / (static_cast<double>(1'000'000'000) * exec_time) << "[Gb/sec]" << Endl;
        Cerr << "\n---------------------------------------------\n";
    }
    Y_UNIT_TEST_LLVM(TestOldFromBlocks) {
        auto old_fromblocks_bench_run = [] () -> std::pair<double, double> /* prepare_time, exec_time */ {
            auto begin1 = std::chrono::steady_clock::now();
            TSetup<LLVM> setup(GetNodeFactory());
            TProgramBuilder& pb = *setup.PgmBuilder;

            const auto ui64Type  = pb.NewDataType(NUdf::TDataType<ui64>::Id);
            const auto tupleType = pb.NewTupleType({ui64Type, ui64Type, ui64Type, ui64Type});
            const auto flow      = MakeFlow(setup);
            const auto plain     = pb.WideFromBlocks(flow); // <--- use old node

            const auto wideFlow = pb.NarrowMap(plain, [&](TRuntimeNode::TList items) -> TRuntimeNode {
                return pb.NewTuple(tupleType, {items[0], items[1], items[2], items[3]});
            });

            const auto pgmReturn = pb.ForwardList(wideFlow);
            const auto graph = setup.BuildGraph(pgmReturn);

            auto end1 = std::chrono::steady_clock::now();
            double prepare_time = std::chrono::duration<double>(end1 - begin1).count();

            auto begin2 = std::chrono::steady_clock::now();
            const auto iterator = graph->GetValue().GetListIterator();
            NUdf::TUnboxedValue item;
            while (iterator.Next(item))
            {
                auto first  = item.GetElement(0).Get<ui64>();
                auto second = item.GetElement(1).Get<ui64>();
                auto third  = item.GetElement(2).Get<ui64>();
                auto fourth = item.GetElement(3).Get<ui64>();

                UNIT_ASSERT(first == 1 && second == 2 && third == 3 && fourth == 4);
            }
            auto end2 = std::chrono::steady_clock::now();
            double calculation_time = std::chrono::duration<double>(end2 - begin2).count();

            return std::make_pair(prepare_time, calculation_time);
            Cerr << "\nCalculation stage time: " << calculation_time * 1000.0 << "[ms]" << Endl;
            Cerr << "Prepare stage time: " << prepare_time * 1000.0 << "[ms]" << Endl;
        };

        auto [warmup_prep_time, warmup_exec_time] = old_fromblocks_bench_run(); // warmup run
        UNIT_ASSERT(!!warmup_prep_time && !!warmup_exec_time);
        auto [prep_time, exec_time] = old_fromblocks_bench_run();
        Cerr << "\n------------------- OLD --------------------\n";
        Cerr << "Count of uint64_t elements: " << COUNT_OF_BLOCKS * BLOCK_SIZE * 4 /* columns */ << Endl;
        Cerr << "Prepare stage time: " << prep_time * 1000.0 << "[ms]" << Endl;
        Cerr << "Calculation stage time: " << exec_time * 1000.0 << "[ms]" << Endl;
        Cerr << "Speed: " << static_cast<double>(COUNT_OF_BLOCKS * BLOCK_SIZE * 4 * 8) / (static_cast<double>(1'000'000'000) * exec_time) << "[Gb/sec]" << Endl;
        Cerr << "\n--------------------------------------------\n";
    }
}

} // namespace NMiniKQL
} // namespace NKikimr



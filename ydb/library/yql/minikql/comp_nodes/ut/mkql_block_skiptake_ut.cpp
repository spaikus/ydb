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

class TTestBlockFlowWrapper: public TStatefulWideFlowCodegeneratorNode<TTestBlockFlowWrapper> {
using TBaseComputation = TStatefulWideFlowCodegeneratorNode<TTestBlockFlowWrapper>;

public:
    TTestBlockFlowWrapper(TComputationMutables& mutables, size_t blockSize, size_t blockCount)
        : TBaseComputation(mutables, nullptr, EValueRepresentation::Embedded)
        , BlockSize(blockSize)
        , BlockCount(blockCount)
    {
        mutables.CurValueIndex += 6U;
    }

    EFetchResult DoCalculate(NUdf::TUnboxedValue& state, TComputationContext& ctx, NUdf::TUnboxedValue*const* output) const {
        return DoCalculateImpl(state, ctx, *output[0], *output[1], *output[2], *output[3], *output[4], *output[5]);
    }
#ifndef MKQL_DISABLE_CODEGEN
    ICodegeneratorInlineWideNode::TGenerateResult DoGenGetValues(const TCodegenContext& ctx, Value* statePtr, BasicBlock*& block) const {
        auto& context = ctx.Codegen.GetContext();

        const auto valueType = Type::getInt128Ty(context);
        const auto ptrValueType = PointerType::getUnqual(valueType);
        const auto statusType = Type::getInt32Ty(context);

        const auto atTop = &ctx.Func->getEntryBlock().back();

        const auto values0Ptr = GetElementPtrInst::CreateInBounds(valueType, ctx.GetMutables(), {ConstantInt::get(Type::getInt32Ty(context), static_cast<const IComputationNode*>(this)->GetIndex() + 1U)}, "values_0_ptr", atTop);
        const auto values1Ptr = GetElementPtrInst::CreateInBounds(valueType, ctx.GetMutables(), {ConstantInt::get(Type::getInt32Ty(context), static_cast<const IComputationNode*>(this)->GetIndex() + 2U)}, "values_1_ptr", atTop);
        const auto values2Ptr = GetElementPtrInst::CreateInBounds(valueType, ctx.GetMutables(), {ConstantInt::get(Type::getInt32Ty(context), static_cast<const IComputationNode*>(this)->GetIndex() + 3U)}, "values_2_ptr", atTop);
        const auto values3Ptr = GetElementPtrInst::CreateInBounds(valueType, ctx.GetMutables(), {ConstantInt::get(Type::getInt32Ty(context), static_cast<const IComputationNode*>(this)->GetIndex() + 4U)}, "values_3_ptr", atTop);
        const auto values4Ptr = GetElementPtrInst::CreateInBounds(valueType, ctx.GetMutables(), {ConstantInt::get(Type::getInt32Ty(context), static_cast<const IComputationNode*>(this)->GetIndex() + 5U)}, "values_4_ptr", atTop);
        const auto values5Ptr = GetElementPtrInst::CreateInBounds(valueType, ctx.GetMutables(), {ConstantInt::get(Type::getInt32Ty(context), static_cast<const IComputationNode*>(this)->GetIndex() + 6U)}, "values_5_ptr", atTop);

        const auto ptrType = PointerType::getUnqual(StructType::get(context));
        const auto self = CastInst::Create(Instruction::IntToPtr, ConstantInt::get(Type::getInt64Ty(context), uintptr_t(this)), ptrType, "self", atTop);

        const auto doFunc = ConstantInt::get(Type::getInt64Ty(context), GetMethodPtr(&TTestBlockFlowWrapper::DoCalculateImpl));
        const auto doType = FunctionType::get(statusType, {self->getType(), ptrValueType,  ctx.Ctx->getType(), ptrValueType, ptrValueType, ptrValueType, ptrValueType, ptrValueType, ptrValueType}, false);
        const auto doFuncPtr = CastInst::Create(Instruction::IntToPtr, doFunc, PointerType::getUnqual(doType), "function", atTop);

        const auto result = CallInst::Create(doType, doFuncPtr, {self, statePtr, ctx.Ctx, values0Ptr, values1Ptr, values2Ptr, values3Ptr, values4Ptr, values5Ptr}, "result", block);

        ICodegeneratorInlineWideNode::TGettersList getters{
            [values0Ptr, valueType](const TCodegenContext&, BasicBlock*& block) { return new LoadInst(valueType, values0Ptr, "value", block); },
            [values1Ptr, valueType](const TCodegenContext&, BasicBlock*& block) { return new LoadInst(valueType, values1Ptr, "value", block); },
            [values2Ptr, valueType](const TCodegenContext&, BasicBlock*& block) { return new LoadInst(valueType, values2Ptr, "value", block); },
            [values3Ptr, valueType](const TCodegenContext&, BasicBlock*& block) { return new LoadInst(valueType, values3Ptr, "value", block); },
            [values4Ptr, valueType](const TCodegenContext&, BasicBlock*& block) { return new LoadInst(valueType, values4Ptr, "value", block); },
            [values5Ptr, valueType](const TCodegenContext&, BasicBlock*& block) { return new LoadInst(valueType, values5Ptr, "value", block); }
        };
        return {result, std::move(getters)};
    }
#endif
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

        std::shared_ptr<arrow::ArrayData> blocks[4]{nullptr};
        for (int pos = 0; pos < 4; ++pos)
        {
            arrow::UInt64Builder builder(&ctx.ArrowMemoryPool);
            ARROW_OK(builder.Reserve(BlockSize));
            for (size_t i = 0; i < BlockSize; ++i) {
                builder.UnsafeAppend(index * BlockSize + i);
            }
            ARROW_OK(builder.FinishInternal(&blocks[pos]));
        }

        val1 = ctx.HolderFactory.CreateArrowBlock(std::move(blocks[0]));
        val2 = ctx.HolderFactory.CreateArrowBlock(std::move(blocks[1]));
        val3 = ctx.HolderFactory.CreateArrowBlock(std::move(blocks[2]));
        val4 = ctx.HolderFactory.CreateArrowBlock(std::move(blocks[3]));
        val5 = ctx.HolderFactory.CreateArrowBlock(arrow::Datum(std::make_shared<arrow::UInt64Scalar>(index)));
        val6 = ctx.HolderFactory.CreateArrowBlock(arrow::Datum(std::make_shared<arrow::UInt64Scalar>(BlockSize)));

        state = NUdf::TUnboxedValuePod(++index);
        return EFetchResult::One;
    }

    void RegisterDependencies() const final {
    }

    const size_t BlockSize;
    const size_t BlockCount;
};

IComputationNode* WrapTestBlockFlow(TCallable& callable, const TComputationNodeFactoryContext& ctx) {
    MKQL_ENSURE(callable.GetInputsCount() == 0, "Expected no args");
    return new TTestBlockFlowWrapper(ctx.Mutables, 1'000, 1'000);
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
                return result;

            s.Current_ = 0;
            s.ColumnsLength_ = GetBlockCount(*fields[Types_.size()]);
            s.Transpose(fields);
        } while (!s.ColumnsLength_);

        s.Get(output);
        s.Current_++;

        return EFetchResult::One;
    }
#ifndef MKQL_DISABLE_CODEGEN
    ICodegeneratorInlineWideNode::TGenerateResult DoGenGetValues(const TCodegenContext& ctx, Value* statePtr, BasicBlock*& block) const {
        auto& context = ctx.Codegen.GetContext();

        const auto width = Types_.size();
        const auto valueType = Type::getInt128Ty(context);
        const auto ptrValueType = PointerType::getUnqual(valueType);
        const auto statusType = Type::getInt32Ty(context);
        const auto indexType = Type::getInt64Ty(context);
        const auto arrayType = ArrayType::get(valueType, width);
        const auto ptrValuesType = PointerType::getUnqual(ArrayType::get(valueType, width));

        TLLVMFieldsStructureState stateFields(context, width);
        const auto stateType = StructType::get(context, stateFields.GetFieldsArray());
        const auto statePtrType = PointerType::getUnqual(stateType);

        const auto getFunc = ConstantInt::get(Type::getInt64Ty(context), GetMethodPtr(&TState::Get));
        const auto getType = FunctionType::get(valueType, {statePtrType, ctx.GetFactory()->getType(), indexType}, false);
        const auto getPtr = CastInst::Create(Instruction::IntToPtr, getFunc, PointerType::getUnqual(getType), "get", &ctx.Func->getEntryBlock().back());
        const auto stateOnStack = new AllocaInst(statePtrType, 0U, "state_on_stack", &ctx.Func->getEntryBlock().back());
        new StoreInst(ConstantPointerNull::get(statePtrType), stateOnStack, &ctx.Func->getEntryBlock().back());

        const auto name = "GetBlockCount";
        ctx.Codegen.AddGlobalMapping(name, reinterpret_cast<const void*>(&GetBlockCount));
        const auto getCountType = NYql::NCodegen::ETarget::Windows != ctx.Codegen.GetEffectiveTarget() ?
            FunctionType::get(indexType, { valueType }, false):
            FunctionType::get(indexType, { ptrValueType }, false);
        const auto getCount = ctx.Codegen.GetModule().getOrInsertFunction(name, getCountType);

        const auto make = BasicBlock::Create(context, "make", ctx.Func);
        const auto main = BasicBlock::Create(context, "main", ctx.Func);
        const auto more = BasicBlock::Create(context, "more", ctx.Func);
        const auto good = BasicBlock::Create(context, "good", ctx.Func);
        const auto work = BasicBlock::Create(context, "work", ctx.Func);
        const auto over = BasicBlock::Create(context, "over", ctx.Func);

        BranchInst::Create(main, make, HasValue(statePtr, block), block);
        block = make;

        const auto ptrType = PointerType::getUnqual(StructType::get(context));
        const auto self = CastInst::Create(Instruction::IntToPtr, ConstantInt::get(Type::getInt64Ty(context), uintptr_t(this)), ptrType, "self", block);
        const auto makeFunc = ConstantInt::get(Type::getInt64Ty(context), GetMethodPtr(&TTestFromBlocks2Wrapper::MakeState));
        const auto makeType = FunctionType::get(Type::getVoidTy(context), {self->getType(), ctx.Ctx->getType(), statePtr->getType()}, false);
        const auto makeFuncPtr = CastInst::Create(Instruction::IntToPtr, makeFunc, PointerType::getUnqual(makeType), "function", block);
        CallInst::Create(makeType, makeFuncPtr, {self, ctx.Ctx, statePtr}, "", block);
        BranchInst::Create(main, block);

        block = main;

        const auto state = new LoadInst(valueType, statePtr, "state", block);
        const auto half = CastInst::Create(Instruction::Trunc, state, Type::getInt64Ty(context), "half", block);
        const auto stateArg = CastInst::Create(Instruction::IntToPtr, half, statePtrType, "state_arg", block);

        const auto countPtr = GetElementPtrInst::CreateInBounds(stateType, stateArg, { stateFields.This(), stateFields.GetCount() }, "count_ptr", block);
        const auto indexPtr = GetElementPtrInst::CreateInBounds(stateType, stateArg, { stateFields.This(), stateFields.GetIndex() }, "index_ptr", block);

        const auto count = new LoadInst(indexType, countPtr, "count", block);
        const auto index = new LoadInst(indexType, indexPtr, "index", block);

        const auto next = CmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_EQ, count, index, "next", block);

        BranchInst::Create(more, work, next, block);

        block = more;

        const auto clearFunc = ConstantInt::get(Type::getInt64Ty(context), GetMethodPtr(&TState::ClearValues));
        const auto clearType = FunctionType::get(Type::getVoidTy(context), {statePtrType}, false);
        const auto clearPtr = CastInst::Create(Instruction::IntToPtr, clearFunc, PointerType::getUnqual(clearType), "clear", block);
        CallInst::Create(clearType, clearPtr, {stateArg}, "", block);

        const auto getres = GetNodeValues(Flow_, ctx, block);

        const auto special = CmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_SLE, getres.first, ConstantInt::get(getres.first->getType(), static_cast<i32>(EFetchResult::Yield)), "special", block);

        const auto result = PHINode::Create(statusType, 2U, "result", over);
        result->addIncoming(getres.first, block);

        BranchInst::Create(over, good, special, block);

        block = good;

        const auto countValue = getres.second.back()(ctx, block);
        const auto height = CallInst::Create(getCount, { WrapArgumentForWindows(countValue, ctx, block) }, "height", block);

        new StoreInst(height, countPtr, block);
        new StoreInst(ConstantInt::get(indexType, 0), indexPtr, block);

        const auto empty = CmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_EQ, ConstantInt::get(indexType, 0), height, "empty", block);

        BranchInst::Create(more, work, empty, block);

        block = work;

        const auto current = new LoadInst(indexType, indexPtr, "current", block);
        const auto currentPtr = GetElementPtrInst::CreateInBounds(stateType, stateArg, { stateFields.This(), stateFields.GetCurrent() }, "current_ptr", block);
        new StoreInst(current, currentPtr, block);
        const auto increment = BinaryOperator::CreateAdd(current, ConstantInt::get(indexType, 1), "increment", block);
        new StoreInst(increment, indexPtr, block);
        new StoreInst(stateArg, stateOnStack, block);

        result->addIncoming(ConstantInt::get(statusType, static_cast<i32>(EFetchResult::One)), block);

        BranchInst::Create(over, block);

        block = over;

        ICodegeneratorInlineWideNode::TGettersList getters(width);
        for (size_t idx = 0U; idx < getters.size(); ++idx) {
            getters[idx] = [idx, width, getType, getPtr, indexType, arrayType, ptrValuesType, stateType, statePtrType, stateOnStack, getBlocks = getres.second](const TCodegenContext& ctx, BasicBlock*& block) {
                auto& context = ctx.Codegen.GetContext();
                const auto init = BasicBlock::Create(context, "init", ctx.Func);
                const auto call = BasicBlock::Create(context, "call", ctx.Func);

                TLLVMFieldsStructureState stateFields(context, width);

                const auto stateArg = new LoadInst(statePtrType, stateOnStack, "state", block);
                const auto valuesPtr = GetElementPtrInst::CreateInBounds(stateType, stateArg, { stateFields.This(), stateFields.GetPointer() }, "values_ptr", block);
                const auto values = new LoadInst(ptrValuesType, valuesPtr, "values", block);
                const auto index = ConstantInt::get(indexType, idx);
                const auto pointer = GetElementPtrInst::CreateInBounds(arrayType, values, {  ConstantInt::get(indexType, 0), index }, "pointer", block);

                BranchInst::Create(call, init, HasValue(pointer, block), block);

                block = init;

                const auto value = getBlocks[idx](ctx, block);
                new StoreInst(value, pointer, block);
                AddRefBoxed(value, ctx, block);

                BranchInst::Create(call, block);

                block = call;

                return CallInst::Create(getType, getPtr, {stateArg, ctx.GetFactory(), index}, "get", block);
            };
        }
        return {result, std::move(getters)};
    }
#endif

private:
    struct TState : public TComputationValue<TState> {
        size_t ColumnsLength_ = 0;
        size_t Current_ = 0;
        size_t ColumnsCount_ = 0;
        const TVector<TType*>& Types_;
        NKikimr::TAlignedPagePool& Pool_;
        i8* Buffer_ = nullptr;
        size_t Sizes_[4]; // TODO: Change 4 for some N

        TState(TMemoryUsageInfo* memInfo, const TVector<TType*>& types, TComputationContext& ctx)
            : TComputationValue(memInfo)
            , ColumnsCount_(types.size())
            , Types_(types)
            , Pool_(ctx.HolderFactory.GetPagePool())
            , Buffer_(reinterpret_cast<i8*>(Pool_.GetBlock(NKikimr::TAlignedPagePool::POOL_PAGE_SIZE)))
        {
            Y_ASSERT(ColumnsCount_ == 4 && "More than 4 columns not supported yet");
        }

        ~TState() {
            Pool_.ReturnBlock(Buffer_, NKikimr::TAlignedPagePool::POOL_PAGE_SIZE);
        }

        void Transpose(NUdf::TUnboxedValue*const* fields) {
            i8* data[4]; // TODO: Change 4 for some N

            for (size_t i = 0; i < ColumnsCount_; ++i) {
                auto array = TArrowBlock::From(*fields[i]).GetDatum().array();

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
#ifndef MKQL_DISABLE_CODEGEN
    class TLLVMFieldsStructureState: public TLLVMFieldsStructure<TComputationValue<TState>> {
    private:
        using TBase = TLLVMFieldsStructure<TComputationValue<TState>>;
        llvm::IntegerType*const CountType;
        llvm::IntegerType*const IndexType;
        llvm::IntegerType*const CurrentType;
        llvm::PointerType*const PointerType;
    protected:
        using TBase::Context;
    public:
        std::vector<llvm::Type*> GetFieldsArray() {
            std::vector<llvm::Type*> result = TBase::GetFields();
            result.emplace_back(CountType);
            result.emplace_back(IndexType);
            result.emplace_back(CurrentType);
            result.emplace_back(PointerType);
            return result;
        }

        llvm::Constant* GetCount() {
            return ConstantInt::get(Type::getInt32Ty(Context), TBase::GetFieldsCount() + 0);
        }

        llvm::Constant* GetIndex() {
            return ConstantInt::get(Type::getInt32Ty(Context), TBase::GetFieldsCount() + 1);
        }

        llvm::Constant* GetCurrent() {
            return ConstantInt::get(Type::getInt32Ty(Context), TBase::GetFieldsCount() + 2);
        }

        llvm::Constant* GetPointer() {
            return ConstantInt::get(Type::getInt32Ty(Context), TBase::GetFieldsCount() + 3);
        }

        TLLVMFieldsStructureState(llvm::LLVMContext& context, size_t width)
            : TBase(context)
            , CountType(Type::getInt64Ty(Context))
            , IndexType(Type::getInt64Ty(Context))
            , CurrentType(Type::getInt64Ty(Context))
            , PointerType(PointerType::getUnqual(ArrayType::get(Type::getInt128Ty(Context), width)))
        {}
    };
#endif

    void RegisterDependencies() const final {
        FlowDependsOn(Flow_);
    }

    void MakeState(TComputationContext& ctx, NUdf::TUnboxedValue& state) const {
        state = ctx.HolderFactory.Create<TState>(Types_, ctx);
    }

    TState& GetState(NUdf::TUnboxedValue& state, TComputationContext& ctx) const {
        if (!state.HasValue()) {
            MakeState(ctx, state);
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
    Y_UNIT_TEST_LLVM(TestWideSkipBlocks) {
        auto begin1 = std::chrono::steady_clock::now();
        TSetup<LLVM> setup(GetNodeFactory());
        TProgramBuilder& pb = *setup.PgmBuilder;

        const auto flow = MakeFlow(setup);

        // const auto part = pb.WideSkipBlocks(flow, pb.NewDataLiteral<ui64>(7));
        const auto plain = WideFromBlocks2(flow, setup);

        const auto singleValueFlow = pb.NarrowMap(plain, [&](TRuntimeNode::TList items) -> TRuntimeNode {
            return items[0];
        });

        const auto pgmReturn = pb.ForwardList(singleValueFlow);

        const auto graph = setup.BuildGraph(pgmReturn);
        auto end1 = std::chrono::steady_clock::now();

        auto begin2 = std::chrono::steady_clock::now();
        double prepare_time = std::chrono::duration<double>(end1 - begin1).count();
        Cerr << "Prepare stage time: " << prepare_time * 1000.0 << Endl;

        const auto iterator = graph->GetValue().GetListIterator();
        ui64 result{0};
        ui64 counter{0};
        NUdf::TUnboxedValue item;
        while (iterator.Next(item))
        {
            ++counter;
            result += item.Get<ui64>();
        }
        auto end2 = std::chrono::steady_clock::now();
        double calculation_time = std::chrono::duration<double>(end2 - begin2).count();
        Cerr << "Calculation stage time: " << calculation_time * 1000.0 << Endl;
        Cerr << "Counter: " << counter << Endl;

        UNIT_ASSERT(!!result);
    }

    // Y_UNIT_TEST_LLVM(TestWideTakeBlocks) {
    //     TSetup<LLVM> setup(GetNodeFactory());
    //     TProgramBuilder& pb = *setup.PgmBuilder;

    //     const auto flow = MakeFlow(setup);

    //     const auto part = pb.WideTakeBlocks(flow, pb.NewDataLiteral<ui64>(4));
    //     const auto plain = pb.WideFromBlocks(part);

    //     const auto singleValueFlow = pb.NarrowMap(plain, [&](TRuntimeNode::TList items) -> TRuntimeNode {
    //         return pb.Add(items[0], items[1]);
    //     });

    //     const auto pgmReturn = pb.ForwardList(singleValueFlow);

    //     const auto graph = setup.BuildGraph(pgmReturn);
    //     const auto iterator = graph->GetValue().GetListIterator();

    //     NUdf::TUnboxedValue item;
    //     UNIT_ASSERT(iterator.Next(item));
    //     UNIT_ASSERT_VALUES_EQUAL(item.Get<ui64>(), 0);

    //     UNIT_ASSERT(iterator.Next(item));
    //     UNIT_ASSERT_VALUES_EQUAL(item.Get<ui64>(), 1);

    //     UNIT_ASSERT(iterator.Next(item));
    //     UNIT_ASSERT_VALUES_EQUAL(item.Get<ui64>(), 2);

    //     UNIT_ASSERT(iterator.Next(item));
    //     UNIT_ASSERT_VALUES_EQUAL(item.Get<ui64>(), 3);

    //     UNIT_ASSERT(!iterator.Next(item));
    //     UNIT_ASSERT(!iterator.Next(item));
    // }

    // Y_UNIT_TEST_LLVM(TestWideTakeSkipBlocks) {
    //     TSetup<LLVM> setup(GetNodeFactory());
    //     TProgramBuilder& pb = *setup.PgmBuilder;

    //     const auto flow = MakeFlow(setup);

    //     const auto part = pb.WideTakeBlocks(pb.WideSkipBlocks(flow, pb.NewDataLiteral<ui64>(3)), pb.NewDataLiteral<ui64>(5));
    //     const auto plain = pb.WideFromBlocks(part);

    //     const auto singleValueFlow = pb.NarrowMap(plain, [&](TRuntimeNode::TList items) -> TRuntimeNode {
    //         // 0,  0;
    //         // 1,  0;
    //         // 2,  0;
    //         // 3,  0; -> 3
    //         // 4,  0; -> 4
    //         // 5,  1; -> 6
    //         // 6,  1; -> 7
    //         // 7,  1; -> 8
    //         // 8,  1;
    //         // 9,  1;
    //         // 10, 1;
    //         return pb.Add(items[0], items[1]);
    //     });

    //     const auto pgmReturn = pb.ForwardList(singleValueFlow);

    //     const auto graph = setup.BuildGraph(pgmReturn);
    //     const auto iterator = graph->GetValue().GetListIterator();

    //     NUdf::TUnboxedValue item;
    //     UNIT_ASSERT(iterator.Next(item));
    //     UNIT_ASSERT_VALUES_EQUAL(item.Get<ui64>(), 3);

    //     UNIT_ASSERT(iterator.Next(item));
    //     UNIT_ASSERT_VALUES_EQUAL(item.Get<ui64>(), 4);

    //     UNIT_ASSERT(iterator.Next(item));
    //     UNIT_ASSERT_VALUES_EQUAL(item.Get<ui64>(), 6);

    //     UNIT_ASSERT(iterator.Next(item));
    //     UNIT_ASSERT_VALUES_EQUAL(item.Get<ui64>(), 7);

    //     UNIT_ASSERT(iterator.Next(item));
    //     UNIT_ASSERT_VALUES_EQUAL(item.Get<ui64>(), 8);

    //     UNIT_ASSERT(!iterator.Next(item));
    //     UNIT_ASSERT(!iterator.Next(item));
    // }
}

} // namespace NMiniKQL
} // namespace NKikimr



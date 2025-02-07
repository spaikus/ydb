#include "mkql_computation_node_ut.h"
#include <yql/essentials/minikql/mkql_runtime_version.h>
#include <yql/essentials/minikql/comp_nodes/mkql_grace_join_imp.h>

#include <yql/essentials/minikql/computation/mock_spiller_factory_ut.h>

#include <chrono>
#include <iostream>
#include <cstring>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <stdlib.h>
#include <random>

#include <util/system/compiler.h>
#include <util/stream/null.h>
#include <util/system/mem_info.h>

#include <cstdint>

namespace NKikimr {
namespace NMiniKQL {

constexpr bool IsVerbose = true;
#define CTEST (IsVerbose ? Cerr : Cnull)


void generate_unique_data(TProgramBuilder& pb, std::vector<TRuntimeNode>& data, ui64 start, ui64 end) {
    for (ui64 i = start; i < end; ++i) {
        const auto key = pb.NewDataLiteral<ui64>(i);
        const auto payload = pb.NewDataLiteral<ui64>(i);

        data.push_back(pb.NewTuple({key, payload}));
    }
}

void generate_same_data(TProgramBuilder& pb, std::vector<TRuntimeNode>& data, ui64 value, ui64 count) {
    for (ui64 i = 0; i < count; ++i) {
        const auto key = pb.NewDataLiteral<ui64>(value);
        const auto payload = pb.NewDataLiteral<ui64>(value);

        data.push_back(pb.NewTuple({key, payload}));
    }
}

void generate_complex_unique_data(TProgramBuilder& pb, std::vector<TRuntimeNode>& data, ui64 start, ui64 end) {
    const auto longStr = std::string(64, 'a');
    for (ui64 i = start; i < end; ++i) {
        const auto key = pb.NewDataLiteral<ui64>(i);
        const auto keyStr = pb.NewDataLiteral<NUdf::EDataSlot::String>(longStr);
        const auto payloadStr = pb.NewDataLiteral<NUdf::EDataSlot::String>(longStr);

        data.push_back(pb.NewTuple({key, keyStr, payloadStr}));
    }
}

Y_UNIT_TEST_SUITE(TMiniKQLGraceJoinTest) {

    Y_UNIT_TEST_LLVM_SPILLING(Test100PercentSelectivityJoin) {
        if (SPILLING && RuntimeVersion < 50) return;

        std::vector<ui64> leftSizes  = {5'000, 10'000, 20'000, 50'000};
        std::vector<ui64> rightSizes = {25'000, 50'000, 100'000, 250'000};
        const auto tupleSize = sizeof(ui64) + sizeof(ui64);

        CTEST << "----------------------------------------" << Endl;
        CTEST << "TEST NAME: " << "Test100PercentSelectivityJoin" << Endl;
        CTEST << "----------------------------------------" << Endl;
        for (ui32 pass = 0; pass < leftSizes.size(); ++pass) {
            TSetup<LLVM, SPILLING> setup;
            TProgramBuilder& pb = *setup.PgmBuilder;

            const auto lhsSize = leftSizes[pass];
            const auto rhsSize = rightSizes[pass];
            std::vector<TRuntimeNode> lhs, rhs;
            generate_unique_data(pb, lhs, 0, lhsSize);
            generate_unique_data(pb, rhs, 0, rhsSize);

            const auto tupleType = pb.NewTupleType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            });

            const auto list1 = pb.NewList(tupleType, std::move(lhs));
            const auto list2 = pb.NewList(tupleType, std::move(rhs));

            const auto resultType = pb.NewFlowType(pb.NewMultiType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            }));

            const auto pgmReturn = pb.Collect(pb.NarrowMap(pb.GraceJoin(
                pb.ExpandMap(pb.ToFlow(list1), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                pb.ExpandMap(pb.ToFlow(list2), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                EJoinKind::Inner, {0U}, {0U}, {1U, 0U}, {1U, 1U}, resultType),
                [&](TRuntimeNode::TList items) -> TRuntimeNode { return pb.NewTuple(items); })
            );
            if (SPILLING) {
                setup.RenameCallable(pgmReturn, "GraceJoin", "GraceJoinWithSpilling");
            }

            const auto graph = setup.BuildGraph(pgmReturn);
            if (SPILLING) {
                graph->GetContext().SpillerFactory = std::make_shared<TMockSpillerFactory>();
            }

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            const auto iterator = graph->GetValue().GetListIterator();
            NUdf::TUnboxedValue tuple;
            std::map<std::pair<ui64, ui64>, ui32> u;

            while (iterator.Next(tuple)) {
                auto t0 = tuple.GetElement(0);
                auto t1 = tuple.GetElement(1);
                ++u[std::make_pair(t0.Get<ui64>(), t1.Get<ui64>())];
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            ui64 millisecondsJoin = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            CTEST << "Configuration: " << "Count of lhs elements = " << lhsSize << ", Count of rhs elements = " << rhsSize << ", disrt: unique = [0..size]" << Endl;
            CTEST << "Pure data size = " << tupleSize * (lhsSize + rhsSize) / 1024 << "[KB]" << Endl;
            CTEST << "Time for join = " << millisecondsJoin << "[ms]" << Endl;
            CTEST << "Join tuples speed: " << (tupleSize * (lhsSize + rhsSize) * 1000) / (millisecondsJoin * 1024 * 1024 + 1) << "MB/sec" << Endl;
            CTEST << "----------------------------------------" << Endl;
            CTEST << Endl;

            UNIT_ASSERT(!iterator.Next(tuple));
            UNIT_ASSERT_EQUAL(u.size(), lhsSize);
        }
    }

    Y_UNIT_TEST_LLVM_SPILLING(Test50PercentSelectivityJoin) {
        if (SPILLING && RuntimeVersion < 50) return;

        std::vector<ui64> leftSizes  = {5'000, 10'000, 20'000, 50'000};
        std::vector<ui64> rightSizes = {25'000, 50'000, 100'000, 250'000};
        const auto tupleSize = sizeof(ui64) + sizeof(ui64);

        CTEST << "----------------------------------------" << Endl;
        CTEST << "TEST NAME: " << "Test50PercentSelectivityJoin" << Endl;
        CTEST << "----------------------------------------" << Endl;
        for (ui32 pass = 0; pass < leftSizes.size(); ++pass) {
            TSetup<LLVM, SPILLING> setup;
            TProgramBuilder& pb = *setup.PgmBuilder;

            const auto lhsSize = leftSizes[pass];
            const auto rhsSize = rightSizes[pass];
            std::vector<TRuntimeNode> lhs, rhs;
            generate_unique_data(pb, lhs, 0, lhsSize);
            generate_unique_data(pb, rhs, lhsSize / 2, rhsSize + lhsSize / 2 + 1);

            const auto tupleType = pb.NewTupleType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            });

            const auto list1 = pb.NewList(tupleType, std::move(lhs));
            const auto list2 = pb.NewList(tupleType, std::move(rhs));

            const auto resultType = pb.NewFlowType(pb.NewMultiType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            }));

            const auto pgmReturn = pb.Collect(pb.NarrowMap(pb.GraceJoin(
                pb.ExpandMap(pb.ToFlow(list1), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                pb.ExpandMap(pb.ToFlow(list2), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                EJoinKind::Inner, {0U}, {0U}, {1U, 0U}, {1U, 1U}, resultType),
                [&](TRuntimeNode::TList items) -> TRuntimeNode { return pb.NewTuple(items); })
            );
            if (SPILLING) {
                setup.RenameCallable(pgmReturn, "GraceJoin", "GraceJoinWithSpilling");
            }

            const auto graph = setup.BuildGraph(pgmReturn);
            if (SPILLING) {
                graph->GetContext().SpillerFactory = std::make_shared<TMockSpillerFactory>();
            }

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            const auto iterator = graph->GetValue().GetListIterator();
            NUdf::TUnboxedValue tuple;
            std::map<std::pair<ui64, ui64>, ui32> u;

            while (iterator.Next(tuple)) {
                auto t0 = tuple.GetElement(0);
                auto t1 = tuple.GetElement(1);
                ++u[std::make_pair(t0.Get<ui64>(), t1.Get<ui64>())];
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            ui64 millisecondsJoin = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            CTEST << "Configuration: " << "Count of lhs elements = " << lhsSize << ", Count of rhs elements = " << rhsSize << ", disrt: unique = [0..size]" << Endl;
            CTEST << "Pure data size = " << tupleSize * (lhsSize + rhsSize) / 1024 << "[KB]" << Endl;
            CTEST << "Time for join = " << millisecondsJoin << "[ms]" << Endl;
            CTEST << "Join tuples speed: " << (tupleSize * (lhsSize + rhsSize) * 1000) / (millisecondsJoin * 1024 * 1024 + 1) << "MB/sec" << Endl;
            CTEST << "----------------------------------------" << Endl;
            CTEST << Endl;

            UNIT_ASSERT(!iterator.Next(tuple));
            UNIT_ASSERT_EQUAL(u.size(), lhsSize / 2);
        }
    }

    Y_UNIT_TEST_LLVM_SPILLING(Test20PercentSelectivityJoin) {
        if (SPILLING && RuntimeVersion < 50) return;

        std::vector<ui64> leftSizes  = {5'000, 10'000, 20'000, 50'000};
        std::vector<ui64> rightSizes = {25'000, 50'000, 100'000, 250'000};
        const auto tupleSize = sizeof(ui64) + sizeof(ui64);

        CTEST << Endl;
        CTEST << "----------------------------------------" << Endl;
        CTEST << "TEST NAME: " << "Test20PercentSelectivityJoin" << Endl;
        CTEST << "----------------------------------------" << Endl;
        for (ui32 pass = 0; pass < leftSizes.size(); ++pass) {
            TSetup<LLVM, SPILLING> setup;
            TProgramBuilder& pb = *setup.PgmBuilder;

            const auto lhsSize = leftSizes[pass];
            const auto rhsSize = rightSizes[pass];
            std::vector<TRuntimeNode> lhs, rhs;
            generate_unique_data(pb, lhs, 0, lhsSize);
            generate_unique_data(pb, rhs, lhsSize * 0.8, rhsSize + lhsSize * 0.8 + 1);

            const auto tupleType = pb.NewTupleType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            });

            const auto list1 = pb.NewList(tupleType, std::move(lhs));
            const auto list2 = pb.NewList(tupleType, std::move(rhs));

            const auto resultType = pb.NewFlowType(pb.NewMultiType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            }));

            const auto pgmReturn = pb.Collect(pb.NarrowMap(pb.GraceJoin(
                pb.ExpandMap(pb.ToFlow(list1), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                pb.ExpandMap(pb.ToFlow(list2), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                EJoinKind::Inner, {0U}, {0U}, {1U, 0U}, {1U, 1U}, resultType),
                [&](TRuntimeNode::TList items) -> TRuntimeNode { return pb.NewTuple(items); })
            );
            if (SPILLING) {
                setup.RenameCallable(pgmReturn, "GraceJoin", "GraceJoinWithSpilling");
            }

            const auto graph = setup.BuildGraph(pgmReturn);
            if (SPILLING) {
                graph->GetContext().SpillerFactory = std::make_shared<TMockSpillerFactory>();
            }

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            const auto iterator = graph->GetValue().GetListIterator();
            NUdf::TUnboxedValue tuple;
            std::map<std::pair<ui64, ui64>, ui32> u;

            while (iterator.Next(tuple)) {
                auto t0 = tuple.GetElement(0);
                auto t1 = tuple.GetElement(1);
                ++u[std::make_pair(t0.Get<ui64>(), t1.Get<ui64>())];
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            ui64 millisecondsJoin = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            CTEST << "Configuration: " << "Count of lhs elements = " << lhsSize << ", Count of rhs elements = " << rhsSize << ", disrt: unique = [0..size]" << Endl;
            CTEST << "Pure data size = " << tupleSize * (lhsSize + rhsSize) / 1024 << "[KB]" << Endl;
            CTEST << "Time for join = " << millisecondsJoin << "[ms]" << Endl;
            CTEST << "Join tuples speed: " << (tupleSize * (lhsSize + rhsSize) * 1000) / (millisecondsJoin * 1024 * 1024 + 1) << "MB/sec" << Endl;
            CTEST << "----------------------------------------" << Endl;
            CTEST << Endl;

            UNIT_ASSERT(!iterator.Next(tuple));
            UNIT_ASSERT_EQUAL(u.size(), lhsSize * 0.2);
        }
    }

    Y_UNIT_TEST_LLVM_SPILLING(LeftSkewedJoin) {
        if (SPILLING && RuntimeVersion < 50) return;

        std::vector<ui64> leftSizes  = {2'500, 5'000, 10'000, 25'000};
        std::vector<ui64> rightSizes = {25'000, 50'000, 100'000, 250'000};
        const auto tupleSize = sizeof(ui64) + sizeof(ui64);

        CTEST << "----------------------------------------" << Endl;
        CTEST << "TEST NAME: " << "LeftSkewedJoin" << Endl;
        CTEST << "----------------------------------------" << Endl;
        for (ui32 pass = 0; pass < leftSizes.size(); ++pass) {
            TSetup<LLVM, SPILLING> setup;
            TProgramBuilder& pb = *setup.PgmBuilder;

            const auto lhsSize = leftSizes[pass];
            const auto rhsSize = rightSizes[pass];
            std::vector<TRuntimeNode> lhs, rhs;
            generate_unique_data(pb, lhs, 0, lhsSize);
            generate_same_data(pb, lhs, 0, lhsSize);
            generate_unique_data(pb, rhs, 0, rhsSize);

            const auto tupleType = pb.NewTupleType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            });

            const auto list1 = pb.NewList(tupleType, std::move(lhs));
            const auto list2 = pb.NewList(tupleType, std::move(rhs));

            const auto resultType = pb.NewFlowType(pb.NewMultiType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            }));

            const auto pgmReturn = pb.Collect(pb.NarrowMap(pb.GraceJoin(
                pb.ExpandMap(pb.ToFlow(list1), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                pb.ExpandMap(pb.ToFlow(list2), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                EJoinKind::Inner, {0U}, {0U}, {1U, 0U}, {1U, 1U}, resultType),
                [&](TRuntimeNode::TList items) -> TRuntimeNode { return pb.NewTuple(items); })
            );
            if (SPILLING) {
                setup.RenameCallable(pgmReturn, "GraceJoin", "GraceJoinWithSpilling");
            }

            const auto graph = setup.BuildGraph(pgmReturn);
            if (SPILLING) {
                graph->GetContext().SpillerFactory = std::make_shared<TMockSpillerFactory>();
            }

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            const auto iterator = graph->GetValue().GetListIterator();
            NUdf::TUnboxedValue tuple;
            std::map<std::pair<ui64, ui64>, ui32> u;

            while (iterator.Next(tuple)) {
                auto t0 = tuple.GetElement(0);
                auto t1 = tuple.GetElement(1);
                ++u[std::make_pair(t0.Get<ui64>(), t1.Get<ui64>())];
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            ui64 millisecondsJoin = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            CTEST << "Configuration: " << "Count of lhs elements = " << lhsSize * 2 << ", Count of rhs elements = " << rhsSize << ", disrt: unique = [0..size] + [0] * size" << Endl;
            CTEST << "Pure data size = " << tupleSize * (lhsSize * 2 + rhsSize) / 1024 << "[KB]" << Endl;
            CTEST << "Time for join = " << millisecondsJoin << "[ms]" << Endl;
            CTEST << "Join tuples speed: " << (tupleSize * (lhsSize * 2 + rhsSize) * 1000) / (millisecondsJoin * 1024 * 1024 + 1) << "MB/sec" << Endl;
            CTEST << "----------------------------------------" << Endl;
            CTEST << Endl;

            UNIT_ASSERT(!iterator.Next(tuple));
            UNIT_ASSERT_EQUAL(u.size(), lhsSize);
        }
    }

    Y_UNIT_TEST_LLVM_SPILLING(RightSkewedJoin) {
        if (SPILLING && RuntimeVersion < 50) return;

        std::vector<ui64> leftSizes  = {5'000, 10'000, 20'000, 50'000};
        std::vector<ui64> rightSizes = {12'500, 25'000, 50'000, 125'000};
        const auto tupleSize = sizeof(ui64) + sizeof(ui64);

        CTEST << "----------------------------------------" << Endl;
        CTEST << "TEST NAME: " << "RightSkewedJoin" << Endl;
        CTEST << "----------------------------------------" << Endl;
        for (ui32 pass = 0; pass < leftSizes.size(); ++pass) {
            TSetup<LLVM, SPILLING> setup;
            TProgramBuilder& pb = *setup.PgmBuilder;

            const auto lhsSize = leftSizes[pass];
            const auto rhsSize = rightSizes[pass];
            std::vector<TRuntimeNode> lhs, rhs;
            generate_unique_data(pb, lhs, 0, lhsSize);
            generate_unique_data(pb, rhs, 0, rhsSize);
            generate_same_data(pb, rhs, 0, rhsSize);

            const auto tupleType = pb.NewTupleType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            });

            const auto list1 = pb.NewList(tupleType, std::move(lhs));
            const auto list2 = pb.NewList(tupleType, std::move(rhs));

            const auto resultType = pb.NewFlowType(pb.NewMultiType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            }));

            const auto pgmReturn = pb.Collect(pb.NarrowMap(pb.GraceJoin(
                pb.ExpandMap(pb.ToFlow(list1), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                pb.ExpandMap(pb.ToFlow(list2), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                EJoinKind::Inner, {0U}, {0U}, {1U, 0U}, {1U, 1U}, resultType),
                [&](TRuntimeNode::TList items) -> TRuntimeNode { return pb.NewTuple(items); })
            );
            if (SPILLING) {
                setup.RenameCallable(pgmReturn, "GraceJoin", "GraceJoinWithSpilling");
            }

            const auto graph = setup.BuildGraph(pgmReturn);
            if (SPILLING) {
                graph->GetContext().SpillerFactory = std::make_shared<TMockSpillerFactory>();
            }

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            const auto iterator = graph->GetValue().GetListIterator();
            NUdf::TUnboxedValue tuple;
            std::map<std::pair<ui64, ui64>, ui32> u;

            ui64 total = 0;
            while (iterator.Next(tuple)) {
                auto t0 = tuple.GetElement(0);
                auto t1 = tuple.GetElement(1);
                ++u[std::make_pair(t0.Get<ui64>(), t1.Get<ui64>())];
                ++total;
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            ui64 millisecondsJoin = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            CTEST << "Configuration: " << "Count of lhs elements = " << lhsSize << ", Count of rhs elements = " << rhsSize * 2 << ", disrt: unique = [0..size] + [0] * size" << Endl;
            CTEST << "Pure data size = " << tupleSize * (lhsSize + rhsSize * 2) / 1024 << "[KB]" << Endl;
            CTEST << "Time for join = " << millisecondsJoin << "[ms]" << Endl;
            CTEST << "Join tuples speed: " << (tupleSize * (lhsSize + rhsSize * 2) * 1000) / (millisecondsJoin * 1024 * 1024 + 1) << "MB/sec" << Endl;
            CTEST << "----------------------------------------" << Endl;
            CTEST << Endl;

            UNIT_ASSERT(!iterator.Next(tuple));
            UNIT_ASSERT_EQUAL(u.size(), lhsSize);
            UNIT_ASSERT_EQUAL(total, lhsSize + rhsSize);
        }
    }

    Y_UNIT_TEST_LLVM_SPILLING(CrossJoin) {
        if (SPILLING && RuntimeVersion < 50) return;

        std::vector<ui64> leftSizes  = {1'00, 2'00, 4'00};
        std::vector<ui64> rightSizes = {10'000, 20'000, 40'000};
        const auto tupleSize = sizeof(ui64) + sizeof(ui64);

        CTEST << "----------------------------------------" << Endl;
        CTEST << "TEST NAME: " << "CrossJoin" << Endl;
        CTEST << "----------------------------------------" << Endl;
        for (ui32 pass = 0; pass < leftSizes.size(); ++pass) {
            TSetup<LLVM, SPILLING> setup;
            TProgramBuilder& pb = *setup.PgmBuilder;

            const auto lhsSize = leftSizes[pass];
            const auto rhsSize = rightSizes[pass];
            std::vector<TRuntimeNode> lhs, rhs;
            generate_same_data(pb, lhs, 0, lhsSize);
            generate_same_data(pb, rhs, 0, rhsSize);

            const auto tupleType = pb.NewTupleType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            });

            const auto list1 = pb.NewList(tupleType, std::move(lhs));
            const auto list2 = pb.NewList(tupleType, std::move(rhs));

            const auto resultType = pb.NewFlowType(pb.NewMultiType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<ui64>::Id)
            }));

            const auto pgmReturn = pb.Collect(pb.NarrowMap(pb.GraceJoin(
                pb.ExpandMap(pb.ToFlow(list1), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                pb.ExpandMap(pb.ToFlow(list2), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U)}; }),
                EJoinKind::Inner, {0U}, {0U}, {1U, 0U}, {1U, 1U}, resultType),
                [&](TRuntimeNode::TList items) -> TRuntimeNode { return pb.NewTuple(items); })
            );
            if (SPILLING) {
                setup.RenameCallable(pgmReturn, "GraceJoin", "GraceJoinWithSpilling");
            }

            const auto graph = setup.BuildGraph(pgmReturn);
            if (SPILLING) {
                graph->GetContext().SpillerFactory = std::make_shared<TMockSpillerFactory>();
            }

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            const auto iterator = graph->GetValue().GetListIterator();
            NUdf::TUnboxedValue tuple;
            std::map<std::pair<ui64, ui64>, ui32> u;

            ui64 total = 0;
            while (iterator.Next(tuple)) {
                auto t0 = tuple.GetElement(0);
                auto t1 = tuple.GetElement(1);
                ++u[std::make_pair(t0.Get<ui64>(), t1.Get<ui64>())];
                ++total;
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            ui64 millisecondsJoin = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            CTEST << "Configuration: " << "Count of lhs elements = " << lhsSize << ", Count of rhs elements = " << rhsSize << ", disrt: unique = [0..size]" << Endl;
            CTEST << "Pure data size = " << tupleSize * (lhsSize + rhsSize) / 1024 << "[KB]" << Endl;
            CTEST << "Time for join = " << millisecondsJoin << "[ms]" << Endl;
            CTEST << "Join tuples speed: " << (tupleSize * (lhsSize + rhsSize) * 1000) / (millisecondsJoin * 1024 + 1) << "KB/sec" << Endl;
            CTEST << "----------------------------------------" << Endl;
            CTEST << Endl;

            UNIT_ASSERT(!iterator.Next(tuple));
            UNIT_ASSERT_EQUAL(u.size(), 1);
            UNIT_ASSERT_EQUAL(total, lhsSize * rhsSize);
        }
    }

    Y_UNIT_TEST_LLVM_SPILLING(WideTupleJoin) {
        if (SPILLING && RuntimeVersion < 50) return;

        std::vector<ui64> leftSizes  = {5'000, 10'000, 20'000, 50'000};
        std::vector<ui64> rightSizes = {25'000, 50'000, 100'000, 250'000};
        const auto tupleSize = sizeof(ui64) + 64 * 2;

        CTEST << "----------------------------------------" << Endl;
        CTEST << "TEST NAME: " << "WideTupleJoin" << Endl;
        CTEST << "----------------------------------------" << Endl;
        for (ui32 pass = 0; pass < leftSizes.size(); ++pass) {
            TSetup<LLVM, SPILLING> setup;
            TProgramBuilder& pb = *setup.PgmBuilder;

            const auto lhsSize = leftSizes[pass];
            const auto rhsSize = rightSizes[pass];
            std::vector<TRuntimeNode> lhs, rhs;
            generate_complex_unique_data(pb, lhs, 0, lhsSize);
            generate_complex_unique_data(pb, rhs, 0, rhsSize);

            const auto tupleType = pb.NewTupleType({
                pb.NewDataType(NUdf::TDataType<ui64>::Id),
                pb.NewDataType(NUdf::TDataType<char*>::Id),
                pb.NewDataType(NUdf::TDataType<char*>::Id)
            });

            const auto list1 = pb.NewList(tupleType, std::move(lhs));
            const auto list2 = pb.NewList(tupleType, std::move(rhs));

            const auto resultType = pb.NewFlowType(pb.NewMultiType({
                pb.NewDataType(NUdf::TDataType<char*>::Id),
                pb.NewDataType(NUdf::TDataType<char*>::Id)
            }));

            const auto pgmReturn = pb.Collect(pb.NarrowMap(pb.GraceJoin(
                pb.ExpandMap(pb.ToFlow(list1), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U), pb.Nth(item, 2U)}; }),
                pb.ExpandMap(pb.ToFlow(list2), [&](TRuntimeNode item) -> TRuntimeNode::TList { return {pb.Nth(item, 0U), pb.Nth(item, 1U), pb.Nth(item, 2U)}; }),
                EJoinKind::Inner, {0U, 1U}, {0U, 1U}, {1U, 0U}, {2U, 1U}, resultType),
                [&](TRuntimeNode::TList items) -> TRuntimeNode { return pb.NewTuple(items); })
            );
            if (SPILLING) {
                setup.RenameCallable(pgmReturn, "GraceJoin", "GraceJoinWithSpilling");
            }

            const auto graph = setup.BuildGraph(pgmReturn);
            if (SPILLING) {
                graph->GetContext().SpillerFactory = std::make_shared<TMockSpillerFactory>();
            }

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            const auto iterator = graph->GetValue().GetListIterator();
            NUdf::TUnboxedValue tuple;
            ui64 total = 0;

            while (iterator.Next(tuple)) {
                auto t0 = tuple.GetElement(0);
                total += TString(t0.AsStringRef()).front() - 'a' + 1;
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            ui64 millisecondsJoin = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            CTEST << "Configuration: " << "Count of lhs elements = " << lhsSize << ", Count of rhs elements = " << rhsSize << ", disrt: unique = [0..size]" << Endl;
            CTEST << "Pure data size = " << tupleSize * (lhsSize + rhsSize) / 1024 << "[KB]" << Endl;
            CTEST << "Time for join = " << millisecondsJoin << "[ms]" << Endl;
            CTEST << "Join tuples speed: " << (tupleSize * (lhsSize + rhsSize) * 1000) / (millisecondsJoin * 1024 * 1024 + 1) << "MB/sec" << Endl;
            CTEST << "----------------------------------------" << Endl;
            CTEST << Endl;

            UNIT_ASSERT(!iterator.Next(tuple));
            UNIT_ASSERT_EQUAL(total, lhsSize);
        }
    }

}


}

}

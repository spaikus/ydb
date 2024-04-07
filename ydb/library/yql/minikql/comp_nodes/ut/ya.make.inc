FORK_SUBTESTS()

SPLIT_FACTOR(60)

IF (SANITIZER_TYPE == "thread" OR WITH_VALGRIND)
    TIMEOUT(3600)
    SIZE(LARGE)
    TAG(ya:fat)
ELSE()
    TIMEOUT(600)
    SIZE(MEDIUM)
ENDIF()

REQUIREMENTS(ram:32)

OWNER(g:yql)

INCLUDE(${ARCADIA_ROOT}/ydb/library/yql/minikql/computation/header.ya.make.inc)
INCLUDE(${ARCADIA_ROOT}/ydb/library/yql/minikql/invoke_builtins/header.ya.make.inc)

SET(ORIG_SRC_DIR ydb/library/yql/minikql/comp_nodes/ut)

SET(ORIG_SOURCES
    mkql_test_factory.cpp
    mkql_block_skiptake_ut.cpp
)

PEERDIR(
    ydb/library/yql/public/udf
    ydb/library/yql/public/udf/arrow
    ydb/library/yql/public/udf/service/exception_policy
    ydb/library/yql/sql/pg_dummy
    ydb/library/yql/utils/simd
)

IF (ARCH_X86_64)

CFLAGS(
    -mprfchw
    -DMKQL_DISABLE_CODEGEN
    -mavx2
)

ENDIF()

YQL_LAST_ABI_VERSION()

IF (MKQL_RUNTIME_VERSION)
    CFLAGS(
        -DMKQL_RUNTIME_VERSION=$MKQL_RUNTIME_VERSION
    )
ENDIF()

ADDINCL(
    ${ORIG_SRC_DIR}
)

COPY(
    WITH_CONTEXT
    AUTO
    FROM ${ORIG_SRC_DIR}
    ${ORIG_SOURCES}
    OUTPUT_INCLUDES
        ydb/library/yql/minikql/computation/mkql_computation_node_codegen.h
)


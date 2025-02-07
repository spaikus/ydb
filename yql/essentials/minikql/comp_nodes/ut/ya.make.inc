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

INCLUDE(${ARCADIA_ROOT}/yql/essentials/minikql/computation/header.ya.make.inc)
INCLUDE(${ARCADIA_ROOT}/yql/essentials/minikql/invoke_builtins/header.ya.make.inc)

SET(ORIG_SRC_DIR yql/essentials/minikql/comp_nodes/ut)

SET(ORIG_SOURCES
    mkql_test_factory.cpp
    mkql_grace_join_ut.cpp
)

PEERDIR(
    yql/essentials/public/udf
    yql/essentials/public/udf/arrow
    yql/essentials/public/udf/service/exception_policy
    yql/essentials/sql/pg_dummy
)

IF (ARCH_X86_64)

CFLAGS(
    -mprfchw
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
        yql/essentials/minikql/computation/mkql_computation_node_codegen.h
)


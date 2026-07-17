"""Tests for mixed-window analysis and lowering."""

import pytest

from codestr.errors import CompileError
from codestr.parser import parse
from codestr.planner import (
    ExecutionPlan,
    MixedWindowBoundary,
    PlanStep,
    _validate_step_dependencies,
    build_execution_plan,
    find_mixed_window_boundary,
)
from codestr.syntax import Call, Column
from codestr.udf.registry import UDFRegistry


@pytest.fixture
def registry():
    return UDFRegistry.get_instance()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "ts_mean(cs_moderate(x), 60)",
            MixedWindowBoundary("ts_mean", "ts", "cs_moderate", "cs"),
        ),
        (
            "cs_mean(ts_mean(x, 60))",
            MixedWindowBoundary("cs_mean", "cs", "ts_mean", "ts"),
        ),
        (
            "ts_mean(abs(cs_moderate(x)), 60)",
            MixedWindowBoundary("ts_mean", "ts", "cs_moderate", "cs"),
        ),
        (
            "ts_mean(add(left=cs_moderate(x), right=y), 60)",
            MixedWindowBoundary("ts_mean", "ts", "cs_moderate", "cs"),
        ),
    ],
)
def test_find_mixed_window_boundary(source, expected, registry):
    assert find_mixed_window_boundary(parse(source), registry) == expected


@pytest.mark.parametrize(
    "source",
    [
        "ts_mean(ts_mean(x, 5), 20)",
        "cs_rank(cs_moderate(x))",
        "cs_rank(x) + ts_mean(y, 20)",
        "abs(x) + log(y)",
    ],
)
def test_same_domain_and_sibling_windows_are_not_mixed(source, registry):
    assert find_mixed_window_boundary(parse(source), registry) is None


def test_unknown_call_is_opaque_to_window_analysis(registry):
    node = parse("ts_mean(not_registered(cs_moderate(x)), 60)")

    assert find_mixed_window_boundary(node, registry) is None


class TestExecutionPlan:
    def test_lowers_ts_over_cs_to_two_stages(self, registry):
        node = parse("ts_mean(cs_moderate(x), 60)")

        plan = build_execution_plan(node, registry)

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 2
        internal, final = plan.steps
        assert internal.internal is True
        assert str(internal.node) == "cs_moderate(x)"
        assert internal.output_name.startswith("__codestr_")
        assert final.internal is False
        assert final.output_name == node.alias
        assert isinstance(final.node, Call)
        assert final.node.args[0] == Column(internal.output_name)

    def test_lowers_alternating_domains_in_topological_order(self, registry):
        node = parse("ts_mean(cs_mean(ts_mean(x, 2, min_samples=1)), 2, min_samples=1)")

        plan = build_execution_plan(node, registry)

        assert len(plan.steps) == 3
        assert [step.internal for step in plan.steps] == [True, True, False]
        for step in plan.steps:
            assert find_mixed_window_boundary(step.node, registry) is None

    def test_deduplicates_repeated_mixed_subexpression(self, registry):
        node = parse("ts_mean(cs_moderate(x) + cs_moderate(x), 2, min_samples=1)")

        plan = build_execution_plan(node, registry)

        assert len(plan.steps) == 2
        internal_name = plan.steps[0].output_name
        final = plan.steps[-1].node
        add_call = final.args[0]
        assert isinstance(add_call, Call)
        assert add_call.args == (Column(internal_name), Column(internal_name))

    def test_single_domain_plan_has_one_public_step(self, registry):
        node = parse("ts_mean(ts_mean(x, 2), 3)")

        plan = build_execution_plan(node, registry)

        assert len(plan.steps) == 1
        assert plan.is_single_stage
        assert plan.steps[0].node == node
        assert plan.steps[0].output_name == node.alias
        assert plan.steps[0].internal is False

    def test_internal_name_avoids_existing_column(self, registry):
        node = parse("ts_mean(cs_moderate(x), 2)")
        first = build_execution_plan(node, registry)
        occupied = first.steps[0].output_name

        second = build_execution_plan(
            node,
            registry,
            existing_columns={occupied},
        )

        assert second.steps[0].output_name == f"{occupied}_1"

        reusable = build_execution_plan(
            node,
            registry,
            existing_columns={occupied},
            reusable_columns={occupied},
        )

        assert reusable.steps[0].output_name == occupied

    def test_internal_names_are_stable_and_preserve_literal_type(self, registry):
        int_node = parse("ts_mean(cs_moderate(x + 1), 2, min_samples=1)")
        float_node = parse("ts_mean(cs_moderate(x + 1.0), 2, min_samples=1)")

        first = build_execution_plan(int_node, registry)
        repeated = build_execution_plan(int_node, registry)
        float_plan = build_execution_plan(float_node, registry)

        assert first.steps[0].output_name == repeated.steps[0].output_name
        assert first.steps[0].output_name != float_plan.steps[0].output_name

    def test_unknown_call_is_not_partially_lowered(self, registry):
        node = parse("ts_mean(not_registered(cs_moderate(x)), 60)")

        plan = build_execution_plan(node, registry)

        assert plan.is_single_stage
        assert plan.steps[0].node == node

    def test_rejects_forward_internal_dependency(self):
        steps = (
            PlanStep(
                node=Column("__codestr_later"),
                output_name="result",
                internal=False,
            ),
            PlanStep(
                node=Column("x"),
                output_name="__codestr_later",
                internal=True,
            ),
        )

        with pytest.raises(CompileError, match="Invalid plan dependency"):
            _validate_step_dependencies(steps)

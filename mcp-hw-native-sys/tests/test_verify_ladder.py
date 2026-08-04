"""Tests for verify-ladder rule matching (mcp_hwnative_sys.verify_ladder)."""

from __future__ import annotations

import pytest

from mcp_hwnative_sys.verify_ladder import verify_ladder_impl


def test_empty_paths_raises():
    with pytest.raises(ValueError):
        verify_ladder_impl([])


def test_codegen_orchestration_greedy_prefix_wins():
    result = verify_ladder_impl(["pypto/src/codegen/orchestration/foo.cpp"])
    # Longer, more specific prefix must match before "pypto/src/codegen/".
    assert result["suggested_tasks"] == ["pypto:codegen_tests", "pypto:unit_tests_fast"]


def test_host_orch_substring_rule():
    result = verify_ladder_impl(["pypto/src/ir/foo_host_orch_bar.cpp"])
    assert "pypto-tooling:host_collectives_ut_sim" in result["suggested_tasks"]


def test_tasks_deduplicated_preserving_order():
    result = verify_ladder_impl(
        [
            "pypto/src/codegen/pto/a.cpp",
            "pypto/src/codegen/b.cpp",
        ]
    )
    tasks = result["suggested_tasks"]
    assert len(tasks) == len(set(tasks))
    assert tasks[0] == "pypto:codegen_tests"


def test_unmatched_path_yields_no_tasks():
    result = verify_ladder_impl(["README.md"])
    assert result["suggested_tasks"] == []
    assert result["matched_rules"] == []


def test_cpp_change_requires_clang_tidy():
    result = verify_ladder_impl(["pypto/src/codegen/pto/foo.cc"])
    assert result["static_checks"] == ["clang-tidy"]
    assert "clang-tidy" in result["note"]


def test_header_in_include_requires_clang_tidy_even_without_task_match():
    # pypto/include/ matches no verify rule but is still a C++ change needing clang-tidy.
    result = verify_ladder_impl(["pypto/include/pypto/ir/expr.h"])
    assert result["static_checks"] == ["clang-tidy"]
    assert result["suggested_tasks"] == []


def test_python_and_markdown_do_not_require_clang_tidy():
    for path in ["pypto/python/pypto/language/distributed/op/collective_api.py", "README.md"]:
        result = verify_ladder_impl([path])
        assert result["static_checks"] == [], path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

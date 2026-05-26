# Copyright 2026 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from verl.utils.reward_score import alebench


class _DummyResult:
    overall_absolute_score = 123.0
    overall_relative_score = 456.0


class _DummySession:
    def private_eval(self, code, code_language, judge_version=None):
        assert "int main" in code
        assert code_language == "cpp17"
        assert judge_version == alebench._JUDGE_VERSION
        return _DummyResult(), 12, 2345


def test_compute_score_uses_private_eval_per_sample(monkeypatch):
    started_problem_ids = []

    def fake_start_session(problem_id):
        started_problem_ids.append(problem_id)
        return _DummySession()

    monkeypatch.setattr(alebench, "_start_session", fake_start_session)

    result = alebench.compute_score(
        data_source="alebench",
        solution_str="```cpp\nint main() { return 0; }\n```",
        ground_truth="ahc001",
    )

    assert started_problem_ids == ["ahc001"]
    assert result == {
        "score": 2345.0,
        "performance": 2345.0,
        "rank": 12.0,
        "overall_absolute_score": 123.0,
        "overall_relative_score": 456.0,
    }


def test_compute_score_empty_code_does_not_start_session(monkeypatch):
    def fail_start_session(problem_id):
        raise AssertionError(f"unexpected ALE-Bench session for {problem_id}")

    monkeypatch.setattr(alebench, "_start_session", fail_start_session)

    result = alebench.compute_score(data_source="alebench", solution_str="", ground_truth="ahc001")

    assert result == {
        "score": 0.0,
        "performance": 0.0,
        "rank": None,
        "overall_absolute_score": 0.0,
        "overall_relative_score": None,
    }

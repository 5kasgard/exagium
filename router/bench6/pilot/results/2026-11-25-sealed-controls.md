# P28 / Q82: sealed attempt grader, known-answer controls (2026-11-25, Claude shift)

Commit `d9faa6fb4` on branch `fix/q82-warm-then-seal`. Rule (queue Q82, fixed 2026-11-24 before this change): the 32 gold/goldtest/empty/cheat controls on the 8 usable tasks must all read as expected before the sealed grader is used.

Change: every seal flag from the start except the network; a warm-up runs the task's own tests once on the unfixed code (test_patch applied, then reversed; tracked tree must be clean); `docker network disconnect`; only then inner.sh applies the attempt's diff, after a probe (TCP connect to 1.1.1.1:53) that refuses to grade if it connects.

Dry run first (2 tasks that failed with `--network none` on Nov 24, gold and empty): runs 45579735491, 45579739021: 4 of 4 as expected.

The 32: runs 45580836409, 45580839219, 45580841977, 45580845114 (modes gold, goldtest, empty, cheat; dispatch order). All 32 logs carry the warm-up marker; none carries the online marker.

```
as expected BerriAI__litellm-15753                        cheat    False    72.2s 2 of 20 tests not PASSED (0 of 2 FAIL_TO_PASS), e.g. tests/test_litellm/responses/litellm_completion
as expected BerriAI__litellm-15753                        empty    False    75.6s 2 of 20 tests not PASSED (0 of 2 FAIL_TO_PASS), e.g. tests/test_litellm/responses/litellm_completion
as expected BerriAI__litellm-15753                        gold     True     72.1s ok
as expected BerriAI__litellm-15753                        goldtest True     48.5s ok
as expected agronholm__anyio-1121                         cheat    False    43.5s 1 of 33 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. tests/test_pytest_plugin.py::test_keyboard_inte
as expected agronholm__anyio-1121                         empty    False    43.7s 1 of 33 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. tests/test_pytest_plugin.py::test_keyboard_inte
as expected agronholm__anyio-1121                         gold     True     41.4s ok
as expected agronholm__anyio-1121                         goldtest True     40.4s ok
as expected apache__iceberg-python-2797                   cheat    False    51.2s 3 of 328 tests not PASSED (0 of 3 FAIL_TO_PASS), e.g. tests/test_schema.py::test_check_schema_compat
as expected apache__iceberg-python-2797                   empty    False    80.1s 3 of 328 tests not PASSED (0 of 3 FAIL_TO_PASS), e.g. tests/test_schema.py::test_check_schema_compat
as expected apache__iceberg-python-2797                   gold     True     50.1s ok
as expected apache__iceberg-python-2797                   goldtest True     87.3s ok
as expected destine-climate-dt__aqua-2768                 cheat    False    50.2s 4 of 17 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/test_sci_tools.py::test_full_globe_select
as expected destine-climate-dt__aqua-2768                 empty    False    93.1s 4 of 17 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/test_sci_tools.py::test_full_globe_select
as expected destine-climate-dt__aqua-2768                 gold     True     92.6s ok
as expected destine-climate-dt__aqua-2768                 goldtest True     52.6s ok
as expected koxudaxi__datamodel-code-generator-3046       cheat    False    77.3s 4 of 518 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/main/jsonschema/test_main_jsonschema.py:
as expected koxudaxi__datamodel-code-generator-3046       empty    False    48.3s 4 of 518 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/main/jsonschema/test_main_jsonschema.py:
as expected koxudaxi__datamodel-code-generator-3046       gold     True     74.2s ok
as expected koxudaxi__datamodel-code-generator-3046       goldtest True     48.2s ok
as expected opsmill__infrahub-8407                        cheat    False   106.9s 1 of 27 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. backend/tests/component/core/diff/test_diff_com
as expected opsmill__infrahub-8407                        empty    False   106.2s 1 of 27 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. backend/tests/component/core/diff/test_diff_com
as expected opsmill__infrahub-8407                        gold     True    108.0s ok
as expected opsmill__infrahub-8407                        goldtest True    116.0s ok
as expected sunpy__sunpy-8548_interface                   cheat    False    37.7s 6 of 6 tests not PASSED (0 of 6 FAIL_TO_PASS), e.g. sunpy/coordinates/tests/test_table_serialization
as expected sunpy__sunpy-8548_interface                   empty    False    36.1s 6 of 6 tests not PASSED (0 of 6 FAIL_TO_PASS), e.g. sunpy/coordinates/tests/test_table_serialization
as expected sunpy__sunpy-8548_interface                   gold     True     67.1s ok
as expected sunpy__sunpy-8548_interface                   goldtest True     66.4s ok
as expected vprusso__toqito-1484_interface                cheat    False    32.0s 15 of 15 tests not PASSED (0 of 15 FAIL_TO_PASS), e.g. toqito/rand/tests/test_random_linearly_indepe
as expected vprusso__toqito-1484_interface                empty    False    30.6s 15 of 15 tests not PASSED (0 of 15 FAIL_TO_PASS), e.g. toqito/rand/tests/test_random_linearly_indepe
as expected vprusso__toqito-1484_interface                gold     True     31.1s ok
as expected vprusso__toqito-1484_interface                goldtest True     30.9s ok
32 of 32 as expected; missing runs: 0 of 32
```

Rule met: 32 of 32 (Nov 24 with `--network none` from the start: 28 of 32).

Not shown: the offline probe was not run once with the network up (a positive control for the probe itself); test-time fetches by the attempt's own code are blocked, but anything cached during the warm-up (build requirements, test data) is available to it, by design. A person merges router/ (Q82).

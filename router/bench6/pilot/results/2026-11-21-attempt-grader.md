# P28 pilot step 2: attempt grader known-answer check, 2026-11-21 (rules in ../PLAN.md, fixed before the runs)

**32 of 32 as expected; rule (all 32, none missing): met.** No model call. Runs on branch `p28/attempt-grader` (bench6-env.yml, ids=usable): gold 44959195886, goldtest 44959198446, empty 44959200539, cheat 44959202923. Dry run first (sunpy only, all four modes as expected): 44958948082, 44958950198, 44958952258, 44958954477. Output of `envcheck.py grade-attempts`:

```
as expected BerriAI__litellm-15753                        cheat    False    44.3s 2 of 20 tests not PASSED (0 of 2 FAIL_TO_PASS), e.g. tests/test_litellm/responses/litellm_completion
as expected BerriAI__litellm-15753                        empty    False    43.8s 2 of 20 tests not PASSED (0 of 2 FAIL_TO_PASS), e.g. tests/test_litellm/responses/litellm_completion
as expected BerriAI__litellm-15753                        gold     True     43.0s ok
as expected BerriAI__litellm-15753                        goldtest True     55.4s ok
as expected agronholm__anyio-1121                         cheat    False    35.7s 1 of 33 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. tests/test_pytest_plugin.py::test_keyboard_inte
as expected agronholm__anyio-1121                         empty    False    37.0s 1 of 33 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. tests/test_pytest_plugin.py::test_keyboard_inte
as expected agronholm__anyio-1121                         gold     True     30.6s ok
as expected agronholm__anyio-1121                         goldtest True     52.0s ok
as expected apache__iceberg-python-2797                   cheat    False    47.5s 3 of 328 tests not PASSED (0 of 3 FAIL_TO_PASS), e.g. tests/test_schema.py::test_check_schema_compat
as expected apache__iceberg-python-2797                   empty    False    47.7s 3 of 328 tests not PASSED (0 of 3 FAIL_TO_PASS), e.g. tests/test_schema.py::test_check_schema_compat
as expected apache__iceberg-python-2797                   gold     True     58.2s ok
as expected apache__iceberg-python-2797                   goldtest True     80.0s ok
as expected destine-climate-dt__aqua-2768                 cheat    False    71.3s 4 of 17 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/test_sci_tools.py::test_full_globe_select
as expected destine-climate-dt__aqua-2768                 empty    False    45.6s 4 of 17 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/test_sci_tools.py::test_full_globe_select
as expected destine-climate-dt__aqua-2768                 gold     True     44.4s ok
as expected destine-climate-dt__aqua-2768                 goldtest True     70.3s ok
as expected koxudaxi__datamodel-code-generator-3046       cheat    False    37.4s 4 of 518 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/main/jsonschema/test_main_jsonschema.py:
as expected koxudaxi__datamodel-code-generator-3046       empty    False    52.9s 4 of 518 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/main/jsonschema/test_main_jsonschema.py:
as expected koxudaxi__datamodel-code-generator-3046       gold     True     37.9s ok
as expected koxudaxi__datamodel-code-generator-3046       goldtest True     60.3s ok
as expected opsmill__infrahub-8407                        cheat    False    86.9s 1 of 27 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. backend/tests/component/core/diff/test_diff_com
as expected opsmill__infrahub-8407                        empty    False    80.9s 1 of 27 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. backend/tests/component/core/diff/test_diff_com
as expected opsmill__infrahub-8407                        gold     True     52.0s ok
as expected opsmill__infrahub-8407                        goldtest True     53.2s ok
as expected sunpy__sunpy-8548_interface                   cheat    False    34.1s 6 of 6 tests not PASSED (0 of 6 FAIL_TO_PASS), e.g. sunpy/coordinates/tests/test_table_serialization
as expected sunpy__sunpy-8548_interface                   empty    False    33.8s 6 of 6 tests not PASSED (0 of 6 FAIL_TO_PASS), e.g. sunpy/coordinates/tests/test_table_serialization
as expected sunpy__sunpy-8548_interface                   gold     True     33.8s ok
as expected sunpy__sunpy-8548_interface                   goldtest True     56.1s ok
as expected vprusso__toqito-1484_interface                cheat    False    29.3s 15 of 15 tests not PASSED (0 of 15 FAIL_TO_PASS), e.g. toqito/rand/tests/test_random_linearly_indepe
as expected vprusso__toqito-1484_interface                empty    False    47.5s 15 of 15 tests not PASSED (0 of 15 FAIL_TO_PASS), e.g. toqito/rand/tests/test_random_linearly_indepe
as expected vprusso__toqito-1484_interface                gold     True     47.1s ok
as expected vprusso__toqito-1484_interface                goldtest True     31.4s ok
32 of 32 as expected; missing runs: 0 of 32
```

What this shows: on these 8 tasks the grader tells the fix from no fix, undoes edits to the test_patch's files, and is not fooled by a root conftest.py that marks every test passed. What it does not show: anything about an agent; and attempts written to fool pytest's reporting in other ways (PLAN.md, "Not ruled out").

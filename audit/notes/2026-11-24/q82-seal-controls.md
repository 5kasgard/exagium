# Q82 seal: bench-6 known-answer controls on PR #125's branch (2026-11-24, Claude shift)

**28 of 32 as expected; rule (all 32, any miss means the grader is fixed before any model attempt): NOT met.**
No model call. Branch `worker/Q82-45393541132` (PR #125: `--network none --cap-drop ALL --security-opt no-new-privileges
--pids-limit 512 --memory 4g --cpus 2` on both `docker run` calls, more reset basenames). bench6-env.yml, ids=usable:
gold 45504000468, goldtest 45504003403, empty 45504007010, cheat 45504010365. Graded by script (expected: gold and
goldtest resolved, empty and cheat not) over the 32 `attempt-*.json` artifacts; 32 present, 8 ids.
Deviation: PLAN.md asks for a one-task dry run in each mode before the 32; this shift ran the 32 directly (no cost
either way: no model call). Same controls on the unsealed grader: 32 of 32 (2026-11-21, results/2026-11-21-attempt-grader.md on main).

## The 4 misses: both are network use at test time, which the seal now blocks
- **apache__iceberg-python-2797**, gold and goldtest: 328 of 328 tests not passed. Log: `Failed to resolve requirements
  from build-system.requires` ... `failed to lookup address information: Temporary failure in name resolution`: the
  task's test command (`uv run`) builds the package and fetches its build requirements at test time.
- **destine-climate-dt__aqua-2768**, gold and goldtest: 1 of 17 not passed (`test_selection_regionmask`). Log:
  `NameResolutionError ... naturalearth.s3.amazonaws.com ... ne_..._admin_0_countries.zip`: the test downloads map data.
- Empty and cheat read not resolved on all 8 (as expected); the seal did not let a cheat through.

## What would fix it (hypotheses for the person who merges router/; none tested)
1. Warm, then cut: start the container with network, run the task's own setup on the unfixed code (uv sync / build
   requirements; one test collection pass so data files are cached), then `docker network disconnect` before the
   attempt's diff is applied and graded. Keeps the seal for everything the attempt's code can do.
2. Or drop the two tasks that need network from `USABLE`: 6 left, the pilot's floor (fewer than 6 kept means too hard),
   so no margin.
Either way the 32 are re-run and must read 32 of 32 before Q82 merges.

```
BerriAI__litellm-15753                        cheat     False  2 of 20 tests not PASSED (0 of 2 FAIL_TO_PASS), e.g. tests/test_litellm/responses/litellm_completion_transform
BerriAI__litellm-15753                        empty     False  2 of 20 tests not PASSED (0 of 2 FAIL_TO_PASS), e.g. tests/test_litellm/responses/litellm_completion_transform
BerriAI__litellm-15753                        gold      True   ok
BerriAI__litellm-15753                        goldtest  True   ok
agronholm__anyio-1121                         cheat     False  1 of 33 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. tests/test_pytest_plugin.py::test_keyboard_interrupt_does
agronholm__anyio-1121                         empty     False  1 of 33 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. tests/test_pytest_plugin.py::test_keyboard_interrupt_does
agronholm__anyio-1121                         gold      True   ok
agronholm__anyio-1121                         goldtest  True   ok
apache__iceberg-python-2797                   cheat     False  328 of 328 tests not PASSED (0 of 3 FAIL_TO_PASS), e.g. tests/test_schema.py::test_check_schema_compatible_opt
apache__iceberg-python-2797                   empty     False  328 of 328 tests not PASSED (0 of 3 FAIL_TO_PASS), e.g. tests/test_schema.py::test_check_schema_compatible_opt
apache__iceberg-python-2797                   gold      False  328 of 328 tests not PASSED (0 of 3 FAIL_TO_PASS), e.g. tests/test_schema.py::test_check_schema_compatible_opt
apache__iceberg-python-2797                   goldtest  False  328 of 328 tests not PASSED (0 of 3 FAIL_TO_PASS), e.g. tests/test_schema.py::test_check_schema_compatible_opt
destine-climate-dt__aqua-2768                 cheat     False  5 of 17 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/test_sci_tools.py::test_full_globe_selection[sample
destine-climate-dt__aqua-2768                 empty     False  5 of 17 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/test_sci_tools.py::test_full_globe_selection[sample
destine-climate-dt__aqua-2768                 gold      False  1 of 17 tests not PASSED (4 of 4 FAIL_TO_PASS), e.g. tests/test_sci_tools.py::test_selection_regionmask
destine-climate-dt__aqua-2768                 goldtest  False  1 of 17 tests not PASSED (4 of 4 FAIL_TO_PASS), e.g. tests/test_sci_tools.py::test_selection_regionmask
koxudaxi__datamodel-code-generator-3046       cheat     False  4 of 518 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/main/jsonschema/test_main_jsonschema.py::test_main
koxudaxi__datamodel-code-generator-3046       empty     False  4 of 518 tests not PASSED (0 of 4 FAIL_TO_PASS), e.g. tests/main/jsonschema/test_main_jsonschema.py::test_main
koxudaxi__datamodel-code-generator-3046       gold      True   ok
koxudaxi__datamodel-code-generator-3046       goldtest  True   ok
opsmill__infrahub-8407                        cheat     False  1 of 27 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. backend/tests/component/core/diff/test_diff_combiner.py::
opsmill__infrahub-8407                        empty     False  1 of 27 tests not PASSED (0 of 1 FAIL_TO_PASS), e.g. backend/tests/component/core/diff/test_diff_combiner.py::
opsmill__infrahub-8407                        gold      True   ok
opsmill__infrahub-8407                        goldtest  True   ok
sunpy__sunpy-8548_interface                   cheat     False  6 of 6 tests not PASSED (0 of 6 FAIL_TO_PASS), e.g. sunpy/coordinates/tests/test_table_serialization.py::test_
sunpy__sunpy-8548_interface                   empty     False  6 of 6 tests not PASSED (0 of 6 FAIL_TO_PASS), e.g. sunpy/coordinates/tests/test_table_serialization.py::test_
sunpy__sunpy-8548_interface                   gold      True   ok
sunpy__sunpy-8548_interface                   goldtest  True   ok
vprusso__toqito-1484_interface                cheat     False  15 of 15 tests not PASSED (0 of 15 FAIL_TO_PASS), e.g. toqito/rand/tests/test_random_linearly_independent_vect
vprusso__toqito-1484_interface                empty     False  15 of 15 tests not PASSED (0 of 15 FAIL_TO_PASS), e.g. toqito/rand/tests/test_random_linearly_independent_vect
vprusso__toqito-1484_interface                gold      True   ok
vprusso__toqito-1484_interface                goldtest  True   ok
28 of 32 as expected; files 32; ids 8
```

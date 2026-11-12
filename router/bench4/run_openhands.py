"""bench-4 OpenHands arm: the SDK's default agent working directly in /workspace (no Agent Server, no browser).

usage: run_openhands.py TASK_FILE MAX_ITERATIONS
Env: BENCH4_BASE_URL (the meter), BENCH4_MODEL. Prints one JSON line with the ending.
Same SDK version as bench-2 and bench-3 (v1.61.0); the difference is that tools run in this process's
container instead of a separate Agent Server container, and browser tools are off (cli_mode), as for every
other bench-4 harness, which has no browser either.
"""
import json, os, sys

from pydantic import SecretStr
from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent

task, maxit = open(sys.argv[1]).read(), int(sys.argv[2])
out = {"error": None}
llm = LLM(usage_id="agent", model="openrouter/" + os.environ["BENCH4_MODEL"], base_url=os.environ["BENCH4_BASE_URL"],
          api_key=SecretStr("dummy"))
conv = Conversation(agent=get_default_agent(llm=llm, cli_mode=True), workspace="/workspace", max_iteration_per_run=maxit)
conv.send_message(task)
try:
    conv.run()
except Exception as e:  # the loop ending in an error is a result, recorded as text
    out["error"] = f"{type(e).__name__}: {e}"[:1500]
out["status"] = str(conv.state.execution_status)
print("@@END@@" + json.dumps(out))

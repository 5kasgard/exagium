"""One repair attempt after a failed proposal, for P5d."""


def run_arm(arm, call, prompt, check):
    if arm not in ("baseline", "packet"):
        raise ValueError(arm)
    reply = call(prompt)
    calls = 1
    error = check(reply)
    if arm == "baseline" or error is None:
        return {"reply": reply, "valid": error is None, "calls": calls, "error": error}
    reply = call(prompt + "\n\n" + error)
    calls = 2
    error = check(reply)
    return {"reply": reply, "valid": error is None, "calls": calls, "error": error}

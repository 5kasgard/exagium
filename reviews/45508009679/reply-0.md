<!-- groq openai/gpt-oss-120b status=200 latency=3.71s -->

**EXPLOITABLE**  
1. **Model‑driven command injection** – The `opencode` binary runs arbitrary shell commands supplied by the LLM. The LLM sees the full issue thread, which is attacker‑controlled. It can be coaxed (prompt injection) into executing `gh api -X POST /repos/${{github.repository}}/dispatches …` or `git push` (the token has `contents: read` only, but a push can be done via the GitHub REST API using the `pulls` permission? No – token lacks `write` on `contents`. Still it can create a new issue comment containing a malicious URL that later triggers other workflows). More concretely, the model can run `curl -X POST -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/repos/owner/repo/actions/workflows/other.yml/dispatches …` to start a workflow that *does* have secret access, achieving privilege escalation.  

2. **Recursive trigger / DoS** – The agent can post a comment that itself begins with “/oc …”. That spawns another run, which can repeat indefinitely until the concurrency group limit is hit, exhausting the runner quota.  

3. **Cache‑poisoning execution** – If an attacker can cause the `actions/cache` step to restore a malicious `~/.opencode/bin/opencode` binary (by publishing a malicious release of the **opencode** tool with the same version tag), the subsequent `opencode github run` will execute attacker code with the same token.  

4. **Exfiltration via issue comments** – The model can embed stolen repository files (read via the token) into a comment, leaking source code or any files that the repo owner has made readable (including other secret‑containing files accidentally left in the repo).  

**SECRETS**  
No secrets are passed explicitly. The only token is `GITHUB_TOKEN` with `contents: read` and `issues: write`. The model can read any file in the repository (including accidentally committed secrets) and dump them in a comment, effectively leaking them. It cannot directly read other repository‑wide secrets because those are not exposed to the runner.  

**CACHE / SUPPLY‑CHAIN**  
- **actions/cache** caches `~/.opencode/bin` keyed by the *opencode* release tag fetched from the public GitHub API. An attacker who can publish a malicious release (or compromise the upstream release assets) can inject a trojan binary that will be restored from cache and executed.  
- **`curl https://opencode.ai/install | bash`** is an unpinned, unsanitized install script. If the host site is compromised or a DNS hijack occurs, arbitrary code can be run before any caching.  

*Cheapest mitigation*: Pin the install script to a specific SHA (download the script, store it in the repo, and `bash ./install.sh`). Also pin the cached version by using a fixed version tag rather than “latest”.  

**CONDITION**  
`author_association == 'OWNER'` blocks most attackers, but a malicious GitHub App installed with **owner** permission can post comments that appear as “OWNER”. Additionally, a compromised personal access token belonging to the owner could be used to post comments via the API, bypassing the check. Edited comments are ignored, so an attacker cannot change a non‑triggering comment to trigger later.  

**VERDICT**  
**SHIP WITH CHANGES**  

- Pin the `opencode` install script (store a copy, reference by SHA).  
- Replace the “latest release” cache key with a static, vetted version.  
- Add a secondary check: require `github.event.comment.user.login == github.repository_owner`.  
- Rate‑limit recursion: abort if a comment created by the bot contains the trigger prefix.  
- Consider restricting the token scope further (remove `contents: read` if not needed).  

These mitigations close the most realistic privilege‑escalation and supply‑chain vectors.

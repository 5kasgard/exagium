You are an independent security reviewer. A first version of this workflow was rejected because a model with a shell
ran in a job holding a write token. This is the revision. Attack it. Be specific; no politeness.

Facts:
- PUBLIC repository; anyone can comment. `main` has no branch protection. The owner is a user account, not an org.
- OPENCODE_API_KEY can reach paid models the owner connected (Anthropic, Google, NVIDIA, Hugging Face) on the OpenCode
  Console; it is billed to the owner. Free models need no key.
- opencode 1.19.6 is installed with `npm ci` from a committed package-lock.json (integrity hashes).
- agent/opencode.json is copied to /etc/opencode/opencode.json (system config; measured: it overrides a hostile repo
  opencode.json). It defines agent "answer": permission bash/edit/webfetch/websearch/external_directory/task/skill deny;
  read/glob/grep/list allow. Measured resolved tools: bash false, edit false, write false, webfetch false, task false;
  read/glob/grep true. external_directory deny covers paths outside the checkout (e.g. /proc).
- agent/pick.py (no model) parses the comment: "/oc [model=<alias|provider/model>] <question>" or "/oc models"; model id
  must match ^[a-z0-9][a-z0-9-]*/[A-Za-z0-9][A-Za-z0-9._:-]{0,120}$; ids not in models.json free_ids are paid and
  refused unless models.json allow_paid is true. It writes prompt.md from the comment, plus issue title/body only if
  the issue author is the owner. Exit 2 on refusal (the refusal text is posted).

```yaml
# Issue agent (queue P36, owner 2026-11-24/25): the owner comments "/oc <question>" on an issue and gets an answer
# there, from the model they pick: "/oc model=<alias or provider/model> <question>"; "/oc models" lists what is reachable.
# Free models run with no key. Paid models (providers the owner connected on the OpenCode Console) run with
# OPENCODE_API_KEY, only when agent/models.json says allow_paid. Spend limits belong at each provider and the Console.
#
# Two jobs, so that no model ever runs where a write token or a key could be read back out (blind review 45508009679):
#   answer: contents: read only. The model gets a read-only agent (agent/opencode.json, installed as the system config,
#           which overrides any repo config): no shell, no edits, no web, no files outside the checkout. It cannot read
#           its own environment, so it cannot print the key. Only the owner's text reaches it: the comment, and the issue
#           title and body only when the owner opened the issue. Other people's comments are never sent.
#   post:   no model. Posts the answer file as a comment with an issues: write token.
# opencode is pinned through agent/package-lock.json (npm ci checks integrity); no curl | bash, no cache.
name: opencode
on:
  issue_comment:   # zizmor: ignore[dangerous-triggers] issues only (no PR head checked out), owner-only by login and association; the model job has contents: read and no shell
    types: [created]
permissions: {}
concurrency:
  group: opencode-${{ github.event.issue.number }}
  cancel-in-progress: false
jobs:
  answer:
    if: >-
      github.event.issue.pull_request == null &&
      github.event.comment.author_association == 'OWNER' &&
      github.event.comment.user.login == github.repository_owner &&
      (startsWith(github.event.comment.body, '/oc') || startsWith(github.event.comment.body, '/opencode'))
    runs-on: ubuntu-24.04
    timeout-minutes: 25
    permissions:
      contents: read
    outputs:
      model: ${{ steps.pick.outputs.model }}
      paid: ${{ steps.pick.outputs.paid }}
      mode: ${{ steps.pick.outputs.mode }}
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
        with:
          fetch-depth: 1
          persist-credentials: false
      - name: pick (script, no model)
        id: pick
        env:
          COMMENT: ${{ github.event.comment.body }}
          ISSUE_TITLE: ${{ github.event.issue.user.login == github.repository_owner && github.event.issue.title || '' }}
          ISSUE_BODY: ${{ github.event.issue.user.login == github.repository_owner && github.event.issue.body || '' }}
        run: |
          set +e
          python3 -I agent/pick.py agent/models.json "$RUNNER_TEMP/out" >> "$GITHUB_OUTPUT"
          rc=$?
          echo "rc=$rc" >> "$GITHUB_OUTPUT"
          [ "$rc" -le 2 ]
      - name: install opencode (pinned) and the read-only agent as the system config
        if: steps.pick.outputs.rc == '0'
        run: |
          (cd agent && npm ci --no-fund --no-audit)
          sudo mkdir -p /etc/opencode && sudo cp agent/opencode.json /etc/opencode/opencode.json
      - name: answer on a free model (no key)
        if: steps.pick.outputs.rc == '0' && steps.pick.outputs.mode == 'ask' && steps.pick.outputs.paid == 'false'
        env:
          MODEL: ${{ steps.pick.outputs.model }}
        run: |
          timeout 900 agent/node_modules/.bin/opencode run --agent answer -m "$MODEL" -f "$RUNNER_TEMP/out/prompt.md" \
            "Answer the request in the attached file. Read the repository files you need and name the paths you used." \
            > "$RUNNER_TEMP/out/answer.md" 2> "$RUNNER_TEMP/err.txt" || echo "The run failed (exit $?); see the workflow log." >> "$RUNNER_TEMP/out/answer.md"
      - name: answer on a paid model (OPENCODE_API_KEY, this step only)
        if: steps.pick.outputs.rc == '0' && steps.pick.outputs.mode == 'ask' && steps.pick.outputs.paid == 'true'
        env:
          MODEL: ${{ steps.pick.outputs.model }}
          OPENCODE_API_KEY: ${{ secrets.OPENCODE_API_KEY }}
        run: |
          timeout 900 agent/node_modules/.bin/opencode run --agent answer -m "$MODEL" -f "$RUNNER_TEMP/out/prompt.md" \
            "Answer the request in the attached file. Read the repository files you need and name the paths you used." \
            > "$RUNNER_TEMP/out/answer.md" 2> "$RUNNER_TEMP/err.txt" || echo "The run failed (exit $?); see the workflow log." >> "$RUNNER_TEMP/out/answer.md"
      - name: list reachable models (no inference)
        if: steps.pick.outputs.rc == '0' && steps.pick.outputs.mode == 'models'
        env:
          OPENCODE_API_KEY: ${{ steps.pick.outputs.paid == 'true' && secrets.OPENCODE_API_KEY || '' }}
        run: |
          timeout 120 agent/node_modules/.bin/opencode models > "$RUNNER_TEMP/out/answer.md" 2> "$RUNNER_TEMP/err.txt" \
            || echo "Listing failed; see the workflow log." >> "$RUNNER_TEMP/out/answer.md"
      - name: errors to the log only (never to the comment)
        if: ${{ !cancelled() }}
        run: '[ -s "$RUNNER_TEMP/err.txt" ] && tail -n 40 "$RUNNER_TEMP/err.txt" || true'
      - name: cap the answer
        if: ${{ !cancelled() }}
        run: |
          f="$RUNNER_TEMP/out/answer.md"; [ -s "$f" ] || echo "No answer was produced; see the workflow log." > "$f"
          head -c 60000 "$f" > "$f.cut" && mv "$f.cut" "$f"
      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
        if: ${{ !cancelled() }}
        with:
          name: answer
          path: ${{ runner.temp }}/out/answer.md
          retention-days: 7
  post:
    needs: answer
    if: ${{ always() && needs.answer.result != 'skipped' && needs.answer.result != 'cancelled' }}
    runs-on: ubuntu-24.04
    timeout-minutes: 5
    permissions:
      issues: write
    steps:
      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
        with:
          name: answer
          path: out
      - name: comment
        env:
          GH_TOKEN: ${{ github.token }}
          REPO: ${{ github.repository }}
          ISSUE: ${{ github.event.issue.number }}
          MODEL: ${{ needs.answer.outputs.model }}
          PAID: ${{ needs.answer.outputs.paid }}
          MODE: ${{ needs.answer.outputs.mode }}
          RUN: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
        run: |
          tier=free; [ "$PAID" = true ] && tier=paid
          { if [ "$MODE" = ask ]; then echo "**opencode** · \`$MODEL\` · $tier · [run]($RUN)"; else echo "**opencode** · [run]($RUN)"; fi
            echo; cat out/answer.md; } > body.md
          gh issue comment "$ISSUE" -R "$REPO" --body-file body.md
```

Answer in this form, under 450 words:
EXPLOITABLE: concrete ways a non-owner, or text that reaches the model, could make either job do more than read the
repo, spend the owner's money, or post one comment; or "none found" with reasoning.
KEY: can the paid step's OPENCODE_API_KEY leak (to the model, the answer, logs, artifacts, or the post job)? How?
SPEND: what stops runaway spend, and what is missing?
RESIDUAL: the single largest remaining risk and its cheapest control.
VERDICT: SHIP AS IS, SHIP WITH CHANGES (list), or DO NOT SHIP.

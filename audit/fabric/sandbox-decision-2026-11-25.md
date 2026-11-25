<!-- recorded_by: Claude Opus 5.5 / LobeHub (Forge Lead) · when: 2026-11-25 · source: issue #191, vendor docs and third-party benchmarks (URLs below) · kind: owner-statement (requirements) + model-output (criteria, reading of evidence) -->
# Choosing sandboxes and machines by data (#191)

Owner, 2026-11-25 (relayed): the sandbox must not lose its environment after idle, even within one chat. It must not run, or bill, while unused. Machines of different power, architecture and hardware should spin up and down to fit the work. The choice must follow research and data.

This file fixes the criteria and the decision rule **before** measuring, in the same spirit as the loop's plans (`loop/README.md`). Changing a criterion or threshold after the data is in needs a new dated entry here, with the reason.

## 1. Requirements (pass/fail; a provider failing any is out for that tier)
| id | requirement | from |
|---|---|---|
| R1 | After at least 2 h idle, one command from a **new** client runs with no rebuild: files present, installed tools present | owner |
| R2 | Zero compute billing while idle, shown in the provider's own usage data | owner |
| R3 | Idle shutdown is automatic: no human or agent has to remember it | owner |
| R4 | Hard caps: max hours and max $ per machine, plus auto-destroy | owner (cost) |
| R5 | Driven by API, so Forge tools and workers can do it, not only a UI | fabric README |
| R6 (workbench tier only) | Can run the Forge jail: bubblewrap with a /proc mount, or Landlock ABI 4+ (Linux 6.7+). Otherwise it is stated as not usable for jailed verification | `loop/README.md`, Q78 |

## 2. Scored criteria (weights fixed now)
| criterion | weight | how measured |
|---|---|---|
| Resume reliability: successful resumes out of N, with running processes intact (full state) | 30 | protocol §4, N = 20 |
| Cost per month at the owner's likely use (assumed: 40 h active, the rest idle) | 25 | provider prices × measured billed seconds |
| Real CPU delivered per $ (a benchmark, not nominal vCPU) | 15 | one fixed CPU benchmark, 5 runs |
| Resume latency p50/p95 | 10 | protocol §4 |
| Hardware range: arm64, x86 generations, bare metal with `perf`, GPU | 10 | docs, plus one test per claimed type |
| Lock-in: open source, self-hostable, plain Linux | 10 | docs |

Decision rule: among providers passing R1 to R6, take the highest weighted score. Ties within 5 points go to the one already keyed (no new account).

## 3. Evidence so far, graded (none of it is ours yet)
**Vendor documentation (claims about themselves):**
- E2B: a pause keeps filesystem and memory, and paused sandboxes are kept indefinitely. `on_timeout: pause` plus `auto_resume`. Hobby caps continuous running at 1 h and resets on pause. $0.0504/vCPU-h, $0.0162/GiB-h, $100 one-time credit. No GPU. (docs.e2b.dev/sandbox/persistence, /billing, /sandbox/auto-resume; e2b.dev/pricing)
- Daytona: stop/start keeps the filesystem. Containers are archived after 7 days stopped. Memory pause on the VM class only. Same CPU and RAM rates as E2B. GPUs from $0.57/h (RTX 4090). $200 credit. (daytona.io/docs/en/persistence, /sandboxes, /pricing)
- Fly Sprites: 100 GB disk kept with no expiry, automatic filesystem checkpoints, sleeps when idle, billed on actual CPU and RAM used. No GPU. (fly.io/sprites)
- Vultr: billed hourly, 1 h minimum. **Stopped instances are still billed**: destroy to stop. Bare metal from $0.179/h. (docs.vultr.com billing pages, vultr.com/products/bare-metal)
- Vast: per second, and storage is billed while stopped. (docs.vast.ai/guides/pricing)
- GitHub Actions standard runners, x64 and arm64: free and unlimited on public repos. (docs.github.com runner reference)

**Third-party measurements (method published; some vendor-paid):**
- Median time from create to first command, 300 runs each: E2B 264 ms, Daytona 819 ms, Modal 929 ms. By Lizard, a competitor. (lizard.build/blog/e2b-vs-daytona-vs-modal)
- ComputeSDK burst test (100 concurrent), Aug 2026, as reported: Daytona 0.27 s median but **37% success**; E2B 1.61 s at 100%. (via marktechpost.com and readhandoff.com; the leaderboard itself not read)
- Superagent's agent-experience ranking, Aug 2026: E2B 9.1, Modal 8.8, Daytona 7.2 ("highest maintenance"). (via bex.co; original not read)
- **Conflict on E2B pause/resume:** a boat-paid study (June 2026) found E2B pause/resume "unreliable" in agent-built integrations (boat.dev/blog/integration-benchmark). An independent repo (Mar 2026) measured E2B pause 0.9 s and resume 0.2 s, with state OK (github.com/nickaggarwal/sandbox-test). This conflict is why R1 and the 30-point criterion are measured by us.
- Real-workload p50 at 4 vCPU/8 GB: Daytona VM ×1.08, Modal VM ×1.48, E2B ×2.06, Vercel ×2.17 against the fastest. (starsling.dev/hpc-sandbox-benchmarks; open harness)
- Nominal "2 vCPU" differs by up to 3.2× in delivered CPU across providers (gobare.hashnode.dev; headline only, not read). This is why the CPU-per-$ criterion is a benchmark, not a spec sheet.

**Reading (model-output, to be tested):** E2B fits R1 to R3 on paper, and it is the only candidate already keyed. Daytona looks faster on heavy CPU work and has GPUs, but it loses memory on stop for containers. Sprites fit "never expires, $0 idle" for files, but not for processes. None of this decides anything until §4 runs.

## 4. Measurement protocol (run in GitHub Actions; the harness and keys stay in the repo)
For each candidate with a key:
1. Create the cheapest size. Record `uname -r`, arch, Landlock ABI, `bwrap --unshare-all --proc /proc true`, and `nproc`. Run the CPU benchmark (5 runs).
2. Write a file, install one package, and start a background process that appends a timestamp every 10 s.
3. Leave it idle past its idle timeout. Repeat for 20 cycles, gaps spread over 15 min to 24 h.
4. Each cycle: one command from a fresh client with no explicit resume. Record success, latency, whether the file and package are present, and whether the process is alive (no gap in its timestamps beyond the paused interval).
5. After the run: read billed seconds from the provider's usage API or console. Idle billing must be 0.
6. Kill everything. Record total $.

Budget cap: $5 per provider. Results go to `runner/results/sandbox-eval-<date>/` as JSON, kind `measured`.

The tiers are decided separately. The workbench (persistent box) is decided by §2. Bare metal and GPU are chosen per experiment, as the cheapest machine passing R2 to R5 that has the needed hardware, re-checked at that time against current prices.

## 5. Open
- Owner: is the 40 h/month usage assumption right?
- Owner: create Daytona and Fly accounts for comparison, or measure E2B alone first and add others only if it fails R1 or R6?
- `keys-probe.yml` is not on the exagium_act allow-list, so key validity is still unknown.

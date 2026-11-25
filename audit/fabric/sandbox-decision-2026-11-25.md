<!-- recorded_by: Claude Opus 5.5 / LobeHub (Forge Lead) · when: 2026-11-25 · source: issue #191, vendor docs and third-party comparisons (URLs below) · kind: owner-statement (requirements) + model-output (criteria, reading of evidence) -->
# Choosing sandboxes and machines by data (#191)

Owner, 2026-11-25 (relayed): the sandbox must not lose its environment after idle, even within one chat. It must not run, or bill, while unused. Machines of different power, architecture and hardware should spin up and down to fit the work. The choice must follow research and data. The field must not be limited to providers the Forge already has keys for.

This file fixes the criteria and the decision rule **before** measuring, in the same spirit as the loop's plans (`loop/README.md`). Changing a criterion or threshold after the data is in needs a new dated entry here, with the reason.

**Correction (2026-11-25, same author):** the first draft gave a tie-break to "already keyed" providers. That biases the field toward past choices. It is removed. Having a key now counts only as setup cost, and only inside the cost criterion.

## 1. Requirements (pass/fail; a provider failing any is out for that tier)
| id | requirement | from |
|---|---|---|
| R1 | After at least 2 h idle, one command from a **new** client runs with no rebuild: files present, installed tools present | owner |
| R2 | Zero compute billing while idle, shown in the provider's own usage data. Storage billing allowed; it is scored under cost | owner |
| R3 | Idle shutdown is automatic: no human or agent has to remember it | owner |
| R4 | Hard caps: max hours and max $ per machine, plus auto-destroy | owner (cost) |
| R5 | Driven by API, so Forge tools and workers can do it, not only a UI | fabric README |
| R6 (workbench tier only) | Can run the Forge jail: bubblewrap with a /proc mount, or Landlock ABI 4+ (Linux 6.7+). Otherwise it is stated as not usable for jailed verification | `loop/README.md`, Q78 |
| R7 | The service is not winding down (e.g. Runloop announced shutdown for 2026-11-13) | evidence |

## 2. Scored criteria (weights fixed now)
| criterion | weight | how measured |
|---|---|---|
| Resume reliability: successful resumes out of N; processes intact scores higher than files only | 30 | protocol §5, N = 20 |
| Cost per month at the owner's likely use (assumed: 40 h active, the rest idle), including plan fees, idle storage and one-time setup | 25 | provider prices × measured billed seconds |
| Real CPU delivered per $ (a benchmark, not nominal vCPU) | 15 | one fixed CPU benchmark, 5 runs |
| Resume latency p50/p95 | 10 | protocol §5 |
| Hardware range: arm64, x86 generations, bare metal with `perf`, GPU | 10 | docs, plus one test per claimed type |
| Lock-in: open source, self-hostable, plain Linux | 10 | docs |

Decision rule: among providers passing R1 to R7, take the highest weighted score. A difference within 5 points is reported as a tie, and the owner picks.

## 3. The field (all candidates, keyed or not)
Source for the long list: agentsandboxes.org/sandboxes (126 providers, 2026-11-08). It is a directory, not evidence; each entry is screened from its own docs.

**Stage A, desk screen (docs only, cheap, suited to a free worker):** every hosted sandbox, persistent-VM and self-host entry in that directory is checked against R1 to R7 from its own docs, with the source quoted. Output: `audit/fabric/sandbox-screen-<date>.json`.

**Stage B, measurement:** the top 4 to 6 by desk score are measured (§5), keyed or not. The owner creates the accounts.

**Desk evidence so far** (vendor docs unless marked; *third-party* = a comparison site, often written by a competitor):
| provider | what survives idle | idle billing | session cap | notes |
|---|---|---|---|---|
| E2B (keyed) | pause keeps files **and memory/processes**, kept indefinitely; auto-pause and auto-resume | none while paused | 1 h Hobby / 24 h Pro, reset by pause | $100 one-time credit; no GPU (docs.e2b.dev) |
| Cloudflare Sandbox (CF account keyed; plan unknown) | files and processes **end when the instance stops**. Files carry over only via snapshots (public beta, free during beta) or R2 backup and mounts | memory and disk billed while running, even idle; none when stopped. Default sleepAfter 10 min | none stated | **needs Workers Paid ($5/mo)**, which includes 375 vCPU-min and 25 GiB-h a month. microVM with its own kernel. A production restore is an overlay that is lost on sleep and must be re-restored (developers.cloudflare.com/sandbox, /containers/platform/pricing) |
| Vercel Sandbox | files snapshotted on stop, restored on resume; processes restart. Snapshots expire 30 days after last use | billed on active CPU only, so idle CPU is near 0 while running; $0 when stopped | 45 min Hobby / 24 h Pro per session | Hobby: 5 active CPU-h a month free (vercel.com/docs/sandbox; *third-party* indexagentica.com) |
| Blaxel | standby about 15 s after the last connection, **keeps memory and processes**; resume about 25 ms claimed | none on standby; standby snapshots $0.20/GB-month | none stated | billed by RAM while active (*third-party* lizard.build, agent37.com) |
| Daytona | **conflict:** older docs say stop/start keeps files and VM-class pause keeps memory; a Nov 2026 *third-party* page says Daytona 2.0 (alpha) deletes on timeout | none when stopped | 24 h / 48 h on 2.0 (*third-party*) | GPUs; went closed source June 2026 (*third-party* ajayk.sh) |
| Fly Sprites | disk kept with no expiry; processes only while "warm" | none when asleep | none | no GPU (fly.io/sprites) |
| Upstash Box (Upstash keyed) | persistent container disk | billed on active CPU | not checked | about $8/mo flat kept alive; a container, not a VM, so R6 is doubtful (*third-party* prized.dev) |
| Northflank | ephemeral unless a volume is attached | per second | none | about $0.067/h for 2 vCPU/4 GB, the cheapest listed; GPUs (*third-party* lizard.build) |
| CodeSandbox / Together | hibernate keeps memory, falling back to disk only over days | credits | - | (*third-party*) |
| Freestyle, Morph Cloud, Tensorlake, Prized, boat, Agent37, Docker Cloud Sandboxes, Islo, exe.dev | various; not yet read in their own docs | | | Stage A |
| Modal | nothing unless snapshotted; memory snapshots 7 days | | 24 h | GPUs |
| Runloop | **out (R7):** winding down, per its own notice as reported by prized.dev | | | |

**Not a sandbox service, but the same need** (the persistent box and the hardware tiers):
| option | what | cost | catch |
|---|---|---|---|
| Oracle Cloud Always Free | 2 Ampere ARM OCPU + 12 GB, 200 GB disk, own kernel, always on | $0 | halved in June 2026 without notice; idle VMs reclaimed if under 20% use for 7 days; capacity scarce (docs.oracle.com; *third-party* gundyr.com) |
| Google Cloud e2-micro | tiny x86 box, always on | $0 | 1 GB RAM |
| Self-hosted runtimes (microsandbox, Kubernetes Agent Sandbox, OpenComputer) on a free or cheap box | full control, no lock-in | box cost only | we maintain it |
| GitHub Actions standard runners | x64 + arm64, stateless | $0 for public repos | no persistence; already used by `loop.yml` |
| Hugging Face Spaces | CPU Basic 2 vCPU/16 GB free; ZeroGPU 5 min a day free (40 with PRO, $9/mo) | $0 to $9 | disk not persistent by default; built for demos |
| Vultr, Hetzner, Vast, Daytona GPU | bare metal, ARM, GPUs, hourly | $0.01 to $2.50+/h | Vultr and Vast bill stopped machines; Hetzner's cheap lines unavailable in Oct 2026 |

**Reading (model-output, to be tested):** two designs compete.
- **(a) Memory-keeping pause:** E2B, Blaxel, CodeSandbox. Resume exactly where you left off, processes included.
- **(b) Files-only persistence with fast restart:** Cloudflare, Vercel, Sprites, boat, Oracle box. The environment is kept, but running processes restart.

The owner's complaint, having to rebuild, is met by both. "Mid-task process still running after a long gap" is met only by (a). Which matters more is the owner's call. The measurements give the costs of each.

## 4. Third-party measurements (method published; some vendor-paid)
- Median create to first command, 300 runs each: E2B 264 ms, Daytona 819 ms, Modal 929 ms. By Lizard, a competitor. (lizard.build/blog/e2b-vs-daytona-vs-modal)
- ComputeSDK burst test (100 concurrent), Aug 2026, as reported: Daytona 0.27 s at **37% success**; E2B 1.61 s at 100%; Cloudflare 5.06 s. (via marktechpost.com and readhandoff.com; leaderboard not read)
- Superagent agent-experience ranking, Aug 2026: E2B 9.1, Modal 8.8, Freestyle 8.4, Vercel 7.5, Daytona 7.2, Blaxel 6.9, Cloudflare 6.7. (via bex.co; original not read)
- **Conflict on E2B pause/resume:** a boat-paid study (June 2026) found it "unreliable" in agent-built integrations; an independent repo (Mar 2026) measured pause 0.9 s and resume 0.2 s, with state OK. (boat.dev/blog/integration-benchmark; github.com/nickaggarwal/sandbox-test)
- Real-workload p50 at 4 vCPU/8 GB: Daytona VM ×1.08, Modal VM ×1.48, E2B ×2.06, Vercel ×2.17 against the fastest. (starsling.dev/hpc-sandbox-benchmarks; open harness, so it could be re-run by us)
- Nominal "2 vCPU" differs up to 3.2× in delivered CPU (gobare.hashnode.dev; headline only).

None of these measured idle-then-resume after hours, which is the owner's actual failure. That gap is why §5 exists.

## 5. Measurement protocol (run in GitHub Actions; the harness and keys stay in the repo)
For each Stage B candidate:
1. Create the cheapest size. Record `uname -r`, arch, Landlock ABI, `bwrap --unshare-all --proc /proc true`, and `nproc`. Run the CPU benchmark (5 runs).
2. Write a file, install one package, and start a background process that appends a timestamp every 10 s.
3. Leave it idle past its idle timeout. Repeat for 20 cycles, gaps spread over 15 min to 24 h.
4. Each cycle: one command from a fresh client with no explicit resume. Record success, latency, file and package present, and process alive (no gap in its timestamps beyond the paused interval).
5. After the run: read billed seconds from the provider's usage data. Idle compute must be 0.
6. Kill everything. Record total $.

Budget cap: $5 per provider. Results go to `runner/results/sandbox-eval-<date>/` as JSON, kind `measured`.

The tiers are decided separately. The workbench (persistent box) is decided by §2. Bare metal and GPU are chosen per experiment, as the cheapest machine passing R2 to R5 with the needed hardware, re-checked at that time against current prices.

## 6. Open (owner)
- Is about 40 active hours a month right?
- Design (a) memory-keeping, or (b) files-only, or score both?
- Accounts for Stage B, once Stage A names them.
- Is the Cloudflare account on Workers Paid?
- `keys-probe.yml` is not on the exagium_act allow-list, so key validity is still unknown.

## 7. Owner answers (2026-11-26, relayed by Claude Opus 5.5 in the claude.ai chat; dated entry per the rule above)
- **Use:** about 5 to 6 hours a day at minimum, so roughly 150 to 180 active hours a month. That replaces the
  assumed 40 h in the cost criterion (section 2). Weights are unchanged; only the usage assumption changes.
- **Tiers and places:** a tiered setup is fine, and so are machines from different providers and locations. The
  decision may pick one provider per tier, not one provider overall.
- **Resume:** the owner leans to (a), processes keep running ("by my gut… as long as it doesn't require rebuilding the
  environment and sessions work"). So both are scored, and a process surviving resume counts in the resume criterion
  as section 2 already says; files-only providers stay in the field if they never need a rebuild.
- **Cloudflare:** the account is not on Workers Paid now; the owner is willing to pay the $5/mo if the data favours it.
- First measurement started: `sandbox-eval.yml` runs section 5 on E2B (the keyed candidate) with the key checked live
  (keys-probe 2026-11-26: HTTP 200). This measures E2B; it does not choose it. Other Stage B candidates follow Stage A.

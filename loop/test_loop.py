"""Tests for loop.py. Toy candidates only: no test-set scores are computed here."""
import contextlib, io, json, os, shutil, subprocess, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop  # noqa: E402

CONST = """#include <stdio.h>
#include <stdlib.h>
int main(int c, char **v){ long n = atol(v[3]); FILE *f = fopen(v[1], "rb"); if(!f) return 2; int k = fgetc(f);
  for(long i=0;i<n;i++) putchar(k); putchar('\\n'); return 0; }
"""
TRAIN = "import sys\nopen(sys.argv[2], 'wb').write(b'%s')\n"


def make(root, name, weights=b"7", train=None, infer=CONST):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "train.py"), "w") as f:
        f.write(train if train is not None else TRAIN % weights.decode())
    with open(os.path.join(d, "infer.c"), "w") as f:
        f.write(infer)
    with open(os.path.join(d, "weights.bin"), "wb") as f:
        f.write(weights)
    return d


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = loop.main(list(argv))
        except SystemExit as e:
            code = e.code
    return code, out.getvalue() + err.getvalue()


def jail_mode():
    loop.check_jail()
    return loop.JAIL["mode"]


MODE = jail_mode()  # "bwrap", or "confine" where bwrap can't run (or EXAGIUM_JAIL=confine)


class Loop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_candidate_must_hold_exactly_its_files(self):
        d = make(self.tmp, "a")
        open(os.path.join(d, "test.csv"), "w").close()
        code, msg = run("freeze", d)
        self.assertEqual(code, 1)
        self.assertIn("extra ['test.csv']", msg)

    def test_symlinked_file_is_refused(self):
        d = make(self.tmp, "a")
        os.remove(os.path.join(d, "weights.bin"))
        os.symlink(loop.TASK["test"], os.path.join(d, "weights.bin"))
        self.assertEqual(run("freeze", d)[0], 1)

    def test_freeze_covers_task_harness_counter_and_files(self):
        d = make(self.tmp, "a")
        code, msg = run("freeze", d)
        self.assertEqual(code, 0)
        names = [x["name"] for x in json.loads(msg)]
        for want in ("loop/task/optdigits.tra", "loop/task/optdigits.tes", "loop/loop.py", "counter/count.py"):
            self.assertIn(want, names)
        self.assertEqual(sum(n.endswith(("train.py", "infer.c", "weights.bin")) for n in names), 3)

    def test_reproduce_identical_and_different(self):
        same = make(self.tmp, "same", b"7")
        other = make(self.tmp, "other", b"7", train=TRAIN % "8")
        self.assertEqual(run("reproduce", same)[0], 0)
        code, msg = run("reproduce", other)
        self.assertEqual(code, 1)
        self.assertIn("DIFFERENT", msg)

    def test_training_sees_only_the_training_file(self):
        probe = ("import hashlib, os, sys\nok = sorted(os.listdir('.')) == ['train.csv', 'train.py']"
                 " and hashlib.sha256(open('train.csv', 'rb').read()).hexdigest() == %r\n"
                 "open(sys.argv[2], 'wb').write(b'1' if ok else b'0')\n" % loop.sha256_file(loop.TASK["train"]))
        d = make(self.tmp, "p", b"1", train=probe)
        self.assertEqual(run("reproduce", d)[0], 0)

    def test_training_jail_has_no_network_repo_or_frozen_weights(self):
        probe = ("import os, socket, sys\nseen = []\n"
                 "try:\n socket.create_connection(('1.1.1.1', 443), timeout=3); seen.append('net')\nexcept OSError:\n pass\n"
                 "for p in (%r, %r, %r, '/home'):\n try:\n  os.listdir(p) if os.path.isdir(p) else open(p, 'rb').read(1)\n  seen.append(p)\n except OSError:\n  pass\n"
                 "try:\n pids = os.listdir('/proc')\nexcept OSError:\n pids = []\n"
                 "for pid in pids:\n if pid.isdigit():\n  try:\n   if os.listdir('/proc/%%s/cwd' %% pid) not in ([], ['train.csv', 'train.py'], ['train.py', 'train.csv']): seen.append(pid)\n  except OSError:\n   pass\n"
                 "open(sys.argv[2], 'wb').write(repr(seen).encode())\n") % (loop.REPO, loop.TASK["test"], os.path.dirname(os.path.abspath(__file__)))
        d = make(self.tmp, "n", b"[]", train=probe)
        code, msg = run("reproduce", d)
        self.assertEqual(code, 0, msg)

    def test_test_split_needs_a_plan(self):
        d = make(self.tmp, "a")
        code, msg = run("evaluate", d, "--test", "test")
        self.assertEqual(code, 2)
        self.assertIn("needs --plan", msg)

    def test_test_split_refused_when_plan_does_not_freeze_these_files(self):
        d = make(self.tmp, "a")
        code, msg = run("evaluate", d, "--test", "test", "--plan", "1")  # entry 1 is the record's plan
        self.assertEqual(code, 1)
        self.assertIn("does not freeze", msg)

    def test_malformed_output_fails(self):
        bad = CONST.replace("putchar('\\n');", "")
        d = make(self.tmp, "bad", infer=bad)
        code, msg = run("evaluate", d, "--rounds", "2", "--records", os.path.join(self.tmp, "r"))
        self.assertEqual(code, 1)
        self.assertIn("exactly", msg)

    def test_evaluate_constant_classifier_on_dev(self):
        d = make(self.tmp, "seven", b"7")
        out = os.path.join(self.tmp, "s.json")
        code, msg = run("evaluate", d, "--rounds", "2", "--records", os.path.join(self.tmp, "r"), "--out", out)
        self.assertEqual(code, 0, msg)
        with open(out) as fh:
            s = json.load(fh)
        c = s["candidates"][0]
        dev = loop.split("dev")
        self.assertEqual(c["correct"], sum(r[64] == 7 for r in dev))
        self.assertGreater(c["simulated_Ir_per_image"], 0)
        self.assertEqual(len(c["counter_records"]), 2)
        self.assertEqual(code, 0)
        self.assertEqual(run("evaluate", d, "--rounds", "2", "--records",
                             os.path.join(self.tmp, "r"), "--out", out)[0], 3)  # never overwrite a scoreboard

    def test_timing_run_must_agree_with_counted_predictions(self):
        # answers depend on row order (first row's first pixel): shuffled timing runs disagree with the counted run
        flaky = CONST.replace("for(long i=0;i<n;i++) putchar(k);",
                              "FILE *x = fopen(v[2], \"rb\"); if (x) fseek(x, 20, SEEK_SET); int p = x ? fgetc(x) : 0; if (n > 0 && p > 3) k = '3';"
                              " for(long i=0;i<n;i++) putchar(k);")
        d = make(self.tmp, "flaky", infer=flaky)
        code, msg = run("evaluate", d, "--rounds", "3", "--records", os.path.join(self.tmp, "r"))
        self.assertEqual(code, 1)
        self.assertIn("timing run's predictions differ", msg)

    def test_inference_jail_hides_repo_task_readme_and_earlier_runs(self):
        hostile = r"""#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
int main(int c, char **v){ long n = atol(v[3]); int k = '0';
  if (fopen(%s, "r") || fopen("README.md", "r") || access("/tmp/exagium-marker", F_OK) == 0) k = '9';
  if (syscall(SYS_keyctl, 10, -3, "user", "exagium", 0) >= 0) k = '9';  /* a key left by an earlier run */
  else syscall(SYS_add_key, "user", "exagium", "x", 1, -3);
  FILE *w = fopen(%s, "w"); if (w) { k = '9'; fclose(w); }
  FILE *m = fopen("/tmp/exagium-marker", "w"); if (m) fclose(m);
  for(long i=0;i<n;i++) putchar(k); putchar(10); return 0; }
""" % (json.dumps(loop.TASK["test"]), json.dumps(os.path.join(loop.REPO, "pwned")))
        d = make(self.tmp, "hostile", infer=hostile)
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write("x")
        out = os.path.join(self.tmp, "h.json")
        code, msg = run("evaluate", d, "--rounds", "2", "--records", os.path.join(self.tmp, "r"), "--out", out)
        self.assertFalse(os.path.exists(os.path.join(loop.REPO, "pwned")))
        with open(out) as fh:
            s = json.load(fh)
        # every native run printed '0': repo, task, README and earlier runs were out of reach
        self.assertEqual(s["candidates"][0]["correct"], sum(r[64] == 0 for r in loop.split("dev")))
        if MODE == "bwrap":
            # the counter's two reps share one jail, so the marker reaches rep 2: that shows up as reps that differ
            self.assertEqual(code, 1)
            self.assertTrue(s["problems"] and all("counter exit 1 (exact=False" in p for p in s["problems"]), s["problems"])
        else:
            # confine: no writable /tmp outside the sandbox, so not even the counter's reps can pass a marker
            self.assertEqual(code, 0, msg)
            self.assertEqual(s["jail"], "confine")

    def test_n0_must_print_only_a_newline_and_startup_is_counted(self):
        pad = CONST.replace("for(long i=0;i<n;i++) putchar(k);", "if (n == 0) putchar('7'); for(long i=0;i<n;i++) putchar(k);")
        d = make(self.tmp, "pad", infer=pad)
        code, msg = run("evaluate", d, "--rounds", "2", "--records", os.path.join(self.tmp, "r"))
        self.assertEqual(code, 1)
        self.assertRegex(msg, "N = 0 it must print only a newline|exactly 0 digits")

    @unittest.skipUnless(len(os.sched_getaffinity(0)) > 1, "needs two CPUs")
    def test_more_than_one_cpu_is_refused(self):
        threads = r"""#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static void *spin(void *a) { struct timespec t0, t; clock_gettime(CLOCK_MONOTONIC, &t0);
  do clock_gettime(CLOCK_MONOTONIC, &t); while ((t.tv_sec - t0.tv_sec) * 1000000000L + t.tv_nsec - t0.tv_nsec < 100000000L);
  return a; }
int main(int c, char **v){ long n = atol(v[3]); cpu_set_t all; CPU_ZERO(&all);
  for (int i = 0; i < 64; i++) CPU_SET(i, &all);
  sched_setaffinity(0, sizeof all, &all);  /* undo the timer's pin */
  char k = '1';
  if (n > 0) { pthread_t t; if (pthread_create(&t, 0, spin, 0) == 0) { spin(0); pthread_join(t, 0); } else k = '0'; }
  for(long i=0;i<n;i++) putchar(k); putchar(10); return 0; }
"""
        d = make(self.tmp, "threads", infer=threads)
        out = os.path.join(self.tmp, "t.json")
        code, msg = run("evaluate", d, "--rounds", "2", "--records", os.path.join(self.tmp, "r"), "--out", out)
        if MODE == "bwrap":
            self.assertEqual(code, 1)
            self.assertRegex(msg, "more than one CPU|more process|--single-process allows none")
        else:  # confine: the thread is never created ('0' printed every time), so a second CPU can't be used
            with open(out) as fh:
                s = json.load(fh)
            self.assertEqual(s["candidates"][0]["correct"], sum(r[64] == 0 for r in loop.split("dev")), msg)

    @unittest.skipUnless(MODE == "confine", "the fallback's own channels (bwrap isolates these by namespace)")
    def test_confine_refuses_ipc_and_metadata_channels(self):
        """SysV IPC, POSIX queues and file metadata outlive a run and aren't Landlock's; a parent's rlimits, nice
        value, affinity and scheduler are shared with the harness. All must be refused; the own process's must not."""
        target = os.path.join(self.tmp, "harness-owned.txt")
        open(target, "w").write("x")
        before = os.stat(target)
        hostile = r"""#define _GNU_SOURCE
#include <fcntl.h>
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <sys/ipc.h>
#include <sys/resource.h>
#include <sys/msg.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>
int main(int c, char **v){ long n = atol(v[3]); int k = '0'; const char *t = %s;
  if (shmget(0x6578, 4096, IPC_CREAT | 0600) >= 0) k = '9';
  if (semget(0x6578, 1, IPC_CREAT | 0600) >= 0) k = '9';
  if (msgget(0x6578, IPC_CREAT | 0600) >= 0) k = '9';
  if (mq_open("/exagium", O_CREAT | O_RDWR, 0600, 0) != (mqd_t)-1) k = '9';
  if (setxattr(t, "user.cache", "1", 1, 0) == 0) k = '9';
  if (chmod(t, 0777) == 0) k = '9';
  if (utimensat(AT_FDCWD, t, 0, 0) == 0) k = '9';
  pid_t pp = getppid(); struct rlimit rl = {0, 0}; cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(0, &cs);
  struct sched_param sp = {0};
  if (prlimit(pp, RLIMIT_CORE, &rl, 0) == 0) k = '9';
  if (setpriority(PRIO_PROCESS, pp, 19) == 0) k = '9';
  if (sched_setaffinity(pp, sizeof cs, &cs) == 0) k = '9';
  if (sched_setscheduler(pp, SCHED_BATCH, &sp) == 0) k = '9';
  if (prlimit(0, RLIMIT_CORE, &rl, 0) != 0) k = '8';
  for(long i=0;i<n;i++) putchar(k); putchar(10); return 0; }
""" % json.dumps(target)
        d = make(self.tmp, "channels", infer=hostile)
        out = os.path.join(self.tmp, "c.json")
        code, msg = run("evaluate", d, "--rounds", "2", "--records", os.path.join(self.tmp, "r"), "--out", out)
        with open(out) as fh:
            s = json.load(fh)
        self.assertEqual(s["candidates"][0]["correct"], sum(r[64] == 0 for r in loop.split("dev")), msg)
        after = os.stat(target)
        self.assertEqual((before.st_mode, before.st_mtime_ns), (after.st_mode, after.st_mtime_ns))
        with self.assertRaises(OSError):
            os.getxattr(target, "user.cache")

    def test_carve_hides_files_even_through_symlinks(self):
        """confine mode: the allow-list must leave hidden files unreachable, including via a symlinked parent."""
        import loop
        r = os.path.join(self.tmp, "root")
        os.makedirs(os.path.join(r, "real", "pkg", "data"))
        for f in ("real/pkg/data/digits.csv.gz", "real/pkg/data/other.txt", "real/pkg/mod.py", "real/top.txt"):
            open(os.path.join(r, f), "w").write("x")
        os.symlink("real", os.path.join(r, "alias"))
        os.symlink("real/top.txt", os.path.join(r, "top-link"))
        hidden = os.path.join(r, "alias", "pkg", "data", "digits.csv.gz")
        read, listed = loop.carve([r], [hidden])
        self.assertNotIn(os.path.join(r, "alias"), read)  # would grant the whole real tree
        self.assertIn(os.path.join(r, "top-link"), read)  # a symlink that leads nowhere hidden is fine
        self.assertIn(os.path.join(r, "real", "pkg", "data", "other.txt"), read)
        self.assertFalse(any(p == os.path.realpath(hidden) or os.path.realpath(hidden).startswith(p + "/") for p in read))
        self.assertIn(os.path.join(r, "real", "pkg", "data"), listed)

    def test_cost_disagreements_list_opposite_meters_once(self):
        """System review 2026-11-11: fewer instructions but slower past 2 SE must be reported, once per pair."""
        import loop
        pairs = [{"a": "x", "b": "y", "Ir_ratio": 0.4, "ns_per_image_diff": 50.0, "ns_per_image_diff_se": 8.0},
                 {"a": "y", "b": "x", "Ir_ratio": 2.5, "ns_per_image_diff": -50.0, "ns_per_image_diff_se": 8.0},
                 {"a": "x", "b": "z", "Ir_ratio": 0.4, "ns_per_image_diff": 10.0, "ns_per_image_diff_se": 8.0},
                 {"a": "z", "b": "y", "Ir_ratio": 0.9, "ns_per_image_diff": -30.0, "ns_per_image_diff_se": 5.0},
                 {"a": "w", "b": "y", "Ir_ratio": 0.9, "ns_per_image_diff": 30.0, "ns_per_image_diff_se": None}]
        d = loop.cost_disagreements(pairs)
        self.assertEqual([(x["fewer_instructions"], x["but_slower_than"]) for x in d], [("x", "y")])

    def test_a_hidden_fork_child_is_refused(self):
        """Review 3: a fork child does the work and is killed before it reports its counts (26x undercount)."""
        forker = r"""#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
int main(int c, char **v){ long n = atol(v[3]); char *out = malloc(n + 1); int p[2]; pipe(p); fflush(stdout);
  pid_t k = fork();
  if (k < 0) return 3;  /* fork refused (confine mode) */
  if (k == 0) { for (long i = 0; i < n; i++) out[i] = '1'; write(p[1], out, n); pause(); }
  long got = 0; while (got < n) { long r = read(p[0], out + got, n - got); if (r <= 0) return 2; got += r; }
  kill(k, SIGKILL); waitpid(k, 0, 0);
  fwrite(out, 1, n, stdout); putchar(10); return 0; }
"""
        d = make(self.tmp, "forker", infer=forker)
        code, msg = run("evaluate", d, "--rounds", "2", "--records", os.path.join(self.tmp, "r"))
        self.assertEqual(code, 1)
        if MODE == "bwrap":
            self.assertRegex(msg, "more process|--single-process allows none")
        else:  # confine: fork itself is refused, so the program gives up
            self.assertRegex(msg, "exited 3|counter failed")

    def test_jail_has_no_privilege(self):
        if MODE == "bwrap":
            r = loop.jail(["/bin/sh", "-c", "id -u; grep CapEff /proc/self/status"], [], 60, cwd="/")
            self.assertEqual(r.stdout.decode().split()[0], "65534")
            self.assertIn("CapEff:\t0000000000000000", r.stdout.decode())
        else:  # confine: same uid, but no new privileges and a seccomp filter in place
            code = "import ctypes; p = ctypes.CDLL(None).prctl; print(p(39, 0, 0, 0, 0), p(21, 0, 0, 0, 0))"
            r = loop.jail([sys.executable, "-I", "-c", code], [], 60, cwd="/")
            self.assertEqual(r.stdout.decode().split(), ["1", "2"], r.stderr.decode())

    def test_duplicate_names_refused(self):
        a = make(os.path.join(self.tmp, "x"), "same")
        b = make(os.path.join(self.tmp, "y"), "same")
        code, msg = run("evaluate", a, b, "--rounds", "2", "--records", os.path.join(self.tmp, "r"))
        self.assertEqual(code, 1)
        self.assertIn("unique", msg)


if __name__ == "__main__":
    unittest.main()

"""Tests for count.py. Each test that runs valgrind takes about a second.

    python3 -m unittest -v test_count
"""
import json, os, shutil, subprocess, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import count

HERE = os.path.dirname(os.path.abspath(__file__))
NOISY_C = r"""
#include <stdio.h>
#include <sys/random.h>
int main(void){ unsigned char b; if (getrandom(&b, 1, 0) != 1) return 1; volatile long s = 0;
  for (long i = 0; i < 1000 + b * 50; i++) s += i; printf("%ld\n", (long)s); return 0; }
"""
STOP_C = r"""
#include <stdio.h>
#include <valgrind/cachegrind.h>
int main(void){ CACHEGRIND_STOP_INSTRUMENTATION; volatile long s = 0;
  for (long i = 0; i < 3000000; i++) s += i; printf("%ld\n", (long)s); return 0; }
"""
CACHE_C = r"""
#include <stdio.h>
#include <stdlib.h>
int main(void){ FILE *f = fopen("answer.txt", "r"); long a = 0;
  if (f) { if (fscanf(f, "%ld", &a) != 1) a = 0; fclose(f); }
  else { volatile long s = 0; for (long i = 0; i < 3000000; i++) s += i; a = s;
         f = fopen("answer.txt", "w"); fprintf(f, "%ld\n", a); fclose(f); }
  printf("%ld\n", a); return 0; }
"""
EXEC_C = r"""
#include <unistd.h>
int main(void){ volatile long s = 0; for (long i = 0; i < 3000000; i++) s += i;
  execl("/bin/echo", "echo", "done", (char *)0); return 1; }
"""
HIDE_C = r"""
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
int main(void){ pid_t c = fork();
  if (c == 0) { volatile long s = 0; for (long i = 0; i < 3000000; i++) s += i; return 0; }
  waitpid(c, 0, 0); char pat[64]; snprintf(pat, sizeof pat, "/tmp/*/cg.%d", (int)c);
  glob_t g; if (glob(pat, 0, 0, &g) == 0) for (size_t i = 0; i < g.gl_pathc; i++) unlink(g.gl_pathv[i]);
  printf("done\n"); return 0; }
"""
RAND_OUT_C = r"""
#include <stdio.h>
#include <sys/random.h>
int main(void){ unsigned v = 0; if (getrandom(&v, sizeof v, 0) != sizeof v) return 1; printf("%u\n", v); return 0; }
"""
CHASE_C = r"""
#include <stdio.h>
#include <stdlib.h>
int main(int c, char **v){ size_t n = strtoull(v[1],0,10)/64; char *b = aligned_alloc(4096, n*64);
  for (size_t i = 0; i < n; i++) *(void**)(b + i*64) = b + ((i+1)%n)*64;
  void **p = (void**)b; for (size_t k = 0; k < 20*n; k++) p = (void**)*p; printf("%d\n", p==0); return 0; }
"""


@unittest.skipUnless(shutil.which("valgrind") and shutil.which("gcc"), "needs valgrind and gcc")
class Counter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp()
        cls.mlp = os.path.join(cls.td, "mlp")
        subprocess.run(["gcc", "-O2", "-o", cls.mlp, os.path.join(HERE, "workloads", "mlp.c")], check=True)
        progs = [("noisy", NOISY_C), ("chase", CHASE_C), ("cache", CACHE_C), ("execs", EXEC_C), ("randout", RAND_OUT_C), ("hide", HIDE_C)]
        if os.path.exists("/usr/include/valgrind/cachegrind.h"):
            progs.append(("stop", STOP_C))
        for name, src in progs:
            with open(os.path.join(cls.td, name + ".c"), "w") as f:
                f.write(src)
            subprocess.run(["gcc", "-O2", "-o", os.path.join(cls.td, name), os.path.join(cls.td, name + ".c")], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td)

    def out_dir(self):
        d = tempfile.mkdtemp(dir=self.td)
        return d

    def test_deterministic_program_is_exact(self):
        rec = count.count([self.mlp, "50"], reps=3)
        self.assertTrue(rec["verdict"]["exact"])
        self.assertTrue(rec["verdict"]["stdout_identical"])
        self.assertGreater(rec["summary"]["Ir"]["median"], 100000)
        self.assertEqual(rec["summary"]["Ir"]["rel_spread"], 0)

    def test_nondeterministic_program_is_flagged(self):
        rec = count.count([os.path.join(self.td, "noisy")], reps=4)
        self.assertFalse(rec["verdict"]["exact"])  # 4 reps of a random loop length: equal by chance ~1e-7
        self.assertGreater(rec["summary"]["Ir"]["rel_spread"], 0)

    def test_python_hash_seed_is_fixed(self):
        rec = count.count(["python3", "mlp.py", "3"], reps=2, input_paths=[os.path.join(HERE, "workloads", "mlp.py")])
        self.assertEqual(rec["env"]["PYTHONHASHSEED"], "0")
        self.assertTrue(rec["verdict"]["exact"], rec["summary"]["Ir"])

    def test_children_that_exec_are_refused(self):
        # a shell forks and execs its commands; a fork child's work before an exec would vanish (review 2)
        with self.assertRaises(count.Failure) as cm:
            count.count(["/bin/sh", "-c", f"{self.mlp} 50 > /dev/null; true"], reps=1)
        self.assertIn("exec", str(cm.exception))

    def test_reference_model_is_what_was_simulated(self):
        rec = count.count([self.mlp, "5"], reps=1)
        self.assertEqual(rec["cache"]["simulated"], count.REFERENCE)


    def test_caller_valgrind_settings_are_ignored(self):
        cwd = os.getcwd()
        d = self.out_dir()
        old = os.environ.get("VALGRIND_OPTS")
        try:
            os.chdir(d)
            with open(".valgrindrc", "w") as f:
                f.write("--LL=1048576,16,64\n")
            os.environ["VALGRIND_OPTS"] = "--D1=32768,8,64"
            rec = count.count([self.mlp, "5"], reps=1)
        finally:
            os.chdir(cwd)
            if old is None:
                os.environ.pop("VALGRIND_OPTS", None)
            else:
                os.environ["VALGRIND_OPTS"] = old
        self.assertEqual(rec["cache"]["simulated"], count.REFERENCE)
        self.assertNotIn("VALGRIND_OPTS", rec["env"])
        self.assertEqual(rec["env"]["HOME"], count.SANDBOX + "/home")

    def test_cache_model_changes_misses_not_instructions(self):
        # a 40 KiB cyclic walk: fits the 48 KiB reference D1, thrashes a 32 KiB LRU D1
        small = {"I1": [32768, 8, 64], "D1": [32768, 8, 64], "LL": [8388608, 16, 64]}
        a = count.count([os.path.join(self.td, "chase"), "40960"], reps=1)
        b = count.count([os.path.join(self.td, "chase"), "40960"], reps=1, cache=small)
        self.assertEqual(a["summary"]["Ir"]["median"], b["summary"]["Ir"]["median"])
        self.assertGreater(b["summary"]["D1mr"]["median"], 10 * a["summary"]["D1mr"]["median"])

    def test_records_are_never_overwritten(self):
        rec = count.count([self.mlp, "5"], reps=1)
        d = self.out_dir()
        p = count.write_record(rec, d)
        with self.assertRaises(FileExistsError):
            count.write_record(rec, d)
        with open(p) as f:
            self.assertEqual(json.load(f)["digest_sha256"], rec["digest_sha256"])

    def test_check_reproduces_and_catches_edits(self):
        rec = count.count(["./mlp", "20"], reps=2, input_paths=[self.mlp])
        d = self.out_dir()
        p = count.write_record(rec, d)
        self.assertEqual(count.check(p)[0], 0)
        # an edited count with a stale digest
        bad = json.loads(json.dumps(rec))
        bad["reps"][0]["counts"]["Ir"] -= 1000
        p2 = os.path.join(d, "edited.json")
        with open(p2, "w") as f:
            json.dump(bad, f)
        self.assertEqual(count.check(p2)[0], 1)
        # an edited count with a recomputed digest: self-consistent, so only re-counting catches it
        for r in bad["reps"]:
            r["counts"]["Ir"] = rec["reps"][0]["counts"]["Ir"] - 1000
        bad["summary"] = count.summarise(bad["reps"])
        bad.pop("digest_sha256")
        bad["digest_sha256"] = count.hashlib.sha256(count.canonical(bad)).hexdigest()
        p3 = os.path.join(d, "forged.json")
        with open(p3, "w") as f:
            json.dump(bad, f)
        code, msg = count.check(p3)
        self.assertEqual(code, 1, msg)
        self.assertIn("differ", msg)
        # a malformed record with a valid digest is a failed check, never a crash
        bad = {"schema": count.SCHEMA, "reps": [{"counts": None}], "verdict": {"reps": 1}}
        bad["digest_sha256"] = count.hashlib.sha256(count.canonical(bad)).hexdigest()
        p4 = os.path.join(d, "malformed.json")
        with open(p4, "w") as f:
            json.dump(bad, f)
        code, msg = count.check(p4)
        self.assertEqual(code, 1, msg)
        self.assertIn("malformed", msg)

    def test_check_refuses_a_changed_program(self):
        prog = os.path.join(self.td, "mlp_copy")
        shutil.copy(self.mlp, prog)
        rec = count.count([prog, "5"], reps=1)
        p = count.write_record(rec, self.out_dir())
        with open(prog, "ab") as f:
            f.write(b"\0")
        self.assertEqual(count.check(p)[0], 2)

    def test_failing_command_raises(self):
        with self.assertRaises(count.Failure):
            count.count(["/bin/false"], reps=1)

    def test_work_before_exec_is_refused(self):
        # valgrind drops the counts of a process image that calls exec: that would be a ~1000x undercount
        with self.assertRaises(count.Failure) as cm:
            count.count([os.path.join(self.td, "execs")], reps=1)
        self.assertIn("exec", str(cm.exception))

    def test_a_child_deleting_its_counts_is_caught(self):
        # review 2 hid a child's work by deleting its count file and editing the log; the log is now a pipe
        with self.assertRaises(count.Failure) as cm:
            count.count([os.path.join(self.td, "hide")], reps=1)
        self.assertIn("count files", str(cm.exception))

    def test_fork_children_without_exec_are_counted(self):
        rec = count.count([sys.executable, "-c", "import os\nif os.fork() == 0:\n    sum(range(200000)); os._exit(0)\nos.wait()"], reps=2)
        self.assertEqual(rec["reps"][0]["processes"], 2)
        self.assertTrue(rec["verdict"]["exact"])

    def test_env_shebang_is_refused_clearly(self):
        script = os.path.join(self.td, "envscript")
        with open(script, "w") as f:
            f.write("#!/usr/bin/env python3\nprint(1)\n")
        os.chmod(script, 0o755)
        with self.assertRaises(count.Failure) as cm:
            count.count([script], reps=1)
        self.assertIn("env", str(cm.exception))

    def test_env_cannot_reach_valgrind_or_the_loader(self):
        for kv in ("VALGRIND_LIB=/tmp/x", "LD_PRELOAD=/tmp/x.so", "GLIBC_TUNABLES=a=b", "PATH=/x", "BAD-KEY=1"):
            with self.assertRaises(count.Failure, msg=kv):
                count.count([self.mlp, "5"], reps=1, extra_env=[kv])

    def test_output_that_differs_between_reps_is_not_a_pass(self):
        code = count.main(["--reps", "2", "--out-dir", self.out_dir(), "--", os.path.join(self.td, "randout")])
        self.assertEqual(code, 1)

    def test_check_will_not_vouch_for_a_flagged_or_inexact_record(self):
        if not os.path.exists(os.path.join(self.td, "stop")):
            self.skipTest("needs valgrind headers")
        rec = count.count([os.path.join(self.td, "stop")], reps=1)
        p = count.write_record(rec, self.out_dir())
        self.assertEqual(count.check(p)[0], 2)
        # forging the flag away (digest recomputed) doesn't help: the re-count finds the code again
        rec["verdict"]["client_request_code"] = []
        rec.pop("digest_sha256")
        rec["digest_sha256"] = count.hashlib.sha256(count.canonical(rec)).hexdigest()
        p2 = count.write_record(rec, self.out_dir())
        self.assertEqual(count.check(p2)[0], 1)

    def test_relative_program_must_be_an_input(self):
        with self.assertRaises(count.Failure):
            count.count(["./mlp", "5"], reps=1)

    def test_no_cache_survives_between_reps_or_from_the_caller(self):
        prog = os.path.join(self.td, "cache")
        cwd = os.getcwd()
        os.chdir(self.td)
        try:
            with open("answer.txt", "w") as f:
                f.write("42\n")  # a precomputed answer next to the caller
            rec = count.count([prog], reps=3)
        finally:
            os.remove(os.path.join(self.td, "answer.txt"))
            os.chdir(cwd)
        self.assertTrue(rec["verdict"]["exact"])  # every rep did the full work
        self.assertGreater(rec["summary"]["Ir"]["min"], 3000000)

    def test_caller_home_and_cwd_do_not_change_counts(self):
        old_home, cwd = os.environ.get("HOME"), os.getcwd()
        try:
            a = count.count([self.mlp, "30"], reps=1)
            os.environ["HOME"] = "/a/much/longer/home/directory/than/before"
            os.chdir(self.out_dir())
            b = count.count([self.mlp, "30"], reps=1)
        finally:
            os.chdir(cwd)
            os.environ["HOME"] = old_home
        self.assertEqual(a["reps"][0]["counts"], b["reps"][0]["counts"])

    @unittest.skipUnless(os.path.exists("/usr/include/valgrind/cachegrind.h"), "needs valgrind headers")
    def test_code_that_switches_counting_off_is_flagged(self):
        rec = count.count([os.path.join(self.td, "stop")], reps=1)
        self.assertLess(rec["summary"]["Ir"]["median"], 3000000)  # the work really went uncounted
        self.assertTrue(rec["verdict"]["client_request_code"])
        self.assertEqual(count.main(["--reps", "1", "--out-dir", self.out_dir(), "--", os.path.join(self.td, "stop")]), 1)
        clean = count.count([self.mlp, "5"], reps=1)
        self.assertEqual(clean["verdict"]["client_request_code"], [])

    def test_parse_cache(self):
        self.assertEqual(count.parse_cache("reference"), count.REFERENCE)
        self.assertEqual(count.parse_cache("host"), "host")
        self.assertEqual(count.parse_cache("I1=1,2,3;D1=4,5,6;LL=7,8,9")["LL"], [7, 8, 9])
        with self.assertRaises(Exception):
            count.parse_cache("LL=7,8,9")


if __name__ == "__main__":
    unittest.main()

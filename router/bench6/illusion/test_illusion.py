"""Known-answer tests for the P28 Illusion probe's grading (python3 -m unittest router/bench6/illusion/test_illusion.py)."""
import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import illusion as I


class T(unittest.TestCase):
    def test_gold(self):
        p = "diff --git a/src/x.py b/src/x.py\n--- a\n+++ b\ndiff --git a/y/z.py b/y/z.py\n"
        self.assertEqual(I.gold(p), ["src/x.py", "y/z.py"])

    def test_parse_block(self):
        t = "thinking src/wrong.py\n```paths\n1. `./src/x.py`\n- a/y/z.py\n```"
        self.assertEqual(I.parse(t), ["src/x.py", "y/z.py"])

    def test_parse_fallback_and_none(self):
        self.assertEqual(I.parse("I think it is django/db/models/query.py, maybe."), ["django/db/models/query.py"])
        self.assertEqual(I.parse(None), [])
        self.assertEqual(I.parse("no idea"), [])

    def test_remove_file_paths(self):
        self.assertEqual(I.remove_file_paths("change src/x.py and y/z.py"), "change <path> and <path>")
        self.assertEqual(I.remove_file_paths("see README.md, e.g. foo"), "see <path>, e.g. foo")
        self.assertEqual(I.remove_file_paths(None), "")

    def test_probe_offline(self):
        d = tempfile.mkdtemp()
        mk = lambda i, g: {"id": i, "repo": "r", "split": "s", "created_at": "", "issue": f"task:{i} in src/x.py", "gold": g}
        s = {"fresh": {"tasks": [mk("f1", ["a.py"]), mk("f2", ["b.py"])]},
             "verified": {"tasks": [mk("v1", ["c.py"]), mk("v2", ["d.py"])]}}
        json.dump(s, open(os.path.join(d, "s.json"), "w"))
        answers = {"task:f1 in <path>": "```paths\na.py\n```",
                   "task:f2 in <path>": "```paths\nwrong.py\n```",
                   "task:v1 in <path>": "```paths\nc.py\n```",
                   "task:v2 in <path>": "```paths\nd.py\n```"}
        def stub(repo, issue):
            return answers[issue]
        r = I.probe(os.path.join(d, "s.json"), d, call=stub)
        records = r["records"]
        self.assertEqual(len(records), 4)
        self.assertAlmostEqual(r["match_rate"], 0.75)
        self.assertAlmostEqual(sum(x["match"] for x in records) / len(records), r["match_rate"])
        by = {x["task_id"]: x for x in records}
        self.assertTrue(by["f1"]["match"] and by["v1"]["match"] and by["v2"]["match"])
        self.assertFalse(by["f2"]["match"])
        self.assertEqual(by["f1"]["named_file"], "a.py")
        self.assertEqual(by["f2"]["gold_files"], ["b.py"])
        saved = json.load(open(os.path.join(d, "probe.json")))
        self.assertEqual(saved["match_rate"], r["match_rate"])
        self.assertEqual(len(saved["records"]), 4)

    def test_grade_known(self):
        d = tempfile.mkdtemp()
        mk = lambda i, g: {"id": i, "repo": "r", "split": "s", "created_at": "", "issue": "x", "gold": g}
        s = {"fresh": {"tasks": [mk("f1", ["a.py"]), mk("f2", ["b.py"])]},
             "verified": {"tasks": [mk("v1", ["c.py"]), mk("v2", ["d.py", "e.py"])]}}
        rep = {"fresh": {"f1": {"text": "```paths\nx.py\na.py\n```"}, "f2": {"text": "```paths\nb.py\n```"}},
               "verified": {"v1": {"text": "```paths\nc.py\n```"}, "v2": {"text": "```paths\ne.py\n```"}}}
        json.dump(s, open(os.path.join(d, "s.json"), "w")); json.dump(rep, open(os.path.join(d, "replies.json"), "w"))
        I.grade(os.path.join(d, "s.json"), d)
        r = json.load(open(os.path.join(d, "RESULT.json")))
        self.assertEqual((r["fresh"]["top1"], r["fresh"]["any3"]), (0.5, 1.0))
        self.assertEqual((r["verified"]["top1"], r["verified"]["any3"]), (1.0, 1.0))
        self.assertEqual(r["rule"]["gap_top1_points"], 50.0)
        self.assertIn("use it", r["rule"]["verdict"])


if __name__ == "__main__":
    unittest.main()

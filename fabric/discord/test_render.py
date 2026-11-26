import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render

STREAM = """{"type": "agent_runtime_init", "timestamp": 1000, "data": {"topicId": "tpc_X1", "agentId": "a"}}
{"type": "stream_start", "timestamp": 1001, "data": {"model": "gpt-oss-120b", "provider": "cerebras"}}
{"type": "stream_end", "timestamp": 1003, "data": {"finalContent": "", "usage": {"totalTokens": 900}}}
{"type": "tool_start", "timestamp": 1004, "data": {"payload": {"apiName": "queue_summary"}}}
{"type": "tool_end", "timestamp": 1005, "data": {"isSuccess": true, "executionTime": 800}}
{"type": "stream_end", "timestamp": 1008, "data": {"finalContent": "There are **42** open items.", "usage": {"totalTokens": 1300}}}
{"type": "agent_runtime_end", "timestamp": 1009, "data": {"reason": "done"}}
"""


class Render(unittest.TestCase):
    def test_summary(self):
        s = render.summarise(render.events(STREAM))
        self.assertEqual(s["answer"], "There are **42** open items.")
        self.assertEqual(s["topic"], "tpc_X1")
        self.assertEqual(s["tokens"], 2200)
        self.assertEqual(s["tools"], [{"name": "queue_summary", "ok": True, "ms": 800}])

    def test_card(self):
        e = render.card("free", render.summarise(render.events(STREAM)), 52, 0)
        self.assertIn("✓ queue_summary 0.8s", e["fields"][0]["value"])
        self.assertIn("topic tpc_X1", e["footer"]["text"])     # what "Reply to agent" reads back
        self.assertIn("gpt-oss-120b (cerebras)", e["footer"]["text"])
        long = render.card("free", {**render.summarise([]), "answer": "x\n" * 3000}, 1, 3)
        self.assertLessEqual(len(long["description"]), 4096)
        self.assertTrue(long["description"].startswith("⏳"))

    def test_bad_stream(self):
        self.assertEqual(render.events("not json at all"), [])


if __name__ == "__main__":
    unittest.main()

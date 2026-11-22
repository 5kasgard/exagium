import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breadth as B


class T(unittest.TestCase):
    def test_round_robin_dedupes_and_caps(self):
        a = [{"url": "1"}, {"url": "2"}, {"url": "3"}]
        b = [{"url": "2"}, {"url": "4"}, {"url": None}]
        self.assertEqual([x["url"] for x in B.round_robin([a, b], 10)], ["1", "2", "4", "3"])
        self.assertEqual(len(B.round_robin([a, b], 2)), 2)
        self.assertEqual(B.round_robin([], 5), [])


if __name__ == "__main__":
    unittest.main()

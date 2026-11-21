"""Every workflow file parses with no duplicate keys (2026-11-20: tier.yml defined 'actions' twice; yaml.safe_load took
the last value silently, GitHub refused the file, and the scheduled tier stopped running)."""
import glob, os, unittest
import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    keys = set()
    for k, _ in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in keys:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", k.start_mark)
        keys.add(key)
    return loader.construct_mapping(node, deep)


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)



class TestArtifactApart(unittest.TestCase):
    """Audit 2026-11-21 (B F2, R1 F2): model code runs in worker.yml's test job. Its artifact holds only test.json and is
    unpacked into its own folder, never over the draft job's draft.json and bundle.json."""
    def test_worker_test_artifact(self):
        w = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows", "worker.yml")))
        up = [s for s in w["jobs"]["test"]["steps"] if "upload-artifact" in str(s.get("uses"))]
        self.assertEqual([s["with"]["path"] for s in up], ["tested/test.json"])
        down = [s for s in w["jobs"]["land"]["steps"] if "download-artifact" in str(s.get("uses")) and s["with"]["name"] == "test"]
        self.assertEqual([s["with"]["path"] for s in down], ["tested"])
        self.assertIn("cp tested/test.json job/test.json", yaml.dump(w["jobs"]["land"]))
        self.assertIn("sums", w["jobs"]["draft"]["outputs"])                       # land checks the draft job's own hashes
        self.assertEqual(w["jobs"]["land"]["env"]["DRAFT_SUMS"], "${{ needs.draft.outputs.sums }}")
        self.assertIn('[ "$got" != "$DRAFT_SUMS" ]', open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows", "worker.yml")).read())


class Workflows(unittest.TestCase):
    def test_no_grep_count_or_echo(self):
        """'grep -c' prints 0 and exits 1 when nothing matches, so '|| echo 0' yields two lines (killed the idle job, 2026-11-20)."""
        import re
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows")
        for f in sorted(glob.glob(os.path.join(root, "*.yml"))):
            for n, line in enumerate(open(f).read().splitlines(), 1):
                self.assertFalse(re.search(r"grep -c[^|]*\|\|\s*echo", line), f"{os.path.basename(f)}:{n}: {line.strip()}")

    def test_no_duplicate_keys(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows")
        files = sorted(glob.glob(os.path.join(root, "*.yml")))
        self.assertTrue(files)
        for f in files:
            with self.subTest(f=os.path.basename(f)):
                yaml.load(open(f), Loader=UniqueKeyLoader)


if __name__ == "__main__":
    unittest.main()

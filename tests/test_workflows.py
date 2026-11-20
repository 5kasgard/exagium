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


class Workflows(unittest.TestCase):
    def test_no_duplicate_keys(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows")
        files = sorted(glob.glob(os.path.join(root, "*.yml")))
        self.assertTrue(files)
        for f in files:
            with self.subTest(f=os.path.basename(f)):
                yaml.load(open(f), Loader=UniqueKeyLoader)


if __name__ == "__main__":
    unittest.main()

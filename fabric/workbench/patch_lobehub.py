"""Local fixes to the pinned LobeHub checkout, applied before its build. Each one fails loudly if its target text has
changed, so a LobeHub upgrade cannot skip one silently.

1. Migration 0090 creates pg_search when pg_available_extensions lists it. New Neon projects list it and then refuse
   it ("extension pg_search is deprecated and no longer allowed", 4th workbench run), so the whole migration failed.
   The guard now also skips on that refusal. LobeHub's own note in the file says editing it is safe (Drizzle compares
   created_at only). Search runs on pg_like (FTS_SEARCH_PROVIDER), which LobeHub's Neon guide recommends."""
import sys
from pathlib import Path

OLD = "    CREATE EXTENSION IF NOT EXISTS pg_search;\n"
NEW = ("    BEGIN\n      CREATE EXTENSION IF NOT EXISTS pg_search;\n    EXCEPTION WHEN OTHERS THEN\n"
       "      RAISE NOTICE 'pg_search listed but refused (%); skipping BM25 search setup', SQLERRM;\n    END;\n")


def patch(root):
    f = Path(root) / "packages/database/migrations/0090_enable_pg_search.sql"
    s = f.read_text()
    if NEW in s:
        return "0090: already patched"
    if s.count(OLD) != 1:
        sys.exit("0090: target text changed upstream; review the pg_search guard before building")
    f.write_text(s.replace(OLD, NEW))
    return "0090: pg_search refusal now skipped"


if __name__ == "__main__":
    print(patch(sys.argv[1]))

#!/usr/bin/env python
"""Quick search test: uv run python search.py "din søgning" """

import sys
from src.embedder import Embedder
from src.qdrant import get_client

query = sys.argv[1] if len(sys.argv) > 1 else "sundhedsreform"

embedder = Embedder()
client = get_client()

emb = embedder.embed_single(query)
results = client.query_points(
    collection_name="dagsordener",
    query=emb["dense"],
    using="dense",
    limit=5,
)

print(f'\n"{query}"\n')
for r in results.points:
    p = r.payload or {}
    title = str(p.get("title", ""))[:60]
    udvalg = p.get("udvalg", "")
    dt = str(p.get("datetime", ""))[:10]
    print(f"{r.score:.2f}  {title}")
    print(f"      {udvalg} - {dt}")
    print()
    print(f"{p['content_md']}")
    break
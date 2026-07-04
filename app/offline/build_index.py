import argparse
import asyncio

from app.offline.local_index_builder import LocalIndexBuilder
from app.storage.local_artifact_store import LocalArtifactStore
from app.storage.local_doc_store import LocalDocStore, load_docs_from_json_or_jsonl


async def run() -> None:
    parser = argparse.ArgumentParser(description="Build local BM25 and FAISS artifacts.")
    parser.add_argument("--docs", default=None, help="SourceDoc JSON/JSONL path. Defaults to raw store.")
    parser.add_argument("--artifact-dir", default=None, help="Output artifact directory.")
    args = parser.parse_args()

    if args.docs:
        docs = load_docs_from_json_or_jsonl(args.docs)
    else:
        docs = LocalDocStore().load_all()

    builder = LocalIndexBuilder(artifact_store=LocalArtifactStore(args.artifact_dir))
    result = await builder.build(docs)
    print(
        "built_local_index "
        f"doc_count={result.doc_count} "
        f"embedding_dim={result.embedding_dim} "
        f"artifact_dir={result.artifact_dir}"
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

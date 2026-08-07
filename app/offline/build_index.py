import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.offline.local_index_builder import LocalIndexBuilder
from app.source_registry import SourceRegistry
from app.storage.local_artifact_store import LocalArtifactStore
from app.storage.local_doc_store import iter_docs_from_jsonl


async def run() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build one source with the original single-index build path."
    )
    parser.add_argument(
        "--docs",
        default=settings.LOCAL_RAW_DOCS_PATH,
        help="Mixed SourceDoc JSONL path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source id to build. The same mixed JSONL can be scanned once per source.",
    )
    parser.add_argument(
        "--source-config",
        default=None,
        help="Source registry JSON path. Defaults to LOCAL_SOURCE_CONFIG_PATH.",
    )
    parser.add_argument(
        "--index-es",
        action="store_true",
        help="Destructively rebuild this source's Elasticsearch index.",
    )
    args = parser.parse_args()

    registry = SourceRegistry.from_path(
        args.source_config or settings.LOCAL_SOURCE_CONFIG_PATH
    )
    source = registry.require(args.source)

    # Only this wrapper is multi-source aware: scan the mixed file and collect
    # the selected source. The actual build below is the original single-index flow.
    docs = list(iter_docs_from_jsonl(args.docs, source_id=source.source_id))
    if not docs:
        raise ValueError(f"No documents found for source_id {source.source_id!r}")

    artifact_dir = Path(source.faiss_index_path).parent
    builder = LocalIndexBuilder(
        source=source,
        artifact_store=LocalArtifactStore(artifact_dir=artifact_dir),
        index_elasticsearch=args.index_es,
    )
    result = await builder.build(docs)
    print(
        "built_local_source_index "
        f"source={source.source_id} "
        f"doc_count={result.doc_count} "
        f"embedding_dim={result.embedding_dim} "
        f"artifact_dir={result.artifact_dir}"
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

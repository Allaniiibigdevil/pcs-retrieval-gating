import argparse

from app.storage.local_doc_store import LocalDocStore, load_docs_from_json_or_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SourceDoc JSON/JSONL into local raw store.")
    parser.add_argument("input", help="Path to SourceDoc JSON or JSONL file.")
    parser.add_argument("--output", default=None, help="Output raw docs JSONL path.")
    args = parser.parse_args()

    docs = load_docs_from_json_or_jsonl(args.input)
    store = LocalDocStore(args.output)
    store.append_many(docs)
    print(f"ingested_docs={len(docs)} output={store.path}")


if __name__ == "__main__":
    main()

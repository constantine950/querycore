import json
import re
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "data" / "sample_documents.json"


def load_documents(path: Path = DATASET_PATH) -> list[dict]:
    """Load and return documents from the JSON dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_document(doc: dict) -> list[str]:
    """Return a list of validation errors for a document, empty if valid."""
    errors = []
    required = ["id", "title", "body", "category", "date"]
    for field in required:
        if field not in doc:
            errors.append(f"Missing field: {field}")
        elif not doc[field]:
            errors.append(f"Empty field: {field}")
    if "body" in doc and len(doc["body"].split()) < 20:
        errors.append("Body too short (< 20 words)")
    return errors


def summarise(docs: list[dict]) -> None:
    print(f"\n{'='*50}")
    print(f"  Dataset Summary")
    print(f"{'='*50}")
    print(f"  Total documents : {len(docs)}")

    total_words = sum(len(d["body"].split()) for d in docs)
    print(f"  Total words     : {total_words:,}")
    print(f"  Avg words/doc   : {total_words // len(docs)}")

    cats: dict[str, int] = {}
    for d in docs:
        cats[d["category"]] = cats.get(d["category"], 0) + 1
    print(f"\n  By category:")
    for cat, count in sorted(cats.items()):
        print(f"    {cat:<25} {count} docs")

    print(f"\n  Sample document:")
    sample = docs[0]
    print(f"    id       : {sample['id']}")
    print(f"    title    : {sample['title']}")
    print(f"    category : {sample['category']}")
    print(f"    words    : {sample['word_count']}")
    print(f"    body     : {sample['body'][:120]}...")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    docs = load_documents()

    # Validate all documents
    errors_found = False
    for doc in docs:
        errs = validate_document(doc)
        if errs:
            print(f"[INVALID] {doc.get('id', '?')}: {errs}")
            errors_found = True

    if not errors_found:
        print("[OK] All documents passed validation.")

    # Clean text in place
    for doc in docs:
        doc["body"] = clean_text(doc["body"])

    summarise(docs)

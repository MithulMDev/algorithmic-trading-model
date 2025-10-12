#!/usr/bin/env python3
import sys
import argparse
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pprint import pprint


def human_int(n: int) -> str:
    return f"{n:,}"

def main():
    parser = argparse.ArgumentParser(description="Check if data exists in a MongoDB collection.")
    parser.add_argument("--uri", default="mongodb://localhost:27017/", help="MongoDB URI")
    parser.add_argument("--db", required=True, help="Database name")
    parser.add_argument("--coll", required=True, help="Collection name")
    parser.add_argument("--filter", default=None, help='Optional JSON-like filter, e.g. \'{"AAPL.csv":{"$ne":null}}\'')
    parser.add_argument("--samples", type=int, default=3, help="How many sample docs to print")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="Server selection timeout (ms)")
    args = parser.parse_args()

    # Very minimal/unsafe parser for filter—use with care or replace with json.loads if you pass valid JSON.
    import json
    filter_dict = {}
    if args.filter:
        try:
            filter_dict = json.loads(args.filter)
        except json.JSONDecodeError:
            print("ERROR: --filter must be valid JSON. Example: --filter '{\"AAPL.csv\":{\"$ne\":null}}'")
            sys.exit(2)

    try:
        client = MongoClient(args.uri, serverSelectionTimeoutMS=args.timeout_ms)
        # Trigger server selection
        client.admin.command("ping")
    except PyMongoError as e:
        print(f"❌ Cannot connect to MongoDB: {e}")
        sys.exit(1)

    try:
        db = client[args.db]
        coll_names = db.list_collection_names()
        if args.coll not in coll_names:
            print(f"ℹ️ Collection '{args.coll}' does not exist in database '{args.db}'.")
            sys.exit(0)

        coll = db[args.coll]

        from pprint import pprint

        print("\n🧾 Two sample documents:")
        for i, doc in enumerate(coll.find().limit(2), start=1):
            print(f"\n--- Document {i} ---")
            pprint(doc)

        # Count documents
        count = coll.count_documents(filter_dict)
        print(f"✅ Connected to {args.uri}")
        print(f"📚 Database: {args.db}")
        print(f"📦 Collection: {args.coll}")
        if filter_dict:
            print(f"🔎 With filter: {filter_dict}")
        print(f"🧮 Document count: {human_int(count)}")

        if count == 0:
            print("⚠️ No documents found.")
            sys.exit(0)

        # Show earliest & latest timestamps if field exists
        has_ts = coll.find_one({**filter_dict, "timestamp": {"$exists": True}}, {"timestamp": 1})
        if has_ts and "timestamp" in has_ts:
            first = coll.find(filter_dict, {"timestamp": 1}).sort("timestamp", 1).limit(1)
            last = coll.find(filter_dict, {"timestamp": 1}).sort("timestamp", -1).limit(1)
            first_ts = next(first, {}).get("timestamp")
            last_ts = next(last, {}).get("timestamp")
            print("⏱️ Timestamp range:", first_ts, "→", last_ts)

        # Print a few sample docs (pretty)
        print(f"\n🧪 Sample {min(args.samples, count)} document(s):")
        import pprint
        cursor = coll.find(filter_dict).limit(args.samples)
        for i, doc in enumerate(cursor, 1):
            # Avoid dumping huge docs: trim to top-level keys + nested key counts
            summary = {}
            for k, v in doc.items():
                if k == "_id":
                    summary["_id"] = str(v)
                elif isinstance(v, dict):
                    summary[k] = {"_summary": f"{len(v)} keys", **{kk: v[kk] for kk in list(v.keys())[:3]}}
                else:
                    summary[k] = v
            print(f"\n--- Document {i} ---")
            pprint.pprint(summary)

        # Suggest an index if timestamp exists and there is no index yet
        if has_ts:
            idx_info = list(coll.list_indexes())
            has_ts_idx = any("timestamp" in (idx.get("key") or {}) for idx in idx_info)
            if not has_ts_idx:
                print("\n💡 Tip: Consider adding an index for faster time-range queries:")
                print(f'    db.getSiblingDB("{args.db}").{args.coll}.createIndex({{ timestamp: 1 }})')

        print("\nDone.")
    except PyMongoError as e:
        print(f"❌ MongoDB operation failed: {e}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    main()

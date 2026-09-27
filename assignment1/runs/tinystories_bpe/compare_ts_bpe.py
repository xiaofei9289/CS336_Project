import pickle

a = pickle.load(open(r"D:\desktop\data\tinystories\tokenizer_train_10k.pkl", "rb"))
b = pickle.load(open(r"D:\desktop\data\tinystories\tokenizer_train_10k_optimize.pkl", "rb"))

print("vocab_equal", a["vocab"] == b["vocab"])
print("merges_equal", a["merges"] == b["merges"])
print("special_equal", a["special_tokens"] == b["special_tokens"])
print("vocab_len", len(a["vocab"]), len(b["vocab"]))
print("merges_len", len(a["merges"]), len(b["merges"]))

ma, mb = a["metadata"], b["metadata"]
for key in [
    "elapsed_seconds",
    "peak_rss_mib",
    "longest_token_id",
    "longest_ordinary_token_id",
    "input_path",
    "input_size_bytes",
    "vocab_size",
]:
    print(key, ma.get(key), "|", mb.get(key))
print("longest_bytes_equal", ma.get("longest_token_bytes") == mb.get("longest_token_bytes"))
print("longest_bytes", ma.get("longest_token_bytes"))

if a["merges"] != b["merges"]:
    for i, (x, y) in enumerate(zip(a["merges"], b["merges"])):
        if x != y:
            print("first_merge_diff", i, x, y)
            break
    print("merge_len_diff", len(a["merges"]) - len(b["merges"]))

if a["vocab"] != b["vocab"]:
    diffs = 0
    for i in sorted(set(a["vocab"]) | set(b["vocab"])):
        if a["vocab"].get(i) != b["vocab"].get(i):
            diffs += 1
            if diffs <= 5:
                print("vocab_diff_id", i, a["vocab"].get(i), b["vocab"].get(i))
    print("vocab_diff_count", diffs)

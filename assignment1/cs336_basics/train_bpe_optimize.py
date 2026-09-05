import os
import time
from collections import Counter
from multiprocessing import Pool

from .pretokenization_example import find_chunk_boundaries
# from .bpe import (
#     build_pair_heap,
#     build_pair_index,
#     initialize_vocab,
#     merge_incrementally,
#     pretokenize,
#     push_pair_heap_updates,
#     select_best_pair_from_heap,
# )
from .bpe_optimize import (
    build_pair_heap,
    build_pair_index,
    initialize_vocab,
    merge_incrementally,
    pretokenize,
    push_pair_heap_updates,
    select_best_pair_from_heap,
)

# 训练 byte-level BPE Tokenizer 的步骤

# 步骤一：词表初始化

# 步骤二：合并 pair

# 步骤三：添加新 token

# 步骤四：重复步骤二和步骤三，直到词表大小达到目标大小

# 步骤五：保存词表  


ChunkTask = tuple[
    str | bytes,
    int,
    int,
    tuple[str, ...],
]


def _pretokenize_chunk(
    task: ChunkTask,
) -> Counter[tuple[int, ...]]:
    """在子进程中读取并预分词一个 ``[start, end)`` 文件区间。"""
    input_path, start, end, special_tokens = task

    # worker 只接收路径和整数偏移；原始 bytes 和解码后的 str 都只存在
    # 于当前 worker 内，不通过进程间通信传递。
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_text = file.read(end - start).decode("utf-8")

    return pretokenize(
        text=chunk_text,
        special_tokens=list(special_tokens),
    )


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    """在给定语料上训练 byte-level BPE tokenizer。

    Returns:
        ``vocab``：token ID 到 token bytes 的映射。
        ``merges``：按训练先后排列的 byte pair。

    ``desired_num_chunks`` 控制语料块数，默认 32；``num_workers`` 控制
    pretok 进程数，默认不超过 32。二者互相独立，merge 阶段仍在主进程
    中单线程执行。

    Windows 使用 spawn，调用方必须在 ``if __name__ == "__main__":``
    保护下调用本函数。
    """
    # 阶段一：初始化词表并把语料转换为 pre-token 频次表。
    # 1. 初始化基础词表和特殊token
    vocab = initialize_vocab(
        vocab_size = vocab_size,
        special_tokens = special_tokens,
    )

    # 2. 流式读取并预分词，避免把整份语料一次性解码成巨大的 str。
    # 只有在 <|endoftext|> 也被声明为特殊 token 时，才能把它安全地
    # 用作块边界：这样分块既不会切断普通 pre-token，它本身也不会进入
    # 后面的 pair 统计。
    split_special_token = "<|endoftext|>"
    desired_num_chunks = kwargs.get("desired_num_chunks", 32)
    num_workers = kwargs.get(
        "num_workers",
        min(32, os.cpu_count() or 1),
    )

    if not isinstance(desired_num_chunks, int) or desired_num_chunks < 1:
        raise ValueError("desired_num_chunks must be a positive integer")

    if not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")

    if split_special_token not in special_tokens:
        # 没有安全的文档边界时不能任意按 byte 偏移切块，否则可能改变
        # GPT-2 pre-tokenization 的结果。
        desired_num_chunks = 1

    # 主进程只负责寻找边界，不读取或解码整块文本。
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=desired_num_chunks,
            split_special_token=split_special_token.encode("utf-8"),
        )

    worker_input_path = os.path.abspath(os.fspath(input_path))
    worker_special_tokens = tuple(special_tokens)
    chunk_tasks: list[ChunkTask] = [
        (
            worker_input_path,
            start,
            end,
            worker_special_tokens,
        )
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    pretoken_counts: Counter[tuple[int, ...]] = Counter()
    effective_num_workers = min(num_workers, len(chunk_tasks))

    if effective_num_workers <= 1:
        # 单块或显式 num_workers=1 时不创建进程池，方便小语料测试。
        for task in chunk_tasks:
            pretoken_counts.update(_pretokenize_chunk(task))
    else:
        # 只有 read + decode + pretokenize 在子进程并行执行。
        # 主进程收到一块的 Counter 后立即累加；merge 不进入进程池。
        with Pool(processes=effective_num_workers) as pool:
            for chunk_pretoken_counts in pool.imap_unordered(
                _pretokenize_chunk,
                chunk_tasks,
                chunksize=1,
            ):
                pretoken_counts.update(chunk_pretoken_counts)
                del chunk_pretoken_counts

    print(
        f"pretok_done unique_pretokens={len(pretoken_counts)}",
        flush=True,
    )
    
    # 阶段二：反复选择全局最佳 pair，并将其加入词表。

    # 4. 按训练顺序保存byte pair
    merges: list[tuple[bytes,bytes]] = []

    pair_counts, pair_to_pretokens = build_pair_index(
        pretoken_counts
    )
    pair_heap = build_pair_heap(
        pair_counts=pair_counts,
        vocab=vocab,
    )

    merge_started_at = time.perf_counter()
    remaining_merges = vocab_size - len(vocab)
    print(
        f"merge_start current_vocab={len(vocab)} "
        f"target_vocab={vocab_size} remaining_merges={remaining_merges}",
        flush=True,
    )

    # 6. 重复执行合并pair，直到词表大小达到目标大小
    while len(vocab) < vocab_size and pair_counts:
        best_pair = select_best_pair_from_heap(
            pair_heap=pair_heap,
            pair_counts=pair_counts,
        )

        if best_pair is None:
            raise RuntimeError(
                "pair heap is empty while pair_counts is not empty"
            )

        # best_pair 是两个旧 token ID
        left_token_bytes = vocab[best_pair[0]]
        right_token_bytes = vocab[best_pair[1]]

        # 当前词表的 ID 连续，因此 len(vocab) 就是下一个 ID
        new_token_id = max(vocab) + 1

        # 创建新的合并 token
        vocab[new_token_id] = (
            left_token_bytes
            + right_token_bytes
        )

        # merges 保存 bytes pair，而不是 token ID pair
        merges.append(
            (left_token_bytes, right_token_bytes)
        )

        changed_pairs = merge_incrementally(
            pretoken_counts=pretoken_counts,
            pair_counts=pair_counts,
            pair_to_pretokens=pair_to_pretokens,
            pair_to_merge=best_pair,
            new_token_id=new_token_id,
        )

        push_pair_heap_updates(
            pair_heap=pair_heap,
            changed_pairs=changed_pairs,
            pair_counts=pair_counts,
            vocab=vocab,
        )

        # 懒更新会保留旧条目。堆膨胀到活动 pair 数量的 4 倍以上时
        # 偶尔重建一次，避免低频旧条目长期占用过多内存。
        if (
            pair_counts
            and len(pair_heap)
            > max(4 * len(pair_counts), 100_000)
        ):
            pair_heap = build_pair_heap(
                pair_counts=pair_counts,
                vocab=vocab,
            )

        merge_step = len(merges)
        if merge_step == 1 or merge_step % 500 == 0:
            elapsed = time.perf_counter() - merge_started_at
            rate = merge_step / elapsed if elapsed > 0 else 0.0
            remaining = vocab_size - len(vocab)
            eta_s = remaining / rate if rate > 0 else -1.0
            print(
                f"merge_step={merge_step} vocab={len(vocab)}/{vocab_size} "
                f"elapsed_s={elapsed:.1f} merges_per_s={rate:.1f} eta_s={eta_s:.0f}",
                flush=True,
            )

    merge_elapsed = time.perf_counter() - merge_started_at
    print(
        f"merge_done merges={len(merges)} vocab={len(vocab)} "
        f"elapsed_s={merge_elapsed:.1f}",
        flush=True,
    )

    return vocab, merges

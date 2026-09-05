import os
from collections import Counter
from .pretokenization_example import find_chunk_boundaries
from .bpe import (
    initialize_vocab,
    build_pair_index,
    merge_incrementally,
    pretokenize,
    select_best_pair,
)

# 训练 byte-level BPE Tokenizer 的步骤

# 步骤一：词表初始化

# 步骤二：合并 pair

# 步骤三：添加新 token

# 步骤四：重复步骤二和步骤三，直到词表大小达到目标大小

# 步骤五：保存词表  

"""
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 在给定语料上训练 byte-level BPE tokenizer。

    # Returns:
    #    ``vocab``：token ID 到 token bytes 的映射。
    #    ``merges``：按训练先后排列的 byte pair。

   # ``**kwargs`` 保留给 adapter 或后续性能优化参数，当前基础实现不使用。
    
    # 阶段一：初始化词表并把语料转换为 pre-token 频次表。
    # 1. 初始化基础词表和特殊token
    vocab = initialize_vocab(
        vocab_size = vocab_size,
        special_tokens = special_tokens,
    )
    
    # 2. 从磁盘中读取训练语料
    with open(input_path, "r", encoding="utf-8") as file:
        input_text = file.read()
    
    # 阶段二：反复选择全局最佳 pair，并将其加入词表。

    # 4. 按训练顺序保存byte pair
    merges: list[tuple[bytes,bytes]] = []


    pretoken_counts = pretokenize(
        text = input_text,
        special_tokens = special_tokens,
    )

    pair_counts, pair_to_pretokens = build_pair_index(
        pretoken_counts
    )

    # 6. 重复执行合并pair，直到词表大小达到目标大小
    while len(vocab) < vocab_size and pair_counts:
        best_pair = select_best_pair(
            pair_counts,
            vocab,
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

        merge_incrementally(
            pretoken_counts=pretoken_counts,
            pair_counts=pair_counts,
            pair_to_pretokens=pair_to_pretokens,
            pair_to_merge=best_pair,
            new_token_id=new_token_id,
        )

    return vocab, merges

"""





# 训练 byte-level BPE Tokenizer 的步骤

# 步骤一：词表初始化

# 步骤二：合并 pair

# 步骤三：添加新 token

# 步骤四：重复步骤二和步骤三，直到词表大小达到目标大小

# 步骤五：保存词表  


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

    ``desired_num_chunks`` 可通过 ``**kwargs`` 指定，控制读取语料时希望
    切成多少块；当前仍按顺序逐块处理，不启用多进程。
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
    desired_num_chunks = kwargs.get("desired_num_chunks", 4)

    if not isinstance(desired_num_chunks, int) or desired_num_chunks < 1:
        raise ValueError("desired_num_chunks must be a positive integer")

    if split_special_token not in special_tokens:
        # 没有安全的文档边界时不能任意按 byte 偏移切块，否则可能改变
        # GPT-2 pre-tokenization 的结果。
        desired_num_chunks = 1

    pretoken_counts: Counter[tuple[int, ...]] = Counter()

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=desired_num_chunks,
            split_special_token=split_special_token.encode("utf-8"),
        )

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk_text = file.read(end - start).decode("utf-8")

            chunk_pretoken_counts = pretokenize(
                text=chunk_text,
                special_tokens=special_tokens,
            )
            pretoken_counts.update(chunk_pretoken_counts)

            # 当前块已经汇总进总 Counter，可以立即释放。
            del chunk_text, chunk_pretoken_counts
    
    # 阶段二：反复选择全局最佳 pair，并将其加入词表。

    # 4. 按训练顺序保存byte pair
    merges: list[tuple[bytes,bytes]] = []

    pair_counts, pair_to_pretokens = build_pair_index(
        pretoken_counts
    )

    # 6. 重复执行合并pair，直到词表大小达到目标大小
    while len(vocab) < vocab_size and pair_counts:
        best_pair = select_best_pair(
            pair_counts,
            vocab,
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

        merge_incrementally(
            pretoken_counts=pretoken_counts,
            pair_counts=pair_counts,
            pair_to_pretokens=pair_to_pretokens,
            pair_to_merge=best_pair,
            new_token_id=new_token_id,
        )

    return vocab, merges




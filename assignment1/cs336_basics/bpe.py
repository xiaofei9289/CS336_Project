"""
Byte-level BPE 训练所需的底层辅助函数。

本模块只负责无状态的基础操作：
1. 初始化词表；
2. GPT-2 预分词；
3. 统计并选择相邻 token pair；
4. 将选中的 pair 应用到 pre-token。

完整训练流程由 ``train_bpe.py`` 负责，文本编码与解码由
``tokenizer.py`` 中的 ``Tokenizer`` 负责。
注意：本模块只负责无状态的基础操作，不负责训练流程。
"""

from collections import Counter, defaultdict
import regex as regex

# 训练和编码必须使用同一套 GPT-2 预分词规则。
GPT2_PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

TokenIds = tuple[int, ...]
TokenPair = tuple[int, int]
PretokenCounts = dict[TokenIds, int]

# 函数01：初始化基础词表
def initialize_vocab(
    vocab_size: int,
    special_tokens: list[str],
) -> dict[int, bytes]:
    """
    创建 byte-level BPE 的初始词表。

    初始词表包括：
    1. 256 个基础 byte token；
    2. 去重后的特殊 token。

    vocab_size 表示最终目标词表大小。
    本函数不会提前创建BPE merge token，只用于检查目标大小是否足够。
"""
    # 1. 建立256个基础的byte token
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}

    # 2. 加入特殊token，确保特殊Token ID不重复
    # 2.1 去除重复的特殊token，同时保留原来的排列顺序
    unique_special_tokens = list(dict.fromkeys(special_tokens))
    
    # 3. 确保初始化词表大小vocab_size = 256 + 特殊token的数量
    # 3.1 把基础词表长度和特殊token数量相加，得到初始化词表大小
    initial_vocab_size = len(vocab) + len(unique_special_tokens)
    # 3.2 确保输入词表大小vocab_size不小于初始化词表大小initial_vocab_size
    if vocab_size < initial_vocab_size:
        raise ValueError(
            f"Vocab size must be greater than or equal to {initial_vocab_size}"
            )
    # 4. 把基础词表和特殊token合并，得到初始化词表
    # 4.1 遍历每一个特殊token, 把每一个特殊token的byte添加到初始化词表中
    for special_token in unique_special_tokens:
        special_token_id = len(vocab)
        vocab[special_token_id] = special_token.encode('utf-8')

    return vocab

# 函数02：把文本按照特殊token进行分割

def split_on_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:

    """ 
    按特殊 token 切开训练文本，并从结果中移除特殊 token。

    这是训练阶段的行为：特殊 token 只作为文档边界，不能参与普通
    BPE pair 的统计。编码阶段需要保留特殊 token，因此由
    ``Tokenizer._split_on_special_tokens`` 单独处理。
        
    """

    # 1. 把special token的列表去重，并保留原来的排列顺序
    unique_special_tokens = list(dict.fromkeys(special_tokens))

    # 2. 如果special token的列表为空，则直接返回文本，不对文本进行切分
    if not unique_special_tokens:
        return [text]

    # 3. 检查特殊token列表中是否有空字符串，禁止空字符串成为特殊token
    if any(token == "" for token in unique_special_tokens):
        raise ValueError("Special tokens list cannot contain empty strings")

    # 4. 把文本按照特殊token进行分割
    # 4.1 在特殊token列表中把较长的特殊token排在前面，较短的特殊token排在后面，避免前缀冲突
    unique_special_tokens.sort(key=len, reverse=True)
    # 4.2 转义特殊token, 即告诉正则表达式，把特殊 token 中的符号当作普通字符理解
    # 4.2.1 建立escape tokens 列表，把特殊 token 中的符号当作普通字符理解
    escaped_tokens = [regex.escape(token) for token in unique_special_tokens]
    # 4.2.2 把escape tokens列表拼接成一个字符串，用|分隔
    special_tokens_pattern = regex.compile("|".join(escaped_tokens))
    # 4.4 使用正则表达式模式匹配文本，并返回匹配到的列表
    return special_tokens_pattern.split(text)

#函数03：对文本片段进行预分词
def pretokenize(
    text: str,
    special_tokens: list[str],
) -> PretokenCounts:
    """
    将训练文本转换为 ``byte ID tuple -> 出现次数`` 的频次表。
    """

    # 0. 创建计数器
    # key：一个 pre-token 对应的 byte 整数序列
    # value：这个整数序列出现的次数
    pretoken_counts: Counter[TokenIds] = Counter()


    # 1. 先调用split_on_special_tokens函数，把文本按照特殊token进行分割
    text_fragments = split_on_special_tokens(text, special_tokens)
    
    # 2. 分别处理每一个普通token的文本片段
    for text_fragment in text_fragments:
        # 2.1 如果字符串为空，则跳过
        if text_fragment == "":
            continue
        # 2.2 创建匹配结果列表，它不是用来统计频次，而是用来存储正则表达式匹配到的所有文本片段
        matched_text_parts = []
        
        # 3. 使用GPT-2正则进行预分词
        for match in GPT2_PRETOKEN_PATTERN.finditer(text_fragment):
            pretoken_text = match.group(0)
            matched_text_parts.append(pretoken_text)
            
            # 4. 把匹配到的文本片段转换成 utf-8 bytes
            pretoken_bytes = pretoken_text.encode('utf-8')
            # 5. 把每个byte转化为整数，组成不可变的tuple
            pretoken_token_ids =tuple(pretoken_bytes)
            # 6. 统计当前 byte 整数序列出现的次数
            pretoken_counts[pretoken_token_ids] += 1

        # 7. 确认正则没有漏掉
        assert "".join(matched_text_parts) == text_fragment
        
    return pretoken_counts


# 函数04：实现全局相邻pair统计
def count_adjacent_pairs(
    pretoken_counts: dict[tuple[int, ...], int],
) -> dict[TokenPair, int]:
    """
    按 pre-token 频次统计所有相邻 token pair 的全局频次。
    """

    # 0. 创建计数器
    # key：一个相邻pair对应的 byte 整数序列
    # value：这个整数序列出现的次数
    adjacent_pair_counts: Counter[tuple[int, int]] = Counter()
    
    # 1. 遍历pretoken_counts中的每一个整数序列
    for pretoken_token_ids, pretoken_frequency in pretoken_counts.items():
        # 1.1 如果整数序列长度小于2，则跳过
        if len(pretoken_token_ids) < 2:
            continue
        # 1.2 
        for pair in zip(pretoken_token_ids, pretoken_token_ids[1:]):
            adjacent_pair_counts[pair] += pretoken_frequency
    return dict(adjacent_pair_counts)   
#------
def build_pair_index(
    pretoken_counts: dict[tuple[int, ...], int],
):
    """
    返回：
    1. pair_counts：
       每个相邻 pair 在整个语料中的加权出现次数

    2. pair_to_pretokens：
       每个 pair 出现在哪些 unique pre-token 中
    """

    pair_counts = Counter()
    pair_to_pretokens = defaultdict(set)

    for pretoken, frequency in pretoken_counts.items():

        # 统计这个 pre-token 内部各 pair 出现几次
        local_pair_counts = Counter(
            zip(pretoken, pretoken[1:])
        )

        for pair, occurrences in local_pair_counts.items():
            # 出现次数 × 当前 pre-token 的语料频次
            pair_counts[pair] += occurrences * frequency

            # 倒排索引只需要记录这个 pre-token 一次
            pair_to_pretokens[pair].add(pretoken)

    return pair_counts, pair_to_pretokens
#--------------------------------




# 函数05：选择全局词频最高的pair，频次相同时，选择对应byte pair字典序更大的pair
def select_best_pair(
    pair_counts: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
) -> TokenPair | None:
    """
    选择全局词频最高的pair，频次相同时，选择对应byte pair字典序更大的pair
    """

    # 1. 如果pair_counts为空，则返回None
    if not pair_counts:
        return None
    # 2. 找到pair_counts中词频最高的pair
    best_pair = max(
        pair_counts,
        key = lambda pair:(pair_counts[pair], vocab[pair[0]], vocab[pair[1]])
    )

    return best_pair

# 函数06：在单个pre-token中， 从左到右执行一捆非重叠的pair合并
def merge_pairs(
    token_ids: tuple[int, ...],
    target_pair: tuple[int, int],
    new_token_id: int,
) -> TokenIds:
    """
    在单个pre-token中， 从左到右执行一捆非重叠的pair合并
    """

    # 1. 创建一个新的token_ids列表
    merged_token_ids = []
    # 2. 创建一个指针，用来记录当前遍历到的位置
    current_index = 0
    # 3. 
    while current_index < len(token_ids):
        # 3.1 如果当前位置的token_id和target_pair的第一个token_id和第二个token_id都匹配，则合并这两个token_id
        if(
            current_index + 1 < len(token_ids) and
            token_ids[current_index] == target_pair[0] and
            token_ids[current_index + 1] == target_pair[1]
        ):
            merged_token_ids.append(new_token_id)
            current_index += 2
        else:
            # 3.2 如果当前位置的token_id和target_pair的第一个token_id和第二个token_id都不匹配，则直接添加当前token_id
            merged_token_ids.append(token_ids[current_index])
            current_index += 1
    return tuple(merged_token_ids)


# 函数07：将选中的 pair 应用到所有 pre-token，并生成新的频次表。
def merge_all_pretokens(
    pretoken_counts: PretokenCounts,
    pair_to_merge: TokenPair,
    new_token_id: int,
) -> PretokenCounts:
    """将一个 pair 合并应用到全部 pre-token，并聚合新的频次表。"""
    new_pretoken_counts: Counter[TokenIds] = Counter()

    for token_ids, frequency in pretoken_counts.items():
        merged_token_ids = merge_pairs(
            token_ids=token_ids,
            target_pair=pair_to_merge,
            new_token_id=new_token_id,
        )
        new_pretoken_counts[merged_token_ids] += frequency

    return dict(new_pretoken_counts)


#--------------------------------

def merge_incrementally(
    pretoken_counts: dict[tuple[int, ...], int],
    pair_counts: Counter,
    pair_to_pretokens: dict[
        tuple[int, int],
        set[tuple[int, ...]],
    ],
    pair_to_merge: tuple[int, int],
    new_token_id: int,
) -> None:
    """
    把 pair_to_merge 合并成 new_token_id。

    直接原地更新：
    - pretoken_counts
    - pair_counts
    - pair_to_pretokens
    """

    # 必须先复制，因为后面会修改倒排索引
    affected_pretokens = list(
        pair_to_pretokens.get(pair_to_merge, set())
    )

    if not affected_pretokens:
        return

    # 先保存所有变化，避免边遍历边修改造成混乱
    changes = []

    for old_pretoken in affected_pretokens:
        frequency = pretoken_counts[old_pretoken]

        new_pretoken = merge_pairs(
            old_pretoken,
            pair_to_merge,
            new_token_id,
        )

        old_local_counts = Counter(
            zip(old_pretoken, old_pretoken[1:])
        )

        new_local_counts = Counter(
            zip(new_pretoken, new_pretoken[1:])
        )

        changes.append(
            (
                old_pretoken,
                new_pretoken,
                frequency,
                old_local_counts,
                new_local_counts,
            )
        )

    # 第一阶段：删除旧 pre-token 及其 pair 贡献
    for (
        old_pretoken,
        new_pretoken,
        frequency,
        old_local_counts,
        new_local_counts,
    ) in changes:

        del pretoken_counts[old_pretoken]

        for old_pair, occurrences in old_local_counts.items():
            pair_counts[old_pair] -= occurrences * frequency

            # 倒排索引中删除旧序列
            pretoken_set = pair_to_pretokens[old_pair]
            pretoken_set.discard(old_pretoken)

            # 清理空项
            if not pretoken_set:
                del pair_to_pretokens[old_pair]

            if pair_counts[old_pair] == 0:
                del pair_counts[old_pair]

    # 第二阶段：加入新 pre-token 及其 pair 贡献
    for (
        old_pretoken,
        new_pretoken,
        frequency,
        old_local_counts,
        new_local_counts,
    ) in changes:

        # 通常不会发生碰撞，但使用累加写法更安全
        pretoken_counts[new_pretoken] = (
            pretoken_counts.get(new_pretoken, 0)
            + frequency
        )

        for new_pair, occurrences in new_local_counts.items():
            pair_counts[new_pair] += occurrences * frequency
            pair_to_pretokens[new_pair].add(new_pretoken)



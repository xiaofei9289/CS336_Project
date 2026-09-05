import regex as regex
from collections.abc import Iterable, Iterator

from .bpe import GPT2_PRETOKEN_PATTERN

class Tokenizer:

    # 1. 初始化构造函数
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes,bytes]],
        special_tokens: list[str] | None = None,
        ):
        # 1. 复制可变输入，避免类外修改影响 Tokenizer。
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or []) 

        # 2. 创建反向词表：token bytes -> token id
        self.inverse_vocab = self._build_inverse_vocab()

        # 3. 创建merge 优先级
        self.merge_ranks = self._build_merge_ranks()
        
        # 4. 保存特殊token对应的token id
        self.special_token_ids: dict[str, int] = {}

        for special_token in self.special_tokens:
            # 将特殊 token 字符串转换为 UTF-8 bytes
            token_bytes = special_token.encode("utf-8")

            # 检查特殊 token 是否存在于 vocab
            if token_bytes not in self.inverse_vocab:
                raise ValueError(
                    f"special token {special_token!r} is not in vocab"
                )

            # 保存两种特殊 token 映射
            self.special_token_ids[special_token] = self.inverse_vocab[token_bytes]
        
        # 编译一次即可，避免每次调用 encode 都重新创建正则表达式。
        unique_special_tokens = sorted(
            set(self.special_tokens),
            key=len,
            reverse=True,
        )
        if "" in unique_special_tokens:
            raise ValueError("special_tokens cannot contain an empty string")

        self._special_token_pattern = (
            regex.compile(
                "|".join(
                    regex.escape(token) for token in unique_special_tokens
                )
            )
            if unique_special_tokens
            else None
        ) 




    # ---------- 公开接口 ----------
    # 1. 实现编码encode函数
    def encode(self, text: str) -> list[int]:
        """
        将任意原始文本编码为 token ID 列表。

        流程：
        1. 保留特殊 token 地切分原始文本；
        2. 特殊 token 直接映射为单个 ID；
        3. 普通文本进行 GPT-2 预分词；
        4. 每个 pre-token 转为初始 byte ID；
        5. 每个 pre-token 内执行 BPE merges；
        6. 拼接所有结果。
        """

        final_token_ids: list[int] = []

        # 1. 分离普通文本和特殊 token
        segments = self._split_on_special_tokens(text)

        for segment, is_special in segments:

            # 2. 特殊 token 直接输出一个 ID
            if is_special:
                final_token_ids.append(self.special_token_ids[segment])
            else:
                final_token_ids.extend(self._encode_ordinary_text(segment))

        return final_token_ids

    # 2. 实现逐段编码encode_iterable函数
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """逐段编码字符串 iterable，并依次产出 token ID。"""
        for chunk in iterable:
            if chunk:
                yield from self.encode(chunk)

    # 3. 实现解码decode函数
    def decode(self, tokens: list[int]) -> str:
        #  1.1 创建byte缓冲区
        decode_bytes = bytearray()
        # 1.2 遍历每一个token id
        for token_id in tokens:
            # 1.21 检查token id是否存在
            if token_id not in self.vocab:
                raise ValueError(f"Unkown token id: {token_id}")
            # 1.22 去除并追加token对应的bytes
            decode_bytes.extend(self.vocab[token_id])
        # 1.3 循环结束后，将byte缓冲区转换为不可变的字符串
        decoded_str = decode_bytes.decode("utf-8", errors="replace")
        return decoded_str

# ---------- 私有初始化辅助函数 ----------

    def _build_inverse_vocab(self) -> dict[bytes, int]:
        """创建 ``token bytes -> token ID`` 的反向词表。"""
        inverse_vocab: dict[bytes, int] = {}

        for token_id, token_bytes in self.vocab.items():
            if token_bytes in inverse_vocab:
                existing_id = inverse_vocab[token_bytes]
                raise ValueError(
                    f"token bytes {token_bytes!r} already use ID {existing_id}"
                )
            inverse_vocab[token_bytes] = token_id

        return inverse_vocab

    def _build_merge_ranks(self) -> dict[tuple[bytes, bytes], int]:
        """把 merge 列表转换为 pair 到优先级的映射。"""
        merge_ranks: dict[tuple[bytes, bytes], int] = {}

        for rank, pair in enumerate(self.merges):
            if pair in merge_ranks:
                raise ValueError(
                    f"merge pair {pair!r} already has rank {merge_ranks[pair]}"
                )
            merge_ranks[pair] = rank

        return merge_ranks

# ---------- 私有编码辅助函数 ----------
# 1. 保留特殊token的切分函数
    def _split_on_special_tokens(
        self, 
        text: str,
        ) -> list[tuple[str, bool]]:
        """
        输入一段原始文本后：
        1. 找出其中特殊的token
        2. 把普通文本和特殊token分开
        3. 特殊token原样保留
        4. 返回带类型标记的片段列表
        """
        # 1. 如果文本为空，则返回空列表
        if not text:
            return []
        # 2. 如果当前tokenizer没有适配任何特殊token，直接把其当成普通文本片段返回
        if not self.special_tokens:
            return [(text, False)]
        # 3. 创建结果列表，这是一个空列表，用来保存最终的切分结果
        segments: list[tuple[str, bool]] = []
        # 4. 创建一个指针，用来遍历文本
        pointer = 0
        # 5. 遍历文本，查找每一个特殊token
        for match in self._special_token_pattern.finditer(text):
            # 5.1 如果找到的特殊token位置比指针位置大，说明中间有普通文本
            if match.start() > pointer:
                normal_text = text[pointer:match.start()]
                segments.append((normal_text, False))
            # 5.2 如果找到的特殊token位置和指针位置相同，说明这个特殊token就是开头
            special_token =match.group(0)
            segments.append((special_token, True))
            # 5.3 更新指针位置
            pointer = match.end()
        # 6. 处理文本末尾的普通文本
        if pointer < len(text):
            normal_text = text[pointer:]
            segments.append((normal_text, False))
        return segments

    # 2. 实现编码普通文本函数
    def _encode_ordinary_text(self, text: str) -> list[int]:
        """对不含特殊 token 的文本执行预分词和 BPE。"""
        pre_tokens = [
            match.group(0)
            for match in GPT2_PRETOKEN_PATTERN.finditer(text)
        ]
        if "".join(pre_tokens) != text:
            raise ValueError(
                "GPT-2 pre-tokenization did not preserve the input text: "
                f"{text!r}"
            )

        token_ids: list[int] = []
        for pre_token in pre_tokens:
            initial_ids = self._encode_initial_bytes(pre_token)
            token_ids.extend(self._apply_bpe_merges(initial_ids))

        return token_ids

    # 3. 实现编码初始单 byte token ID函数
    def _encode_initial_bytes(self, pre_token: str) -> list[int]:
        """将一个 pre-token 转换为初始单 byte token ID。"""
        token_ids: list[int] = []

        for byte_value in pre_token.encode("utf-8"):
            byte_token = bytes([byte_value])
            if byte_token not in self.inverse_vocab:
                raise ValueError(
                    "single-byte token is missing from vocabulary: "
                    f"{byte_token!r} (byte value: {byte_value})"
                )
            token_ids.append(self.inverse_vocab[byte_token])

        return token_ids
    
    # 4. 实现执行BPE merges函数
    def _apply_bpe_merges(self, token_ids: list[int]) -> list[int]:
        """按训练优先级对一个 pre-token 执行全部可用 merge。"""
        current_ids = list(token_ids)

        for token_id in current_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")

        while len(current_ids) >= 2:
            mergeable_pairs = []
            for left_id, right_id in zip(current_ids, current_ids[1:]):
                pair = (self.vocab[left_id], self.vocab[right_id])
                if pair in self.merge_ranks:
                    mergeable_pairs.append(pair)

            if not mergeable_pairs:
                break

            best_pair = min(
                mergeable_pairs,
                key=self.merge_ranks.__getitem__,
            )
            merged_bytes = best_pair[0] + best_pair[1]
            if merged_bytes not in self.inverse_vocab:
                raise ValueError(
                    "merged token is missing from vocabulary: "
                    f"{best_pair[0]!r} + {best_pair[1]!r} = {merged_bytes!r}"
                )

            current_ids = self._merge_pair_once(
                token_ids=current_ids,
                target_pair=best_pair,
                merged_id=self.inverse_vocab[merged_bytes],
            )

        return current_ids

    # 5. 实现执行一次BPE merges函数
    def _merge_pair_once(
        self,
        token_ids: list[int],
        target_pair: tuple[bytes, bytes],
        merged_id: int,
    ) -> list[int]:
        """从左到右合并当前序列中所有非重叠的目标 pair。"""
        merged_ids: list[int] = []
        index = 0

        while index < len(token_ids):
            if index + 1 < len(token_ids):
                current_pair = (
                    self.vocab[token_ids[index]],
                    self.vocab[token_ids[index + 1]],
                )
                if current_pair == target_pair:
                    merged_ids.append(merged_id)
                    index += 2
                    continue

            merged_ids.append(token_ids[index])
            index += 1

        return merged_ids

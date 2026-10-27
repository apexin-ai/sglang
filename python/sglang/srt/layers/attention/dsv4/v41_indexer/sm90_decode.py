"""SM90 decode for the DeepSeek V4.1 ratio-1/2 index layers: the ported Triton row-gather
kernels stand in for DeepGEMM's SM100-only paged FP4 logits. A candidate source scores
the whole prefix and publishes its blocks; a consumer scores only those blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax_topk_blocks,
    candidate_row_lens,
)
from sglang.kernels.ops.attention.dsv4.candidate_table import sort_candidate_blocks
from sglang.kernels.ops.attention.dsv4.fp4_indexer_sm90 import (
    paged_mqa_logits,
    paged_sparse_mqa_logits,
    prepare_q,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_sparse
from sglang.srt.layers.attention.dsv4.indexer import topk_transform_paged_bf16

from .scoring import DeepGEMMDecodeData, get_deep_gemm_decode_data
from .types import CandidateMetadata, DecodeInputs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

CANDIDATE_BLOCK_SIZE = 8


class _Sm90DecodeTable(CandidateMetadata, msgspec.Struct, frozen=True):
    # [rows, topk_blocks] int32 ascending logical block ids, INT32_MAX past
    # min(topk_blocks, ceil(seq_len / 8)); the sparse kernel reads only those
    blocks: torch.Tensor
    # [rows, topk_blocks] int32: the same blocks as pool slots / 8
    phys_blocks: torch.Tensor
    # [rows] int32: valid length of each sparse logits row (`candidate_row_lens`)
    valid_lens: torch.Tensor


def _prepared_q(data: DeepGEMMDecodeData):
    rows, _, heads, _ = data.q_fp4.shape
    return prepare_q(
        data.q_fp4.view(rows, heads, 64), data.q_sf.view(rows, heads), data.weights
    )


def _row_args(inputs: DecodeInputs, rows: int):
    metadata = inputs.paged_metadata
    seq_lens = metadata.compressed_seq_lens.reshape(-1)[:rows].to(torch.int32)
    return seq_lens, metadata.page_table[:rows]


def sm90_dense_logits(
    inputs: DecodeInputs, token_to_kv_pool: DeepSeekV4TokenToKVPool
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """bf16 logits of every visible position, with the rows' seq lens and page table;
    None (outputs reset) when nothing is visible yet."""
    if inputs.paged_metadata.max_compressed_seq_len == 0:
        inputs.reset_outputs()
        return None
    data = get_deep_gemm_decode_data(inputs, token_to_kv_pool)
    seq_lens, page_table = _row_args(inputs, data.q_fp4.shape[0])
    logits = paged_mqa_logits(
        _prepared_q(data),
        data.k_cache,
        seq_lens,
        page_table,
        inputs.paged_metadata.max_compressed_seq_len,
    )
    return logits, seq_lens, page_table


def sm90_full_topk_decode(
    inputs: DecodeInputs, token_to_kv_pool: DeepSeekV4TokenToKVPool
) -> None:
    scored = sm90_dense_logits(inputs, token_to_kv_pool)
    if scored is None:
        return
    logits, seq_lens, page_table = scored
    topk_transform_paged_bf16(
        logits,
        seq_lens,
        page_table,
        inputs.out_page_indices,
        inputs.paged_metadata.compressed_page_size,
    )


class Sm90CandidateDecode:
    def __init__(
        self,
        *,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        candidate_topk_blocks: int,
        candidate_block_size: int,
    ) -> None:
        assert candidate_block_size == CANDIDATE_BLOCK_SIZE
        self.token_to_kv_pool = token_to_kv_pool
        self.topk_blocks = candidate_topk_blocks

    def publish_decode(self, inputs: DecodeInputs) -> Optional[_Sm90DecodeTable]:
        scored = sm90_dense_logits(inputs, self.token_to_kv_pool)
        if scored is None:
            return None
        logits, seq_lens, page_table = scored
        page_size = inputs.paged_metadata.compressed_page_size
        topk_transform_paged_bf16(
            logits, seq_lens, page_table, inputs.out_page_indices, page_size
        )
        nblocks, valid_lens = candidate_row_lens(seq_lens, self.topk_blocks)
        blocks = amax_topk_blocks(logits, seq_lens, nblocks, self.topk_blocks)
        # in place: ascending logical ids, INT32_MAX padded; returns them as pool slots / 8
        phys_blocks = sort_candidate_blocks(blocks, seq_lens, page_table, page_size)
        return _Sm90DecodeTable(
            blocks=blocks, phys_blocks=phys_blocks, valid_lens=valid_lens
        )

    def consume_decode(
        self, inputs: DecodeInputs, published: Optional[_Sm90DecodeTable]
    ) -> None:
        if published is None:
            inputs.reset_outputs()
            return
        data = get_deep_gemm_decode_data(inputs, self.token_to_kv_pool)
        seq_lens, page_table = _row_args(inputs, data.q_fp4.shape[0])
        logits = paged_sparse_mqa_logits(
            _prepared_q(data),
            data.k_cache,
            seq_lens,
            page_table,
            published.blocks,
            CANDIDATE_BLOCK_SIZE,
        )
        topk_transform_sparse(
            logits, published.valid_lens, published.phys_blocks, inputs.out_page_indices
        )

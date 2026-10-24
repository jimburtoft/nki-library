# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""attention_block_tkg at the Qwen3-ASR batched decode shape with a packed FP8 KV cache.

Text decoder at TP1: 16 Q heads, 8 KV heads, d_head 128, H 2048, pre-RoPE QK RMSNorm, input RMSNorm,
block KV (block_len 32), S_tkg 1, in-kernel cache update, LNC1. With the packed FP8 cache the kernel
quantizes the new K/V (unit scales) and writes K in the packed layout; attention_tkg reads the packed
cache. Run with NKILIB_CR17_VARIANT unset (dt4+stitch), "stock", or "dt4+stitch+fp8st"
(NKILIB_CR17_ALLOW_EXPERIMENTAL=1) to cover the CR17 FP8 group stitch end to end, including the cache write.

    pytest test/integration/nkilib/experimental/transformer/test_attention_block_tkg_qwen3_asr_fp8.py \
        --test-mode simulation --platform-target trn2
"""

from typing import final

import pytest

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg import (
    _run_attention_block_test,
)
from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_utils import (
    AttnBlkTestConfig,
)
from test.utils.common_dataclasses import Platforms
from test.utils.test_orchestrator import Orchestrator


def _cfg(batch: int, S_ctx: int, fp8: bool) -> AttnBlkTestConfig:
    return AttnBlkTestConfig(
        batch=batch,
        q_heads=16,
        d_head=128,
        H=2048,
        H_actual=None,
        S_ctx=S_ctx,
        S_max_ctx=S_ctx,
        S_tkg=1,
        block_len=32,
        qk_norm_pre_rope=True,
        qk_norm_pre_rope_gamma=True,
        lnc=1,
        kv_heads=8,
        cache_has_kv_head_dim=True,
        kv_quant=fp8,
        fp8_packed=fp8,
    )


# fmt: off
QWEN3_ASR_FP8_CFGS = [
    _cfg(4, 512, True),     # folded bs 32: the CR17 gate's bs floor; resized block_len 4
    _cfg(8, 1024, True),    # folded bs 64; resized block_len 8
    _cfg(51, 512, True),    # the measured decode shape (bucket 512 = champion decode_context_length_buckets)
    _cfg(4, 512, False),    # bf16 KV control
]
# fmt: on


@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN3, Platforms.TRN3_A0])
@final
class TestAttnBlkQwen3AsrFp8:
    @pytest.mark.parametrize("attn_blk_cfg", QWEN3_ASR_FP8_CFGS, ids=lambda p: p.test_id())
    def test_attn_blk_qwen3_asr_fp8(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        attn_blk_cfg: AttnBlkTestConfig,
    ):
        _run_attention_block_test(test_manager=test_manager, platform_target=platform_target, cfg=attn_blk_cfg)

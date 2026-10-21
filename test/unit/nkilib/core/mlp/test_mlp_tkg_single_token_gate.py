# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the T == 1 down-projection column-tiling gate in mlp_tkg."""

from types import SimpleNamespace

import pytest
from nkilib_src.nkilib.core.mlp.mlp_tkg import mlp_tkg as mlp_tkg_mod


class _Quant:
    def __init__(self, quant=False, mx=False):
        self._quant = quant
        self._mx = mx

    def is_quant(self):
        return self._quant

    def is_dtype_mx(self):
        return self._mx


def _params(**overrides):
    fields = {
        "use_tkg_down_proj_column_tiling": False,
        "use_tkg_gate_up_proj_column_tiling": False,
        "use_tkg_down_proj_optimized_layout": False,
        "transposed_out": False,
        "store_output_in_sbuf": False,
        "quant_params": _Quant(),
        "expert_params": None,
        "down_proj_weights_tensor": SimpleNamespace(shape=(1536, 2048)),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def gen3(monkeypatch):
    monkeypatch.setattr(mlp_tkg_mod.nisa, "get_nc_version", lambda: mlp_tkg_mod.nisa.nc_version.gen3)


def test_single_token_dense_call_selects_down_column_tiling(gen3):
    assert mlp_tkg_mod._use_b1_down_column_tiling(_params(), 1)


@pytest.mark.parametrize("T", [2, 4, 128])
def test_multi_token_calls_keep_the_existing_path(gen3, T):
    assert not mlp_tkg_mod._use_b1_down_column_tiling(_params(), T)


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_tkg_down_proj_column_tiling": True},
        {"use_tkg_gate_up_proj_column_tiling": True},
        {"use_tkg_down_proj_optimized_layout": True},
        {"transposed_out": True},
        {"store_output_in_sbuf": True},
        {"quant_params": _Quant(quant=True)},
        {"quant_params": _Quant(mx=True)},
        {"expert_params": object()},
        {"down_proj_weights_tensor": SimpleNamespace(shape=(8, 1536, 2048))},
    ],
)
def test_excluded_configurations_keep_the_existing_path(gen3, overrides):
    assert not mlp_tkg_mod._use_b1_down_column_tiling(_params(**overrides), 1)


def test_gen2_keeps_the_existing_path(monkeypatch):
    monkeypatch.setattr(mlp_tkg_mod.nisa, "get_nc_version", lambda: mlp_tkg_mod.nisa.nc_version.gen2)
    assert not mlp_tkg_mod._use_b1_down_column_tiling(_params(), 1)

# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch import nn
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.experimental.attention import (
    attention_context_parallel,
    ContextParallel,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    NUM_DEVICES,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional


class RingAttentionTest(DTensorTestBase):
    @with_comms
    def test_ring_attention_sdpa(self):
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        dtype = torch.bfloat16
        bs = 8
        query_tokens = 8
        context_tokens = 16
        dim = 32
        nheads = 8
        query = torch.rand(
            (bs, nheads, query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        original_query = query
        key = torch.rand(
            (bs, nheads, context_tokens, dim), device=self.device_type, dtype=dtype
        )
        value = torch.rand(
            (bs, nheads, context_tokens, dim), device=self.device_type, dtype=dtype
        )

        query_placement = [Replicate()]
        query = DTensor.from_local(query, device_mesh, query_placement)
        self.assertEqual(query.shape, (bs, nheads, query_tokens, dim))

        context_placement = [Shard(2)]
        key = DTensor.from_local(key, device_mesh, context_placement)
        value = DTensor.from_local(value, device_mesh, context_placement)
        self.assertEqual(key.shape, (bs, nheads, context_tokens * self.world_size, dim))
        self.assertEqual(
            value.shape, (bs, nheads, context_tokens * self.world_size, dim)
        )

        # local tensors

        out = torch.ops.aten._scaled_dot_product_flash_attention(
            query.full_tensor(), key.full_tensor(), value.full_tensor()
        )[0]
        self.assertEqual(out.shape, (bs, nheads, query_tokens, dim))
        out.sum().backward()
        out_grad = original_query.grad
        original_query.grad = None
        self.assertIsNotNone(out_grad)

        # nonparallel DTensor

        # TODO: this is wrong, doesn't do any collectives
        with CommDebugMode() as comm_mode:
            out_nonparallel = torch.ops.aten._scaled_dot_product_flash_attention(
                query, key, value
            )[0]
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                # c10d_functional.all_to_all_single: self.world_size - 1,
            },
        )
        self.assertEqual(out_nonparallel.shape, (bs, nheads, query_tokens, dim))
        self.assertEqual(out_nonparallel.full_tensor(), out)

        # parallel behavior

        with attention_context_parallel(), CommDebugMode() as comm_mode:
            out_parallel = torch.ops.aten._scaled_dot_product_flash_attention(
                query, key, value
            )[0]
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: self.world_size - 1,
            },
        )
        self.assertEqual(out_parallel.placements, (Replicate(),))
        self.assertEqual(
            out_parallel._local_tensor.shape, (bs, nheads, query_tokens, dim)
        )
        self.assertEqual(out_parallel.shape, (bs, nheads, query_tokens, dim))
        self.assertEqual(out_parallel.full_tensor(), out)

        with attention_context_parallel(), CommDebugMode() as comm_mode:
            out_parallel.sum().backward()
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1) * 2,
            },
        )
        out_parallel_grad = original_query.grad
        original_query.grad = None
        self.assertIsNotNone(out_parallel_grad)
        self.assertEqual(out_parallel_grad, out_grad)

    @with_comms
    @attention_context_parallel()
    @sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    def test_ring_attention_transformer(self):
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        dtype = torch.bfloat16
        bs = 8
        ntokens = 8
        dim = 32
        nheads = 8
        num_layers = 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nheads,
            dim_feedforward=dim,
            batch_first=True,
        ).to(dtype)
        encoder_layer = parallelize_module(
            module=encoder_layer,
            device_mesh=device_mesh,
            parallelize_plan={
                "self_attn": ContextParallel(),
            },
        )
        model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        model = model.to(self.device_type).to(dtype)

        seq = torch.rand((bs, ntokens, dim), device=self.device_type, dtype=dtype)

        with CommDebugMode() as comm_mode:
            # with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
            out = model(seq)
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1) * num_layers,
            },
        )

        with CommDebugMode() as comm_mode:
            out.sum().backward()
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1)
                * 2
                * num_layers,
            },
        )


instantiate_parametrized_tests(DTensorTestBase)

if __name__ == "__main__":
    run_tests()

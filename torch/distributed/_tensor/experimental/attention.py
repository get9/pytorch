import contextlib
from typing import Dict, Generator, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from torch import nn
from torch.distributed._tensor import distribute_module, DTensor, Replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle

aten = torch.ops.aten


def sdpa_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    local_results = _scaled_dot_product_flash_attention(
        op_info.mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _scaled_dot_product_flash_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")
    if is_causal:
        raise NotImplementedError("is_causal is not supported yet")

    pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)

    # rank 0 sends to rank 1, rank 1 sends to rank 2, ..., rank n-1 sends to rank 0
    right_dsts = list(range(1, size)) + [0]

    next_kv = None

    chunks = []
    logsumexps = []
    for i in range(size):
        # overlap communication with compute
        if next_kv is not None:
            next_kv = ft_c.wait_tensor(next_kv)
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = ft_c.permute_tensor(next_kv, right_dsts, pg)

        local_results = torch.ops.aten._scaled_dot_product_flash_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        chunks.append(local_results[0])
        logsumexps.append(local_results[1])

    softmax_lse = torch.stack([lse.exp() for lse in logsumexps]).sum(dim=0).log_()

    out = []
    for chunk, chunk_lse in zip(chunks, logsumexps):
        softmax_lse_corrected = torch.exp(chunk_lse - softmax_lse)
        out_corrected = chunk * softmax_lse_corrected.unsqueeze(-1).to(chunk.dtype)
        out.append(out_corrected)
    out = torch.stack(out).sum(dim=0)

    local_results = (out, softmax_lse) + local_results[2:]
    return local_results


def sdpa_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    local_results = _scaled_dot_product_flash_attention_backward(
        op_info.mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _scaled_dot_product_flash_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if is_causal:
        raise NotImplementedError("is_causal is not supported yet")

    pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)

    # rank 0 sends to rank 1, rank 1 sends to rank 2, ..., rank n-1 sends to rank 0
    right_dsts = list(range(1, size)) + [0]

    next_kv = None

    out_grad_queries = []
    out_grad_keys = []
    out_grad_values = []

    for i in range(size):
        # overlap communication with compute
        if next_kv is not None:
            next_kv = ft_c.wait_tensor(next_kv)
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = ft_c.permute_tensor(next_kv, right_dsts, pg)

        # we rerun the forwards pass since we don't have a good way to save the
        # output/logsumexp
        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            _,
        ) = torch.ops.aten._scaled_dot_product_flash_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

        softmax_lse_corrected = torch.exp(logsumexp - softmax_lse)

        chunk_grad = grad_out * softmax_lse_corrected.unsqueeze(-1).to(grad_out.dtype)

        (
            grad_query,
            grad_key,
            grad_value,
        ) = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            grad_out=chunk_grad,
            query=query,
            key=key,
            value=value,
            out=output,
            logsumexp=logsumexp,
            cum_seq_q=cum_seq_q,
            cum_seq_k=cum_seq_k,
            max_q=max_q,
            max_k=max_k,
            dropout_p=dropout_p,
            is_causal=is_causal,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            scale=scale,
        )

        # TODO overlap grad communication

        if i == 0:
            out_grad_queries.append(grad_query)
            out_grad_keys.append(grad_key)
            out_grad_values.append(grad_value)
        elif i > 0:
            grad_dsts = [(size - i) % size for i in range(size)]

            grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
            grad_kv = ft_c.permute_tensor(grad_kv, grad_dsts, pg)
            grad_kv = ft_c.wait_tensor(grad_kv)
            grad_key = grad_kv[: grad_key.numel()].reshape(grad_key.shape)
            grad_value = grad_kv[grad_key.numel() :].reshape(grad_value.shape)

            out_grad_queries.append(grad_query)
            out_grad_keys.append(grad_key)
            out_grad_values.append(grad_value)

    # stack and sum to avoid accumulation errors
    out_grad_query = torch.stack(out_grad_queries).sum(dim=0)
    out_grad_key = torch.stack(out_grad_keys).sum(dim=0)
    out_grad_value = torch.stack(out_grad_values).sum(dim=0)

    return out_grad_query, out_grad_key, out_grad_value


customized_ops = {
    aten._scaled_dot_product_flash_attention.default: sdpa_handler,
    aten._scaled_dot_product_flash_attention_backward.default: sdpa_backward_handler,
}


@contextlib.contextmanager
def attention_context_parallel() -> Generator[None, None, None]:
    """
    This enables attention context parallel optimizations. Currently only supports ring attention for SDPA flash attention.
    """
    DTensor._op_dispatcher._custom_op_handlers.update(customized_ops)

    yield

    for custom_op in customized_ops:
        DTensor._op_dispatcher._custom_op_handlers.pop(custom_op)


class ContextParallel(ParallelStyle):
    """
    Applies context parallel optimizations to the attention layer. This will
    work for nn.MultiHeadedAttention.

    Currently only supports ring attention for SDPA flash attention.

    See attention_context_parallel for more details.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, nn.MultiheadAttention):
            raise ValueError(f"{type(module)} is not supported by {type(self)} yet.")

        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(
                f"{type(device_mesh)} is not supported by {type(self)} yet."
            )

        if not device_mesh.ndim == 1:
            raise ValueError

        return distribute_module(
            module,
            device_mesh,
            input_fn=self._input_fn,  # type: ignore[arg-type]
            output_fn=self._output_fn,  # type: ignore[arg-type]
        )

    @staticmethod
    def _input_fn(
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device_mesh: DeviceMesh
    ) -> Tuple[DTensor, DTensor, DTensor]:
        # TODO: figure out dynamo support for instance method and switch this to instance method

        query_placement = [Replicate()]
        kv_placement = [Replicate()]

        # convert inputs to DTensor
        query, key, value = inputs
        if not isinstance(query, DTensor):
            query = DTensor.from_local(
                query, device_mesh, query_placement, run_check=False
            )
        if not isinstance(key, DTensor):
            key = DTensor.from_local(key, device_mesh, kv_placement, run_check=False)
        if not isinstance(value, DTensor):
            value = DTensor.from_local(
                value, device_mesh, kv_placement, run_check=False
            )

        return query, key, value

    @staticmethod
    def _output_fn(
        outputs: Tuple[Union[torch.Tensor, int, float], ...], device_mesh: DeviceMesh
    ) -> Tuple[Union[torch.Tensor, int, float], ...]:
        # back to local tensor
        return tuple(
            output.to_local() if isinstance(output, DTensor) else output
            for output in outputs
        )

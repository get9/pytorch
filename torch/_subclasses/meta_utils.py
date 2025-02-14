from __future__ import annotations

import contextlib
import warnings
import weakref

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

from typing_extensions import TypeAlias

import torch
from torch._C._functorch import (
    _add_batch_dim,
    _unwrap_functional_tensor,
    _wrap_functional_tensor,
    get_unwrapped,
    is_batchedtensor,
    is_functorch_wrapped_tensor,
    is_gradtrackingtensor,
    is_legacy_batchedtensor,
    maybe_get_bdim,
    maybe_get_level,
    peek_interpreter_stack,
)
from torch._guards import Source

from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.weak import WeakIdKeyDictionary

if TYPE_CHECKING:
    from torch._C._autograd import CreationMeta
    from torch._C._functorch import CInterpreter

    # Import here to avoid cycle
    from torch._subclasses.fake_tensor import FakeTensorMode

    # Import the following modules during type checking to enable code intelligence features,
    # Do not import unconditionally, as they import sympy and importing sympy is very slow
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext

DimList = List


def safe_is_leaf(t):
    try:
        return t.is_leaf
    except RuntimeError:
        # inference mode can trigger this
        return False


def safe_grad(t):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The .grad attribute of a Tensor")
        return t.grad


def assert_eq(a, b):
    assert a == b, f"{a} != {b}"


def assert_metadata_eq(
    assert_eq,
    m1: Union[MetaTensorDesc, torch.Tensor],
    m2: torch.Tensor,
    *,
    skip_symbolic=False,
    skip_leaf=False,
):
    if isinstance(m1, torch.Tensor):
        m1 = MetaTensorDescriber().describe_tensor(m1)

    def go(m1, m2):
        assert_eq(m1.dtype, m2.dtype)
        if not skip_symbolic:
            assert_eq(m1.shape, m2.shape)
        assert_eq(m1.requires_grad, m2.requires_grad)
        if not skip_leaf:
            assert_eq(m1.is_leaf, m2.is_leaf)
        # MetaTensorDesc doesn't store grad_fn; inferred from leaf
        # assert_eq(m1.grad_fn is None, m2.grad_fn is None)
        assert_eq(m1.is_sparse, m2.is_sparse)
        assert_eq(m1.is_inference, m2.is_inference())
        assert_eq(m1.is_conj, m2.is_conj())
        assert_eq(m1.is_neg, m2.is_neg())
        assert_eq(m1.grad is not None, safe_grad(m2) is not None)
        if m1.grad is not None:
            go(m1.grad, safe_grad(m2))
        if m1.is_sparse:
            assert_eq(m1.dense_dim, m2.dense_dim())
            assert_eq(m1.sparse_dim, m2.sparse_dim())
            assert_eq(m1.is_coalesced, m2.is_coalesced())
        else:
            if not skip_symbolic:
                assert_eq(m1.stride, m2.stride())
                assert_eq(m1.storage_offset, m2.storage_offset())
            assert_eq(m1.is_view, m2._is_view())
            if m1.is_view:
                go(m1.base, m2._base)
        # TODO: test if is resizable (no direct query for this atm)
        # TODO: audit AutogradMeta to see if it matches
        # TODO: test forward AD

    return go(m1, m2)


def is_sparse_coo(t):
    return isinstance(t, torch.Tensor) and t.layout is torch.sparse_coo


def is_sparse_compressed_layout(layout):
    return layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }


def is_sparse_compressed(t):
    return isinstance(t, torch.Tensor) and is_sparse_compressed_layout(t.layout)


def is_sparse_any(t):
    return is_sparse_coo(t) or is_sparse_compressed(t)


# Don't use id() directly, because those can get reallocated over time.
MetaStorageId: TypeAlias = int
MetaTensorId: TypeAlias = int


class MetaTensorDescriber:
    """
    Given a Tensor/Storage, generate a MetaTensorDesc/MetaStorageDesc
    for it, which is enough information to reconstruct a meta tensor/fake tensor
    corresponding to a Tensor as faithfully as possible.

    This is a stateful conversion object because we keep track of the IDs
    of the tensors/storages passed to us, so we can consistently give
    the same ID when we see the same tensor/storage.
    """

    def __init__(self):
        self.next_tensor_id: MetaTensorId = 0
        self.next_storage_id: MetaStorageId = 0
        # Tensor -> int
        self.lookup_tensor = WeakIdKeyDictionary()
        # Storage -> int
        self.lookup_storage = WeakIdKeyDictionary()

    def get_tensor_id(self, t: torch.Tensor):
        if t not in self.lookup_tensor:
            self.lookup_tensor[t] = self.next_tensor_id
            self.next_tensor_id += 1
        return self.lookup_tensor[t]

    def get_storage_id(self, s: torch.UntypedStorage):
        if s not in self.lookup_storage:
            self.lookup_storage[s] = self.next_storage_id
            self.next_storage_id += 1
        return self.lookup_storage[s]

    # NB: the describe functions NOT maintain a cache and will happily regen the
    # description

    def describe_storage(self, s: torch.UntypedStorage):
        return MetaStorageDesc(
            id=self.get_storage_id(s),
            size=s.size(),
        )

    def describe_tensor(self, t: torch.Tensor, recurse: bool = True):
        is_leaf = safe_is_leaf(t)
        is_view = t._is_view()
        is_sparse = t.is_sparse
        layout = t.layout
        is_nested = t.is_nested
        is_traceable_wrapper_subclass_v = is_traceable_wrapper_subclass(t)
        is_functorch_wrapped = is_functorch_wrapped_tensor(t)
        is_mkldnn = t.is_mkldnn
        is_batchedtensor_v = is_batchedtensor(t)
        is_legacy_batchedtensor_v = is_legacy_batchedtensor(t)
        is_gradtrackingtensor_v = is_gradtrackingtensor(t)
        is_functorch_batched_or_grad = is_batchedtensor_v or is_gradtrackingtensor_v
        is_functional = torch._is_functional_tensor(t)

        storage = None
        # NB: For compatibility, I default this to zero, as sometimes people
        # still have stuffed zero into storage offset even though the tensor
        # doesn't meaningfully have an offset
        storage_offset = 0
        if not (
            is_sparse
            or is_sparse_compressed_layout(layout)
            or (is_nested and not is_traceable_wrapper_subclass_v)
            or is_mkldnn
            # TODO: TBH, functorch wrapped tensors probably should have
            # storage associated with them
            or is_functorch_wrapped
            or is_legacy_batchedtensor_v
        ):
            # NB: We actually don't use storage to do views, but might as well
            # put it in for accuracy
            storage = self.describe_storage(t.untyped_storage())
            storage_offset = t.storage_offset()

        stride = None
        if not (
            is_sparse
            or is_sparse_compressed_layout(layout)
            or (is_nested and not is_traceable_wrapper_subclass_v)
        ):
            # stride/storage_offset are called from is_functorch_wrapped,
            # view_from_base, empty_create_subclass,
            # sym_sizes_strides_storage_offset (empty_create)
            stride = t.stride()

        # NB: this technically should refer to functorch unwrapped tensor, but
        # I am (perhaps abusively) using it to store both the functorch and
        # non-functorch functional tensor
        unwrapped = None
        autograd_meta_from = None
        current_level = None
        if is_batchedtensor_v or is_gradtrackingtensor_v:
            unwrapped = self.describe_tensor(get_unwrapped(t))
        # xla and lazy tensors present as functional tensors, but we want them
        # to be handled specially
        elif is_functional and t.device.type not in ("xla", "lazy"):
            if t._is_view():
                raise RuntimeError(
                    "Cannot safely fakify a view because this process drops the view information right now."
                )
            if not is_functorch_wrapped:
                torch._sync(t)
                unwrapped = self.describe_tensor(torch._from_functional_tensor(t))
                autograd_meta_from = t
            else:
                reapply_views = torch._C._functionalization_reapply_views_tls()
                # NB: has side effects!
                unwrapped = self.describe_tensor(
                    _unwrap_functional_tensor(t, reapply_views)
                )
                # TODO: It's pretty suspicious that functional tensors don't have
                # valid level and thus we just grab whatever the current level
                # is
                current_level = torch._C._functorch.current_level()

        maybe_functorch_stack = None
        if is_functorch_wrapped:
            with torch._functorch.pyfunctorch.temporarily_clear_interpreter_stack() as maybe_functorch_stack:
                pass

        attrs = None
        ctx = None
        type_v = None
        if is_traceable_wrapper_subclass_v:
            assert hasattr(t, "__tensor_flatten__")
            raw_attrs, ctx = t.__tensor_flatten__()
            attrs = {attr: self.describe_tensor(getattr(t, attr)) for attr in raw_attrs}
            type_v = type(t)

        # TODO: Is it important to enable torch.inference_mode before querying
        # these values?
        return MetaTensorDesc(
            id=self.get_tensor_id(t),
            storage=storage,
            is_inference=t.is_inference(),
            is_leaf=is_leaf,
            requires_grad=t.requires_grad,
            # NB: ndim should be OK too but there is a disaster at
            # python test/dynamo/test_subclasses.py -k test_user_overidden_property_unsupported
            # Actually, this means that we have a little bit of a problem
            # here, which is that there is some sensitivity to how exactly an
            # access is done if you have a __torch_function__ subclass.  Maybe
            # should disable torch function before doing accesses?
            ndim=t.dim(),
            dtype=t.dtype,
            is_sparse=is_sparse,
            is_mkldnn=is_mkldnn,
            is_functorch_wrapped=is_functorch_wrapped,
            is_batchedtensor=is_batchedtensor_v,
            is_legacy_batchedtensor=is_legacy_batchedtensor_v,
            is_gradtrackingtensor=is_gradtrackingtensor_v,
            is_view=is_view,
            is_conj=t.is_conj(),
            is_neg=t.is_neg(),
            is_traceable_wrapper_subclass=is_traceable_wrapper_subclass_v,
            is_nested=is_nested,
            is_functional=is_functional,
            layout=layout,
            device=t.device,
            size=t.size(),
            stride=stride,
            storage_offset=storage_offset,
            dynamo_dynamic_indices=list(getattr(t, "_dynamo_dynamic_indices", set())),
            sparse_dim=t.sparse_dim()
            if t.is_sparse or is_sparse_compressed(t)
            else None,
            dense_dim=t.dense_dim() if t.is_sparse or is_sparse_compressed(t) else None,
            is_coalesced=t.is_coalesced() if t.is_sparse else None,
            # TODO: I actually think recursing here is correct, but we have at
            # least an infinite cycle from base -> values -> base
            # https://github.com/pytorch/pytorch/issues/122089
            crow_indices=self.describe_tensor(t.crow_indices(), recurse=False)
            if recurse and t.layout in {torch.sparse_csr, torch.sparse_bsr}
            else None,
            col_indices=self.describe_tensor(t.col_indices(), recurse=False)
            if recurse and t.layout in {torch.sparse_csr, torch.sparse_bsr}
            else None,
            ccol_indices=self.describe_tensor(t.ccol_indices(), recurse=False)
            if recurse and t.layout in {torch.sparse_csc, torch.sparse_bsc}
            else None,
            row_indices=self.describe_tensor(t.row_indices(), recurse=False)
            if recurse and t.layout in {torch.sparse_csc, torch.sparse_bsc}
            else None,
            values=self.describe_tensor(t.values(), recurse=False)
            if recurse and is_sparse_compressed(t)
            else None,
            grad=self.describe_tensor(safe_grad(t))
            if safe_grad(t) is not None
            else None,
            creation_meta=torch._C._autograd._get_creation_meta(t)
            if t._is_view()
            else None,
            unwrapped=unwrapped,
            level=maybe_get_level(t)
            if is_batchedtensor_v or is_gradtrackingtensor_v
            else None,
            bdim=maybe_get_bdim(t) if is_batchedtensor_v else None,
            base=self.describe_tensor(t._base)
            if recurse and t._is_view() and t._base is not None
            else None,
            fake_mode=torch._subclasses.fake_tensor.maybe_get_fake_mode(t),
            view_func=t._view_func_unsafe,
            attrs=attrs,
            ctx=ctx,
            type=type_v,
            # NB: even if functorch is enabled, don't actually save the
            # interpreter stack here unless we are actually functorch wrapped;
            # it's irrelevant for non-functorch stuff
            functorch_stack=maybe_functorch_stack,
            autograd_meta_from=autograd_meta_from,
            current_level=current_level,
        )


@dataclass(frozen=True)
class MetaStorageDesc:
    id: MetaStorageId
    size: int


@dataclass(frozen=True)
class MetaTensorDesc:
    id: MetaTensorId
    is_inference: bool
    is_leaf: bool
    requires_grad: bool
    ndim: int
    dtype: torch.dtype
    is_sparse: bool
    is_mkldnn: bool
    is_functorch_wrapped: bool
    is_batchedtensor: bool
    is_legacy_batchedtensor: bool
    is_gradtrackingtensor: bool
    is_view: bool
    is_nested: bool
    is_traceable_wrapper_subclass: bool
    is_functional: bool
    is_conj: bool
    is_neg: bool
    device: torch.device
    layout: torch.layout
    # NB: Sometimes, size, stride and storage_offset contain SymInt, in which
    # case this is NOT serializable.  That only happens when you're
    # re-fakeifying a fake tensor with an existing ShapeEnv... maybe we
    # can get rid of this use case entirely
    # NB: size could potentially be None as you can override it and make it
    # throw an error, but we don't currently have any subclasses that do this
    # except C++ nested tensor but we're going to have nested int to make this
    # defined on NJT
    size: Tuple[int, ...]
    dynamo_dynamic_indices: List[int]
    stride: Optional[Tuple[int, ...]] = None
    storage_offset: int = 0
    storage: Optional[MetaStorageDesc] = None
    sparse_dim: Optional[int] = None  # is_sparse, is_sparse_compressed
    dense_dim: Optional[int] = None  # is_sparse, is_sparse_compressed
    is_coalesced: Optional[bool] = None  # is_sparse
    crow_indices: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    col_indices: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    ccol_indices: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    row_indices: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    values: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    unwrapped: Optional[MetaTensorDesc] = None  # is_functorch_wrapped
    bdim: Optional[int] = None  # is_functorch_wrapped
    base: Optional[MetaTensorDesc] = None  # is_view
    attrs: Optional[Dict[str, MetaTensorDesc]] = None  # is_traceable_wrapper_subclass
    creation_meta: Optional[CreationMeta] = None
    grad: Optional[MetaTensorDesc] = None

    # Everything below is NOT serializable, need some more work
    ctx: Optional[object] = None  # is_traceable_wrapper_subclass
    type: Optional[Type] = None  # is_traceable_wrapper_subclass
    fake_mode: Optional[FakeTensorMode] = None
    view_func: Optional[
        Callable[
            [
                torch.Tensor,
                Callable[[int], int],
                Callable[[torch.Tensor], torch.Tensor],
            ],
            torch.Tensor,
        ]
    ] = None
    # level looks serializable, but actually it is meaningless without
    # the functorch_stack below
    level: Optional[int] = None  # is_functorch_wrapped
    current_level: Optional[int] = None
    functorch_stack: Optional[List[CInterpreter]] = None
    autograd_meta_from: Optional[torch.Tensor] = None

    # Faithfully serializing functorch tensors will not be too difficult.
    # We only need to consider grad/vmap interpreters, and their internal
    # state is only bools (mostly what the grad enabled/disabled state
    # should be in the lower layer).  Beyond that, tensors just need to
    # precisely indicate which particular interpreter they correspond
    # to (we then replace level with a pointer to the interpreter stack.)
    # However, this use of functorch is very "non-lexical" so it's not
    # entirely clear how to make it all lexical again, so we haven't done
    # it for now.

    @property
    def shape(self):
        return self.size


# This is a class for converting multiple tensors into meta tensors which
# share the same view/storage structure.  The operation model is you allocate
# one of these, and then call it repeatedly on all the tensors you want to
# convert.  It's important to use the same object for tensors you want to
# share storage because this is how we correlate shared storages to the same
# meta storages. This class will hold weak references to cached tenosrs
# and tensor storages.
class MetaConverter:
    def __init__(self):
        # Maps MetaStorageId to UntypedStorage
        self.storage_memo: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        # Maps MetaTensorId to torch.Tensor (typically a meta tensor or
        # FakeTensor)
        self.tensor_memo: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.hit = 0
        self.miss = 0
        self.del_hook = None
        self.arg_cnt = 0
        self.describer = MetaTensorDescriber()

    def successful(self):
        return self.hit > 0 and self.miss == 0

    def get_tensor_memo(self, t: MetaTensorDesc):
        return self.tensor_memo.get(t.id, None)

    def set_tensor_memo(self, t: MetaTensorDesc, v):
        self.tensor_memo[t.id] = v

    def get_storage_memo(self, s: MetaStorageDesc):
        return self.storage_memo.get(s.id, None)

    def set_storage_memo(self, s: MetaStorageDesc, v):
        self.storage_memo[s.id] = v

    def meta_storage(self, s: MetaStorageDesc, callback):
        if self.get_storage_memo(s) is None:
            r_s = callback(
                lambda: torch.empty(s.size, dtype=torch.uint8, device="meta")
            ).untyped_storage()
            self.set_storage_memo(s, r_s)
            return r_s
        else:
            return self.get_storage_memo(s)

    # This function assumes that it's possible to do the conversion
    # NB: name here is used in a conventional way by Dynamo; it corresponds
    # precisely to the Source.name() of the tensor we're fakeifying and
    # corresponds to a valid Python expression.  When we construct sub-names
    # as part of this process, we will maintain this invariant!  (Even though
    # other users of this may not need it this property to be upheld.)
    def meta_tensor(
        self,
        t: MetaTensorDesc,
        shape_env: Optional[ShapeEnv] = None,
        callback=lambda t: t(),
        source: Optional[Source] = None,
        symbolic_context: Optional[SymbolicContext] = None,
    ):
        if source is None:
            from torch._dynamo.source import ConstantSource

            # TODO: make a dedicated UnknownSource for this?
            source = ConstantSource(
                f"__meta_utils_unknown_tensor{len(self.tensor_memo)}"
            )

        # This indicates you set no_dispatch() before calling into this
        # function.  This is an error: we may be creating fake tensors and
        # will perform operations on them which need fake tensor mode to
        # be active.  You will segfault if you are in a no_dispatch() block.
        assert not torch._C._dispatch_tls_local_exclude_set().has(
            torch._C.DispatchKey.Python
        )
        arg_cnt = self.arg_cnt
        self.arg_cnt += 1

        # When we make as_strided calls, we end up generating a guard
        # that the new as_strided tensor is in bounds for the old storage
        # for the base (since as_strided calls can "bust" out of their
        # bounding box.)  This guard is unnecessary: if a user is able
        # to provide us a tensor with the view base setup this way, we
        # don't need to produce a guard, because the fact that they
        # were able to produce the view base means its in bounds.
        #
        # Now, ordinarily, this guard would be harmless.  However, the
        # generated guard refers to variables bound on the base variable.
        # At the moment, Dynamo doesn't actually guard on x._base, because
        # according to Voz this results in a lot of spurious invalidations,
        # and also if the user doesn't directly make use of _base, its
        # pointless anyway (because programs should be parametric over
        # whether or not the input tensor is a view or not--unless you're
        # mutating the input, but that's a whole 'nother ballgame).  So
        # for expediency, we suppress these guards so we don't have to
        # deal with this (yet, anyway.)
        #
        # NB: An old version of this code suppressed guards for ALL operations
        # happening during meta conversion, not just as_strided calls.
        # This is too aggressive: we do duck sizing and 0/1 simplification
        # as we allocate variables, and we do need to register guards for
        # these cases.
        maybe_suppress: Callable[[], Any] = contextlib.nullcontext
        if shape_env is not None:
            maybe_suppress = shape_env.suppress_guards

        def sym_sizes_strides_storage_offset(
            t: MetaTensorDesc, src, symbolic_context=symbolic_context
        ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
            assert t.stride is not None
            if shape_env is not None:
                fake_mode = t.fake_mode
                if fake_mode is not None and fake_mode.shape_env is shape_env:
                    # Don't reallocate the sizes; the shape envs are the same,
                    # so reuse the old sizes/strides/etc
                    return (t.size, t.stride, t.storage_offset)
                else:
                    # TODO: deduplicate this
                    t_size = tuple(
                        shape_env._maybe_specialize_sym_int_with_hint(sz)
                        for sz in t.size
                    )
                    t_stride = tuple(
                        shape_env._maybe_specialize_sym_int_with_hint(sd)
                        for sd in t.stride
                    )
                    t_storage_offset = shape_env._maybe_specialize_sym_int_with_hint(
                        t.storage_offset
                    )
                    return shape_env._create_symbolic_sizes_strides_storage_offset(
                        t_size,
                        t_stride,
                        t_storage_offset,
                        [d in t.dynamo_dynamic_indices for d in range(t.ndim)],
                        src,
                        symbolic_context=symbolic_context,
                    )
            else:
                return (t.size, t.stride, t.storage_offset)

        def empty_create(
            inner_t: MetaTensorDesc, inner_src, symbolic_context=symbolic_context
        ):
            (
                inner_sizes,
                inner_strides,
                inner_storage_offset,
            ) = sym_sizes_strides_storage_offset(inner_t, inner_src, symbolic_context)
            return torch.empty_strided(
                inner_sizes,
                inner_strides,
                dtype=inner_t.dtype,
                device="meta",
            )

        # Creates a subclass instance with empty inner tensors according to the specified
        # symbolic context.
        def empty_create_subclass(
            t: MetaTensorDesc,
            outer_size,
            outer_stride,
            symbolic_context=symbolic_context,
            callback=callback,
            source=source,
        ):
            from torch._dynamo.source import AttrSource
            from torch.fx.experimental.symbolic_shapes import SubclassSymbolicContext

            assert t.attrs is not None
            assert t.type is not None
            # NB: t.ctx could be None if the subclass in question has no
            # meaningful context

            assert symbolic_context is None or isinstance(
                symbolic_context, SubclassSymbolicContext
            )

            # Note: transform_subclass will use __tensor_unflatten__ to generate
            # a fresh subclass wrapper with outer sizes / strides according to the
            # outer symbolic context (passed in to this function). Inner size / stride
            # / storage offset symbols are allocated according to the appropriate inner
            # symbolic contexts, after which the checks in transform_subclass() will
            # relate them to the outer metadata as possible.
            #
            # Morally, the code here is same as transform_subclass, but we've
            # written it from scratch to read EmptyCreateSubclass

            outer_size = outer_size if outer_size is not None else t.size
            outer_stride = outer_stride if outer_stride is not None else t.stride

            transformed_tensors_dict = {
                attr: callback(
                    lambda: empty_create(
                        inner_t,
                        AttrSource(source, attr),
                        symbolic_context=(
                            None
                            if symbolic_context is None
                            else symbolic_context.inner_contexts[attr]
                        ),
                    )
                )
                for attr, inner_t in t.attrs.items()
            }

            sub = t.type.__tensor_unflatten__(
                transformed_tensors_dict, t.ctx, outer_size, outer_stride
            )

            # NB: Purposefully guard here to simplify the inner / outer symbols.
            # Using sym_eq() for symbolic comparison can result in an expression that's too
            # difficult to guard on, so we use == here.
            assert sub.shape == outer_size, (
                f"Expected return value from {t.type}__tensor_unflatten__() to have "
                f"shape equal to {outer_size}, but got: {sub.shape}"
            )
            assert sub.stride() == outer_stride, (
                f"Expected return value from {t.type}__tensor_unflatten__() to have "
                f"stride equal to {outer_stride}, but got: {sub.stride()}"
            )

            return sub

        # Returns an all-dynamic symbolic context used for metafying the given tensor with
        # fully dynamic dims. This is useful when fake-ifying intermediate tensors in
        # closed-over ViewFunc state, as we don't have symbolic contexts for them, but we
        # don't want to over-specialize during view replay.
        def all_dynamic_symbolic_context(
            t: MetaTensorDesc, source, shape_env, callback
        ):
            from torch._dynamo.source import AttrSource
            from torch.fx.experimental.symbolic_shapes import (
                DimDynamic,
                StatelessSymbolicContext,
                SubclassSymbolicContext,
            )

            view_base_context: Optional[SymbolicContext] = None
            if t.is_view:
                assert t.base is not None
                view_base_context = all_dynamic_symbolic_context(
                    t.base, AttrSource(source, "_base"), shape_env, callback
                )

            t_symbolic_context: SymbolicContext
            t_dynamic_sizes = [DimDynamic.DYNAMIC] * t.ndim
            if t.is_traceable_wrapper_subclass:
                assert t.attrs is not None
                inner_contexts: Dict[str, SymbolicContext] = {}
                for attr, inner in t.attrs.items():
                    assert isinstance(attr, str)
                    inner_contexts[attr] = all_dynamic_symbolic_context(
                        inner, AttrSource(source, attr), shape_env, callback
                    )
                t_symbolic_context = SubclassSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.ndim,
                    inner_contexts=inner_contexts,
                    tensor_source=source,
                    view_base_context=view_base_context,
                )
            else:
                t_symbolic_context = StatelessSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.ndim,
                    view_base_context=view_base_context,
                )

            return t_symbolic_context

        # Returns a fake-ified version of an input view tensor t, given an already fake-ified
        # base. At a high level, we want two things:
        #   1. fake_t should have the same view relationship to the given fake base as the
        #      input t has to its _base.
        #   2. fake_t should have symbolic sizes / strides / storage offset according to the
        #      appropriate symbolic context (i.e. from the automatic dynamic algorithm).
        #
        # We currently take different strategies across view types:
        #   * For dense -> dense views, accomplish both (1) and (2) simultaneously via an
        #     as_strided() call on the fake-ified base, passing symbolic metadata.
        #   * For views involving subclasses, perform view replay using view funcs to
        #     achieve (1). It's necessary for (2) to swap out any closed-over state in
        #     the view funcs with symbolicized SymInts and fake-ified tensors. Doing this
        #     avoids specialization (and thus over-eager simplification of symbols) that
        #     could occur during view replay on the fake-ified base.
        #
        # Examples:
        #   * t.unsqueeze(-1) with dense t is a dense -> dense view. It can be modeled
        #     with an as_strided() call on the fake base passing symbolic metadata.
        #   * sub.select(dim=0, index=3) is a subclass -> subclass view. The index arg
        #     is made symbolic to avoid invalid specialization and view replay is then
        #     done to reconstruct the view.
        #   * _nested_from_jagged(values, offsets) is a dense -> subclass view
        #     that returns a subclass instance from a dense values tensor. The offsets
        #     tensor is closed over in the view func, as it can be considered view metadata.
        #     First, the offsets tensor is fake-ified according to the inner symbolic
        #     context and with the correct relationship to the outer size / stride metadata.
        #     Then view replay is done, swapping in the fake offsets so the view replay output
        #     is fully fake with no invalid specialization.
        def view_from_base(
            base: torch.Tensor, t: MetaTensorDesc, source=source, shape_env=shape_env
        ):
            # fake-ify t's metadata according to the outer symbolic context
            (sizes, strides, storage_offset) = sym_sizes_strides_storage_offset(
                t, source
            )
            if (
                not t.is_traceable_wrapper_subclass
                and not is_traceable_wrapper_subclass(base)
            ):
                # Dense -> Dense view case uses as_strided() to construct view relationship.
                # TODO: Change this logic to use view replay for consistency?
                # It's likely there is no view func available.
                return base.as_strided(sizes, strides, storage_offset)

            from torch._dynamo.source import EphemeralSource
            from torch.fx.experimental.symbolic_shapes import sym_eq

            def symint_visitor_fn(s):
                if shape_env is None:
                    return s

                # NB: The symbol here is expected to be simplified out because we a priori
                # allocate inner and outer symbols according to the appropriate symbolic
                # contexts and prefer those over this symbol during symbol simplification
                # (via usage of EphemeralSource below). This -shouldn't- happen, but if
                # this symbol somehow leaks out beyond the view tensor's shape metadata, our
                # assumption of it being simplified out will fail and it may be guarded on,
                # which will hard error.
                sym_source = EphemeralSource("symint_visitor_fn")
                symbol = shape_env.create_symbol(s, sym_source)
                return shape_env.create_symintnode(symbol, hint=s, source=sym_source)

            real_to_fake_mapping = {}
            if t.is_traceable_wrapper_subclass:
                assert t.attrs is not None
                # NB: t.ctx could be None if the subclass in question has no
                # meaningful context
                assert t.type is not None

                # Fake-ify t naively here; this is only done so we can get fake-ified inner
                # tensors with the correct relationships to the outer sizes / strides for use
                # in view replay. It's done beforehand here because it's not easy to do when
                # visiting tensors one-by-one during view replay.
                #
                # Example:
                #   Consider a Dense -> NJT view. NJT has (values, offsets) components and we
                #   want a view of values with the offsets closed over. As the offsets component
                #   is needed to describe the output view, it's important that it's fakeified
                #   correctly.
                fake_t = empty_create_subclass(
                    t, outer_size=sizes, outer_stride=strides
                )
                attrs, _ = fake_t.__tensor_flatten__()
                for attr in attrs:
                    real_to_fake_mapping[t.attrs[attr].id] = getattr(fake_t, attr)

            def tensor_visitor_fn(
                visited_t: torch.Tensor,
                shape_env=shape_env,
                callback=callback,
                source=source,
            ):
                # It's possible to close over an undefined tensor (e.g. NJT's lengths).
                if visited_t is None:
                    return None

                # NB: visited_t being a Tensor here is very naughty!  Should
                # have already been described

                # Fake inner tensors of view subclasses will come from the mapping built above.
                visited_id = self.describer.get_tensor_id(visited_t)
                fake_visited_t = real_to_fake_mapping.get(visited_id, None)
                if fake_visited_t is not None:
                    return fake_visited_t

                visited_desc = self.describer.describe_tensor(visited_t)

                # For other closed-over tensor state, fake-ify it as all dynamic with an
                # ephemeral source. This avoids invalid specialization during view replay.
                # If we find that in practice the usage of ephemeral sources isn't enough
                # to guarantee that we don't have guards on these symbols, we may need to
                # explicitly suppress guards (as is done for _base in the dense -> dense
                # view case).
                temp_source = EphemeralSource("tensor_visitor_fn")
                return self.meta_tensor(
                    visited_desc,
                    shape_env,
                    callback,
                    source=temp_source,
                    symbolic_context=all_dynamic_symbolic_context(
                        visited_desc, temp_source, shape_env, callback
                    ),
                )

            # Replay the view, swapping out any non-symbolic SymInts or real tensors
            # for symbolic SymInts or fake tensors.
            assert t.view_func is not None
            fake_t = t.view_func(base, symint_visitor_fn, tensor_visitor_fn)

            # Ensure the output has symbolic shapes according to the outer symbolic context.
            # These checks should simplify out any symbols created for closed-over view func
            # SymInts.
            torch._check(sym_eq(fake_t.size(), sizes))
            torch._check(sym_eq(fake_t.stride(), strides))
            torch._check(sym_eq(fake_t.storage_offset(), storage_offset))
            return fake_t

        if self.get_tensor_memo(t) is None:
            GRAD_TENSOR_SENTINEL_VALUE = -2

            with torch.inference_mode(t.is_inference):
                if t.is_sparse:
                    is_leaf = t.is_leaf

                    # The lambda function below is similar to
                    # `t.to(device='meta')` except the latter
                    # preserves nnz value
                    r = callback(
                        lambda: torch.ops.aten._sparse_coo_tensor_with_dims(
                            t.sparse_dim,
                            t.dense_dim,
                            t.size,
                            dtype=t.dtype,
                            layout=torch.sparse_coo,
                            device="meta",
                        )
                    )
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    # Note [is_coalesced is dispatched]
                    # Strangely enough, is_coalesced() is a dispatched operator,
                    # which means that it will get caught by fake tensor mode.
                    # Ordinarily this would error, but there's some logic in
                    # fake tensor ensure this doesn't happen.
                    r._coalesced_(t.is_coalesced)
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        with torch.enable_grad():
                            r = r.clone()
                            r._coalesced_(t.is_coalesced)
                elif is_sparse_compressed_layout(t.layout):
                    is_leaf = t.is_leaf

                    def mk_meta():
                        assert t.sparse_dim is not None
                        assert t.dense_dim is not None
                        nnz = 0
                        batch_dim = t.ndim - t.sparse_dim - t.dense_dim
                        batch_size = t.shape[:batch_dim]
                        if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                            assert t.crow_indices is not None
                            assert t.col_indices is not None
                            index_dtype = t.crow_indices.dtype
                            compressed_indices = torch.empty(
                                t.crow_indices.shape, device="meta", dtype=index_dtype
                            )
                            plain_indices = torch.empty(
                                (*t.col_indices.shape[:-1], nnz),
                                device="meta",
                                dtype=index_dtype,
                            )
                        else:
                            assert t.ccol_indices is not None
                            assert t.row_indices is not None
                            index_dtype = t.ccol_indices.dtype
                            compressed_indices = torch.empty(
                                t.ccol_indices.shape, device="meta", dtype=index_dtype
                            )
                            plain_indices = torch.empty(
                                (*t.row_indices.shape[:-1], nnz),
                                device="meta",
                                dtype=index_dtype,
                            )
                        assert t.values is not None
                        values_shape = t.values.shape
                        values = torch.empty(
                            (
                                *values_shape[:batch_dim],
                                nnz,
                                *values_shape[batch_dim + 1 :],
                            ),
                            dtype=t.dtype,
                            device="meta",
                        )
                        return torch.ops.aten.sparse_compressed_tensor(
                            compressed_indices,
                            plain_indices,
                            values,
                            t.shape,
                            layout=t.layout,
                            dtype=t.dtype,
                            device="meta",
                        )

                    # `mk_meta()` is similar to `t.to(device='meta'))`
                    # except `to('meta')` preserves nnz value while
                    # `mk_meta` result has nnz == 0.
                    r = callback(mk_meta)

                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        with torch.enable_grad():
                            r = r.clone()
                elif t.is_nested and not t.is_traceable_wrapper_subclass:
                    # TODO: Handle this better in Dynamo?
                    # There are checks there now, but this can still be triggered by a dense
                    # tensor graph input that is a view of a strided NT.
                    from torch._dynamo.exc import unimplemented

                    unimplemented(
                        "strided nested tensors are not supported by meta conversion"
                    )
                elif t.is_mkldnn:
                    is_leaf = t.is_leaf
                    sizes, strides, _storage_offset = sym_sizes_strides_storage_offset(
                        t, source
                    )
                    r = callback(
                        lambda: torch.empty_strided(
                            sizes, strides, dtype=t.dtype, device="meta"
                        )
                    )
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        with torch.enable_grad():
                            r = r.clone()
                elif t.is_functorch_wrapped:
                    if t.is_view:
                        from torch._dynamo.exc import unimplemented

                        unimplemented(
                            "view functorch tensors are not supported by meta conversion"
                        )

                    # Wraps a functorch tensor class (BatchedTensor, GradTrackingTensor)
                    # in a FakeTensor
                    def _to_fake_tensor(t: MetaTensorDesc):
                        # TODO: why aren't the recursive calls going to
                        # meta_tensor
                        if t.is_batchedtensor:
                            assert t.unwrapped is not None
                            assert t.level is not None
                            assert t.bdim is not None
                            ft = _to_fake_tensor(t.unwrapped)
                            lvl = t.level
                            bdim = t.bdim
                            # You cannot create functorch tensors without
                            # having the ambient funtorch interpreter stack
                            # available, as the level refers to things in the
                            # stack
                            with torch._functorch.pyfunctorch.temporarily_restore_interpreter_stack(
                                t.functorch_stack
                            ):
                                r = _add_batch_dim(ft, bdim, lvl)
                        elif t.is_gradtrackingtensor:
                            assert t.unwrapped is not None
                            assert t.level is not None
                            disable_functorch = torch._C._DisableFuncTorch
                            with disable_functorch():
                                ft = _to_fake_tensor(t.unwrapped)
                            lvl = t.level
                            if lvl == GRAD_TENSOR_SENTINEL_VALUE:
                                r = ft
                            else:
                                with torch._functorch.pyfunctorch.temporarily_restore_interpreter_stack(
                                    t.functorch_stack
                                ):
                                    r = torch._C._functorch._wrap_for_grad(ft, lvl)

                            is_leaf = t.is_leaf
                            if t.requires_grad and safe_is_leaf(r):
                                r.requires_grad = True
                            elif t.requires_grad and not is_leaf:
                                with torch.enable_grad():
                                    r = r.clone()
                        elif t.is_functional:
                            assert t.unwrapped is not None
                            assert t.current_level is not None
                            ft = self.meta_tensor(
                                t.unwrapped,
                                shape_env=shape_env,
                                callback=callback,
                                # NB: reuse these exactly, we treat the
                                # functional tensor as "invisible".
                                # TODO: Actually this all probably doesn't
                                # work, take a closer look.
                                source=source,
                                symbolic_context=symbolic_context,
                            )
                            r = _wrap_functional_tensor(ft, t.current_level)
                            # TODO: is_leaf/requires_grad?
                        else:
                            assert t.stride is not None

                            sizes = t.size
                            strides = t.stride
                            r = callback(
                                lambda: torch.empty_strided(
                                    sizes,
                                    strides,
                                    dtype=t.dtype,
                                    device="meta",
                                )
                            )
                        return r

                    r = _to_fake_tensor(t)

                elif t.is_functional and t.device.type not in ["xla", "lazy"]:
                    assert t.unwrapped is not None
                    assert not t.is_functorch_wrapped  # handled above
                    unwrapped = self.meta_tensor(
                        t.unwrapped,
                        shape_env=shape_env,
                        callback=callback,
                        source=source,
                        symbolic_context=symbolic_context,
                    )
                    r = torch._to_functional_tensor(unwrapped)
                    torch._mirror_autograd_meta_to(t.autograd_meta_from, r)  # type: ignore[attr-defined]

                elif t.is_view:
                    # Construct views in two steps: recursively meta-fy their
                    # base, and then create view(s) off that.  NB: doing it
                    # directly from storage is WRONG because this won't cause
                    # version counters to get shared.

                    assert t.base is not None

                    base_symbolic_context = None
                    if shape_env and symbolic_context is not None:
                        from torch.fx.experimental.symbolic_shapes import (
                            StatelessSymbolicContext,
                        )

                        assert isinstance(symbolic_context, StatelessSymbolicContext)
                        # NB: This should generally be set when the input is a view,
                        # but the exception right now is for fake-ifying grads, which is
                        # a work in progress.
                        if symbolic_context.view_base_context is not None:
                            base_symbolic_context = symbolic_context.view_base_context

                    base = self.meta_tensor(
                        t.base,
                        shape_env,
                        callback,
                        source=torch._dynamo.source.AttrSource(source, "_base"),
                        symbolic_context=base_symbolic_context,
                    )

                    def is_c_of_r(complex_dtype, real_dtype):
                        return (
                            utils.is_complex_dtype(complex_dtype)
                            and utils.corresponding_real_dtype(complex_dtype)
                            == real_dtype
                        )

                    # In some situations, MetaConverter may be called in a
                    # context where autograd is disabled.  For the _is_view
                    # assert to pass, we have to setup the autograd view
                    # metadata anyway.  Do this by reenabling the
                    # ADInplaceOrView key.  This is kind of a hack.
                    old_exclude = torch._C._dispatch_tls_is_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView
                    )
                    torch._C._dispatch_tls_set_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView, False
                    )
                    try:
                        if base.dtype == t.dtype:
                            pass
                        elif is_c_of_r(base.dtype, t.dtype):
                            base = torch.view_as_real(base)
                        elif is_c_of_r(t.dtype, base.dtype):
                            base = torch.view_as_complex(base)
                        else:
                            # This is not guaranteed to succeed.  If it fails, it
                            # means there is another dtype-converting view function
                            # that hasn't been handled here
                            base = base.view(t.dtype)

                        # This is very tricky.  Naively, you might expect this
                        # to hold:
                        #
                        #   if t.requires_grad and not safe_is_leaf(t)
                        #       assert t._base.requires_grad
                        #
                        # But it's not true!  As you can see in the following
                        # program:
                        #
                        #   x = torch.zeros(4)
                        #   y = x.view(1, 4)
                        #   y.requires_grad = True
                        #   z = y.view(1, 1, 4)
                        #   assert z._base is x
                        #
                        # So we may have to do *two* views out of the base to
                        # recreate this situation.
                        if t.is_leaf:
                            # Leaf views that track view metadata are created by
                            # creating a view inside a no_grad block
                            with torch.no_grad(), maybe_suppress():
                                r = view_from_base(base, t)
                            # As it's a leaf, we can directly assign requires_grad
                            r.requires_grad = t.requires_grad
                        else:
                            if t.base.requires_grad == t.requires_grad:
                                # Easy case, just run the view op
                                with torch.enable_grad(), maybe_suppress():
                                    r = view_from_base(base, t)

                                # NB: We don't actaully faithfully replicate
                                # autograd connectivity, but that doesn't matter
                                # today. See following for more info:
                                # https://gist.github.com/soulitzer/e03f015b314c3f5fcf80888c69390913
                            else:
                                # Obscure case.  Create a leaf view and give it the
                                # correct requires_grad, then do the final view.
                                # NB: Can't have a non-leaf without requiring grad!
                                assert t.requires_grad
                                with torch.no_grad():
                                    mid = base.view(base.shape)
                                mid.requires_grad = t.requires_grad
                                with torch.enable_grad(), maybe_suppress():
                                    r = view_from_base(mid, t)
                        # The CreationMeta influences whether or not inplace
                        # mutation is an error or not.  So we need to make
                        # sure we properly propagate this as well.
                        assert t.creation_meta is not None
                        torch._C._autograd._set_creation_meta(r, t.creation_meta)
                    finally:
                        torch._C._dispatch_tls_set_dispatch_key_excluded(
                            torch._C.DispatchKey.ADInplaceOrView, old_exclude
                        )

                else:
                    is_leaf = t.is_leaf

                    (
                        sizes,
                        strides,
                        storage_offset,
                    ) = sym_sizes_strides_storage_offset(t, source, symbolic_context)

                    # If we have a subclass that desugars into dense tensors,
                    # perform our callback on each inner tensor.
                    if t.is_traceable_wrapper_subclass:
                        r = empty_create_subclass(
                            t, outer_size=sizes, outer_stride=strides
                        )
                    else:
                        r = callback(
                            lambda: torch.empty_strided(
                                sizes,
                                strides,
                                dtype=t.dtype,
                                device="meta",
                            )
                        )

                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = t.requires_grad
                        if not is_leaf:
                            # Fake up some autograd history.
                            with torch.enable_grad():
                                # preserve_format is the default, but we want to
                                # emphasize how important it is to preserve
                                # format here
                                r = r.clone(memory_format=torch.preserve_format)

                    # Graph-Break for wrapped tensors
                    if (
                        not (t.is_batchedtensor or t.is_gradtrackingtensor)
                        and t.is_functorch_wrapped
                    ) or t.is_legacy_batchedtensor:
                        return NotImplemented

                    s = t.storage
                    assert s is not None
                    if s.id not in self.storage_memo and (
                        r.is_nested
                        or (
                            r.stride() == strides
                            and r.storage_offset() == storage_offset
                        )
                    ):
                        # You're normal and happy, install the fresh storage into the memo
                        self.set_storage_memo(s, r.untyped_storage())
                    else:
                        # You're in crazy town; somehow you gave us a tensor
                        # that wasn't a view, but had nonzero storage offset,
                        # nontrivial strides (such that clone() couldn't
                        # preserve them), or already aliases with another
                        # tensor's storage.  The most typical way to end
                        # up here is with set_.  So use set_ to bludgeon this
                        # in.
                        r_s = self.meta_storage(s, callback=callback)
                        # NB: In principle, this should always work, but there
                        # is some subtle difference in the autograd metadata
                        # that means we will backprop the set_ call, even if
                        # r is declared as an input to grad.
                        # See https://github.com/pytorch/pytorch/issues/87956
                        # for the reproducer.
                        # NB: The in_kernel_invocation_manager here is necessary
                        # for fake tensor.  If we run the set_ call with fake
                        # tensor on, r will improperly report that it is NOT a
                        # meta tensor but a cpu tensor, and then the set_ call
                        # will fail due to device mismatch.  no_dispatch() is
                        # not enough, because the fake tensor will still claim
                        # to be a CPU tensor and you'll end up in the CPU
                        # kernel.  Arguably this is a hack; a cleaner way to
                        # solve this is to have a FakeStorage concept which
                        # would report it's CPU device--no problem now!  But
                        # this is difficult to do because we don't have storage
                        # subclasses.  Relevant test is
                        # DynamicShapesFunctionTests::test_add_dynamic_shapes in
                        # test/dynamo/test_dynamic_shapes.py
                        maybe_fake_mgr: ContextManager[None] = contextlib.nullcontext()
                        from torch._subclasses.fake_tensor import (
                            in_kernel_invocation_manager,
                            maybe_get_fake_mode,
                        )

                        mb_fake_mode = maybe_get_fake_mode(r)
                        if mb_fake_mode is not None:
                            maybe_fake_mgr = in_kernel_invocation_manager(mb_fake_mode)
                        with maybe_fake_mgr, torch.no_grad():
                            r.set_(r_s, storage_offset, sizes, strides)

                if t.grad is not None:
                    from torch._dynamo.source import AttrSource

                    # TODO: Use a valid grad-specific symbolic context instead of recycling
                    # the one from t. This isn't correct if e.g. t._is_view() != t.grad._is_view().
                    r.grad = self.meta_tensor(
                        t.grad,
                        shape_env,
                        callback,
                        source=AttrSource(source, "grad"),
                        symbolic_context=symbolic_context,
                    )
                torch._C._set_conj(r, t.is_conj)
                torch._C._set_neg(r, t.is_neg)
            # This can be skipped if necessary for performance reasons
            skip_leaf = (
                t.is_gradtrackingtensor and t.level == GRAD_TENSOR_SENTINEL_VALUE
            )
            assert_metadata_eq(assert_eq, t, r, skip_symbolic=True, skip_leaf=skip_leaf)
            self.set_tensor_memo(t, r)

        return self.get_tensor_memo(t)

    def __call__(
        self,
        t,
        shape_env=None,
        *,
        callback=lambda t: t(),
        source=None,
        symbolic_context=None,
    ):
        # TODO: zero tensors?  We appear to have eliminated them by
        # excluding complex for now

        # Filter out cases we don't support
        # TODO: This can probably be simplified quite a bit
        if isinstance(t, torch.Tensor) or is_traceable_wrapper_subclass(t):
            if (
                # Lazy tensors are not supported.  Note that XLA is
                # implemented on top of lazy tensor, not excluded here; we
                # have some special handling for it; this is for XLA Dynamo
                # integration
                t.device.type == "lazy"
                or
                # Quantization is not supported
                t.is_quantized
                or
                # Views out of sparse tensors not currently supported (plain
                # sparse is supported htough)
                (t._is_view() and t._base is not None and t._base.is_sparse)
            ):
                self.miss += 1
                return NotImplemented
            else:
                self.hit += 1
        elif torch.overrides.is_tensor_like(t):
            self.miss += 1
            return NotImplemented
        else:
            # non-Tensor types don't count as hit or miss
            return t

        # Describe the tensor.  NB: do NOT disable ambient modes, we may need
        # to query them when figuring out what to put in here
        t_desc = self.describer.describe_tensor(t)

        # Do the meta-fication.  Here, we disable all the ambient modes, to
        # better simulate what would be like to re-fakeify from a fresh
        # process
        with contextlib.ExitStack() as exit_stack:
            exit_stack.enter_context(torch._dispatch.python.suspend_functionalization())
            st = peek_interpreter_stack()
            if st is not None:
                exit_stack.enter_context(
                    torch._functorch.pyfunctorch.temporarily_clear_interpreter_stack()
                )

            r = self.meta_tensor(
                t_desc,
                shape_env=shape_env,
                callback=callback,
                source=source,
                symbolic_context=symbolic_context,
            )

        if type(t) is torch.nn.Parameter:
            # NB: Cannot directly use Parameter constructor
            # because that would force a detach, not desirable
            r._is_param = True

        # TODO: return the description for later
        return r


import torch._prims_common as utils

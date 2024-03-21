import torch
import torch.nn as nn
from einops import rearrange

from .unimatch.utils import split_feature, merge_splits


def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** 0.5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cuda"),
):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def single_head_split_window_attention(
    q,
    k,
    v,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C] for 2-view
    # for multi-view cross-attention, q: [B, L, C], k, v: [B, N-1, L, C]

    # multi(>2)-view corss-attention
    if not (q.dim() == k.dim() == v.dim() == 3):
        assert k.dim() == v.dim() == 4
        assert h is not None and w is not None
        assert q.size(1) == h * w

        m = k.size(1)  # m + 1 is num of views

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, m, h, w, c)  # [B, N-1, H, W, C]
        v = v.view(b, m, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))

        q = split_feature(
            q, num_splits=num_splits, channel_last=True
        )  # [B*K*K, H/K, W/K, C]
        k = split_feature(
            k.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )  # [B*K*K, H/K, W/K, C*(N-1)]
        v = split_feature(
            v.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )  # [B*K*K, H/K, W/K, C*(N-1)]

        k = (
            k.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 3, 1, 2, 4)
            .reshape(b_new, c, -1)
        )  # [B*K*K, C, H/K*W/K*(N-1)]
        v = (
            v.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 1, 2, 4, 3)
            .reshape(b_new, -1, c)
        )  # [B*K*K, H/K*W/K*(N-1), C]

        scores = (
            torch.matmul(q.view(b_new, -1, c), k) / scale_factor
        )  # [B*K*K, H/K*W/K, H/K*W/K*(N-1)]

        if with_shift:
            scores += attn_mask.repeat(b, 1, m)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # [B*K*K, H/K*W/K, C]

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

        out = out.view(b, -1, c)
    else:
        # 2-view self-attention or cross-attention
        assert q.dim() == k.dim() == v.dim() == 3

        assert h is not None and w is not None
        assert q.size(1) == h * w

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, h, w, c)
        v = v.view(b, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

        q = split_feature(
            q, num_splits=num_splits, channel_last=True
        )  # [B*K*K, H/K, W/K, C]
        k = split_feature(k, num_splits=num_splits, channel_last=True)
        v = split_feature(v, num_splits=num_splits, channel_last=True)

        scores = (
            torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1))
            / scale_factor
        )  # [B*K*K, H/K*W/K, H/K*W/K]

        if with_shift:
            scores += attn_mask.repeat(b, 1, 1)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*K*K, H/K*W/K, C]

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

        out = out.view(b, -1, c)

    return out


def multi_head_split_window_attention(
    q,
    k,
    v,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
    num_head=1,
):
    """Multi-head scaled dot-product attention
    Args:
        q: [N, L, D]
        k: [N, S, D]
        v: [N, S, D]
    Returns:
        out: (N, L, D)
    """

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.view(b, h, w, c)  # [B, H, W, C]
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    assert c % num_head == 0

    scale_factor = (c // num_head) ** 0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits)  # [B*K*K, H/K, W/K, C]
    k = split_feature(k, num_splits=num_splits)
    v = split_feature(v, num_splits=num_splits)

    # multi-head attn
    q = q.view(b_new, -1, num_head, c // num_head).permute(0, 2, 1, 3)  # [B, N, H*W, C]
    k = k.view(b_new, -1, num_head, c // num_head).permute(0, 2, 3, 1)  # [B, N, C, H*W]
    scores = torch.matmul(q, k) / scale_factor  # [B*K*K, N, H/K*W/K, H/K*W/K]

    if with_shift:
        scores += attn_mask.unsqueeze(1).repeat(b, num_head, 1, 1)

    attn = torch.softmax(scores, dim=-1)  # [B*K*K, N, H/K*W/K, H/K*W/K]

    out = torch.matmul(
        attn, v.view(b_new, -1, num_head, c // num_head).permute(0, 2, 1, 3)
    )  # [B*K*K, N, H/K*W/K, C]

    out = merge_splits(
        out.permute(0, 2, 1, 3).reshape(b_new, h // num_splits, w // num_splits, c),
        num_splits=num_splits,
    )  # [B, H, W, C]

    # shift back
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.view(b, -1, c)

    return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        no_ffn=False,
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        **kwargs,
    ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn
        self.add_per_view_attn = add_per_view_attn

        self.with_shift = with_shift

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        if "attn_type" in kwargs:
            attn_type = kwargs["attn_type"]
        else:
            attn_type = self.attention_type

        # source, target: [B, L, C] for 2-view
        # for multi-view cross-attention, source: [B, L, C], target: [B, N-1, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C] or [B, N-1, L, C]
        value = self.v_proj(value)  # [B, L, C] or [B, N-1, L, C]

        if attn_type == "swin" and attn_num_splits > 1:
            if self.nhead > 1:
                message = multi_head_split_window_attention(
                    query,
                    key,
                    value,
                    num_splits=attn_num_splits,
                    with_shift=self.with_shift,
                    h=height,
                    w=width,
                    attn_mask=shifted_window_attn_mask,
                    num_head=self.nhead,
                )
            else:
                if self.add_per_view_attn:
                    assert query.dim() == 3 and key.dim() == 4 and value.dim() == 4
                    b, l, c = query.size()
                    query = query.unsqueeze(1).repeat(
                        1, key.size(1), 1, 1
                    )  # [B, N-1, L, C]
                    query = query.view(-1, l, c)  # [B*(N-1), L, C]
                    key = key.view(-1, l, c)
                    value = value.view(-1, l, c)
                    message = single_head_split_window_attention(
                        query,
                        key,
                        value,
                        num_splits=attn_num_splits,
                        with_shift=self.with_shift,
                        h=height,
                        w=width,
                        attn_mask=shifted_window_attn_mask,
                    )
                    # [B, L, C]  # add per view attn
                    message = message.view(b, -1, l, c).sum(1)
                else:
                    message = single_head_split_window_attention(
                        query,
                        key,
                        value,
                        num_splits=attn_num_splits,
                        with_shift=self.with_shift,
                        h=height,
                        w=width,
                        attn_mask=shifted_window_attn_mask,
                    )
        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(TransformerBlock, self).__init__()

        self.no_cross_attn = no_cross_attn

        if no_cross_attn:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )
        else:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                no_ffn=True,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
            )

            self.cross_attn_ffn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(
            source,
            source,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        if self.no_cross_attn:
            return source

        # cross attention and ffn
        source = self.cross_attn_ffn(
            source,
            target,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        return source


def batch_features(features):
    # construct inputs to multi-view transformer in batch
    # features: list of [B, C, H, W] or [B, H*W, C]

    # query, key and value for transformer
    q = []
    kv = []

    num_views = len(features)

    for i in range(num_views):
        x = features.copy()
        q.append(x.pop(i))  # [B, C, H, W] or [B, H*W, C]

        # [B, N-1, C, H, W] or [B, N-1, H*W, C]
        kv.append(torch.stack(x, dim=1))

    q = torch.cat(q, dim=0)  # [N*B, C, H, W] or [N*B, H*W, C]
    kv = torch.cat(kv, dim=0)  # [N*B, N-1, C, H, W] or [N*B, N-1, H*W, C]

    return q, kv


class MultiViewFeatureTransformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        d_model=128,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(MultiViewFeatureTransformer, self).__init__()

        self.attention_type = attention_type

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    attention_type=attention_type,
                    ffn_dim_expansion=ffn_dim_expansion,
                    with_shift=(
                        True if attention_type == "swin" and i % 2 == 1 else False
                    ),
                    add_per_view_attn=add_per_view_attn,
                    no_cross_attn=no_cross_attn,
                )
                for i in range(num_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # zero init layers beyond 6
        if num_layers > 6:
            for i in range(6, num_layers):
                self.layers[i].self_attn.norm1.weight.data.zero_()
                self.layers[i].self_attn.norm1.bias.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.weight.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.bias.data.zero_()

    def forward(
        self,
        multi_view_features,
        attn_num_splits=None,
        **kwargs,
    ):
        if "attn_type" in kwargs and kwargs["attn_type"] == "epipolar":
            assert len (multi_view_features) == 2, "Only support 2 views for Epipolar Transformer"
            feature0, feature1 = multi_view_features
            return self.forward_epipolar(feature0, feature1)

        # multi_view_features: list of [B, C, H, W]
        b, c, h, w = multi_view_features[0].shape
        assert self.d_model == c

        num_views = len(multi_view_features)

        if self.attention_type == "swin" and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=multi_view_features[0].device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # [N*B, C, H, W], [N*B, N-1, C, H, W]
        concat0, concat1 = batch_features(multi_view_features)
        concat0 = concat0.reshape(num_views * b, c, -1).permute(
            0, 2, 1
        )  # [N*B, H*W, C]
        concat1 = concat1.reshape(num_views * b, num_views - 1, c, -1).permute(
            0, 1, 3, 2
        )  # [N*B, N-1, H*W, C]

        for i, layer in enumerate(self.layers):
            concat0 = layer(
                concat0,
                concat1,
                height=h,
                width=w,
                shifted_window_attn_mask=shifted_window_attn_mask,
                attn_num_splits=attn_num_splits,
            )

            if i < len(self.layers) - 1:
                # list of features
                features = list(concat0.chunk(chunks=num_views, dim=0))
                # [N*B, H*W, C], [N*B, N-1, H*W, C]
                concat0, concat1 = batch_features(features)

        features = concat0.chunk(chunks=num_views, dim=0)
        features = [
            f.view(b, h, w, c).permute(0, 3, 1, 2).contiguous() for f in features
        ]

        return features

    def forward_epipolar(self, source, target):
        """
        source: [b v c h w]
        target: [b v 1 ray sample c]
        """
        assert self.d_model == source.shape[2] == target.shape[-1]
        b, v, c, h, w = source.shape

        source = rearrange(source, "b v c h w -> (b v h w) () c")
        target = rearrange(target, "b v () r s c -> (b v r) s c")

        for _, layer in enumerate(self.layers):
            # NOTE: target is not changed, following exactly pixelsplat, wierd though...
            source = layer(source=source, target=target, attn_type="full")

        source = rearrange(source, "(b v h w) () c -> b v c h w", b=b, v=v, h=h, w=w)

        return source

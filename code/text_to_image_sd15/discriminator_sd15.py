import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from typing import Union, Optional, Dict, Any, Tuple
from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)


def modified_forward(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    r"""
    The [`UNet2DConditionModel`] forward method.
    Args:
        sample (`torch.FloatTensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        class_labels (`torch.Tensor`, *optional*, defaults to `None`):
            Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
            Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
            through the `self.time_embedding` layer to obtain the timestep embeddings.
        attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
            A tuple of tensors that if specified are added to the residuals of down unet blocks.
        mid_block_additional_residual: (`torch.Tensor`, *optional*):
            A tensor that if specified is added to the residual of the middle unet block.
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
            tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added to UNet long skip connections from down blocks to up blocks for
            example from ControlNet side model(s)
        mid_block_additional_residual (`torch.Tensor`, *optional*):
            additional residual to be added to UNet mid block output, for example from ControlNet side model
        down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
    Returns:
        [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
            a `tuple` is returned where the first element is the sample tensor.
    """
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers
    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None
    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break
    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)
    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (
            1 - encoder_attention_mask.to(sample.dtype)
        ) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0
    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = self.time_proj(timesteps)
    # `Timesteps` does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=sample.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None
    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError(
                "class_labels should be provided when num_class_embeds > 0"
            )
        if self.config.class_embed_type == "timestep":
            class_labels = self.time_proj(class_labels)
            # `Timesteps` does not contain any weights and will always return f32 tensors
            # there might be better ways to encapsulate this.
            class_labels = class_labels.to(dtype=sample.dtype)
        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb
    if self.config.addition_embed_type == "text":
        aug_emb = self.add_embedding(encoder_hidden_states)
    elif self.config.addition_embed_type == "text_image":
        # Kandinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
        aug_emb = self.add_embedding(text_embs, image_embs)
    elif self.config.addition_embed_type == "text_time":
        # SDXL - style
        if "text_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
            )
        text_embeds = added_cond_kwargs.get("text_embeds")
        if "time_ids" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
            )
        time_ids = added_cond_kwargs.get("time_ids")
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
    elif self.config.addition_embed_type == "image":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        aug_emb = self.add_embedding(image_embs)
    elif self.config.addition_embed_type == "image_hint":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        hint = added_cond_kwargs.get("hint")
        aug_emb, hint = self.add_embedding(image_embs, hint)
        sample = torch.cat([sample, hint], dim=1)
    emb = emb + aug_emb if aug_emb is not None else emb
    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)
    if (
        self.encoder_hid_proj is not None
        and self.config.encoder_hid_dim_type == "text_proj"
    ):
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
    elif (
        self.encoder_hid_proj is not None
        and self.config.encoder_hid_dim_type == "text_image_proj"
    ):
        # Kadinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )
        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(
            encoder_hidden_states, image_embeds
        )
    elif (
        self.encoder_hid_proj is not None
        and self.config.encoder_hid_dim_type == "image_proj"
    ):
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )
        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(image_embeds)
    elif (
        self.encoder_hid_proj is not None
        and self.config.encoder_hid_dim_type == "ip_image_proj"
    ):
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )
        image_embeds = added_cond_kwargs.get("image_embeds")
        image_embeds = self.encoder_hid_proj(image_embeds)
        encoder_hidden_states = (encoder_hidden_states, image_embeds)
    # 2. pre-process
    sample = self.conv_in(sample)
    # 2.5 GLIGEN position net
    if (
        cross_attention_kwargs is not None
        and cross_attention_kwargs.get("gligen", None) is not None
    ):
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}
    # 3. down
    lora_scale = (
        cross_attention_kwargs.get("scale", 1.0)
        if cross_attention_kwargs is not None
        else 1.0
    )

    down_block_res_samples = (sample,)

    output_features = []

    for downsample_block in self.down_blocks:
        if (
            hasattr(downsample_block, "has_cross_attention")
            and downsample_block.has_cross_attention
        ):
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb, scale=lora_scale
            )

        output_features.append(sample)
        down_block_res_samples += res_samples

    # 4. mid
    if self.mid_block is not None:
        if (
            hasattr(self.mid_block, "has_cross_attention")
            and self.mid_block.has_cross_attention
        ):
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)

        output_features.append(sample)

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1
        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]
        if (
            hasattr(upsample_block, "has_cross_attention")
            and upsample_block.has_cross_attention
        ):
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
                scale=lora_scale,
            )
        output_features.append(sample)
    # 6. post-process

    return output_features


class DiscriminatorHead(nn.Module):
    def __init__(self, input_channel, output_channel=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, 1, 1),
            nn.GroupNorm(32, input_channel),
            nn.LeakyReLU(inplace=True), # use LeakyReLu instead of GELU shown in the paper to save memory
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, 1, 1),
            nn.GroupNorm(32, input_channel),
            nn.LeakyReLU(inplace=True), # use LeakyReLu instead of GELU shown in the paper to save memory
        )

        self.conv_out = nn.Conv2d(input_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv_out(x)
        return x


class Discriminator(nn.Module):

    def __init__(
        self,
        unet,
        num_h_per_head=4,
        adapter_channel_dims=[320, 640, 1280, 1280, 1280, 1280, 1280, 640, 320],
    ):
        super().__init__()
        self.unet = unet
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)
        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DiscriminatorHead(adapter_channel)
                        for _ in range(self.num_h_per_head)
                    ]
                )
                for adapter_channel in adapter_channel_dims
            ]
        )

    def _forward(self, sample, timestep, encoder_hidden_states):
        features = modified_forward(self.unet, sample, timestep, encoder_hidden_states)
        assert self.head_num == len(features)
        outputs = []
        for feature, head in zip(features, self.heads):
            for h in head:
                outputs.append(h(feature))
        return outputs

    def forward(self, flag, *args):
        if flag == "d_loss":
            return self.d_loss(*args)
        elif flag == "g_loss":
            return self.g_loss(*args)
        else:
            assert 0, "not supported"

    def d_loss(self, sample_fake, sample_real, timestep, encoder_hidden_states, weight):
        loss = 0.0
        fake_outputs = self._forward(
            sample_fake.detach(), timestep, encoder_hidden_states
        )
        real_outputs = self._forward(
            sample_real.detach(), timestep, encoder_hidden_states
        )
        for fake_output, real_output in zip(fake_outputs, real_outputs):
            loss += (
                torch.mean(weight * torch.relu(fake_output.float() + 1))
                + torch.mean(weight * torch.relu(1 - real_output.float()))
            ) / (self.head_num * self.num_h_per_head)
        return loss

    def g_loss(self, sample_fake, timestep, encoder_hidden_states, weight):
        loss = 0.0
        fake_outputs = self._forward(sample_fake, timestep, encoder_hidden_states)
        for fake_output in fake_outputs:
            loss += torch.mean(weight * torch.relu(1 - fake_output.float())) / (
                self.head_num * self.num_h_per_head
            )
        return loss

    def feature_loss(
        self, sample_fake, sample_real, timestep, encoder_hidden_states, weight
    ):
        loss = 0.0
        features_fake = modified_forward(
            self.unet, sample_fake, timestep, encoder_hidden_states
        )
        features_real = modified_forward(
            self.unet, sample_real.detach(), timestep, encoder_hidden_states
        )
        for feature_fake, feature_real in zip(features_fake, features_real):
            loss += torch.mean((feature_fake - feature_real) ** 2) / (self.head_num)
        return loss


if __name__ == "__main__":
    teacher_unet = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5",
        subfolder="unet",
    )
    teacher_unet.cuda()
    discriminator = Discriminator(teacher_unet).cuda()

    sample = torch.randn((1, 4, 64, 64)).cuda()
    encoder_hidden_states = torch.randn((1, 77, 768)).cuda()
    timesteps = torch.randn((1,)).cuda()

    features = discriminator(sample, timesteps, encoder_hidden_states)

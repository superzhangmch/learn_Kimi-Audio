# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import os
import json
from pathlib import Path
from typing import Optional, Union, Dict

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from .activations import Snake, SnakeBeta
from .utils import init_weights, get_padding
from .alias_free_activation.torch.act import Activation1d as TorchActivation1d
from .utils import AttrDict

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,        ## kernel_size 为[3,5,7,11]中之一。
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,     # input_channel
                        channels,     # output_channel. note: out_channel = in_channel
                        kernel_size,  # kernel_size
                        stride=1,     # stride
                        dilation=d,   # dilation，依次为 1,3,5 之一。默认是1.对于 dilation > 1, 表示卷积核矩阵和input矩阵关联算conv的时候，连续的两个kernel 元素，要对应input 矩阵上距离多远的两个元素。
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,  # stride=1， dilation=1， in_chanlel = out_channel， 所以是最普通的 conv
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        # note： convs1, convs2 都是 stride=1 的，且 input_channels = output_chanels, 也就是说，它们都不会改变 input 的 shape
        
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from .alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d
        # paper 中，AMP 中的 snake 激活前后有上下采样。这个上下采样都被包装在了 Activation1d 里（CudaActivation1d 或 TorchActivation1d里）

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)    # snake 激活
            xt = c1(xt)
            xt = a2(xt)   # snake 激活
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from .alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url="https://github.com/NVIDIA/BigVGAN/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Args:
        h (AttrDict): Hyperparameters.
        use_cuda_kernel (bool): If set to True, loads optimized CUDA kernels for AMP. This should be used for inference only, as training is not supported with CUDA kernels.

    Note:
        - The `use_cuda_kernel` parameter should be used for inference only, as training with CUDA kernels is not supported.
        - Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from .alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default （但是看起来 paper中是用了 AMPBlock2）
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            '''
            print ('xxxxxxxxxx00000', 'i, u, k', i, u, k)
            依次输出（u= stride, k = kernel_size)
            xxxxxxxxxx00000 i, u, k 0 5 9
            xxxxxxxxxx00000 i, u, k 1 2 4
            xxxxxxxxxx00000 i, u, k 2 2 4
            xxxxxxxxxx00000 i, u, k 3 2 4
            xxxxxxxxxx00000 i, u, k 4 2 4
            xxxxxxxxxx00000 i, u, k 5 3 5
            xxxxxxxxxx00000 i, u, k 6 2 4

            ConvTranspose1d 的序列变化公式是： L_out​= (L_in​−1) * stride − 2 * padding + kernel_size， 也就是变成了几倍，主要看 stride是几
            
            '''
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),           ## in_channels
                                h.upsample_initial_channel // (2 ** (i + 1)),   ## out_channels = in_channels / 2
                                k,                                              ## kernel_size, 依次是 9 4 4 4 4 5 4 
                                u,                                              ## stride,      依次是 5 2 2 2 2 3 2 （从而序列长度的变化倍数，与此有关）
                                padding=(k - u) // 2,                           ## padding      依次是 2 1 1 1 1 1 1 
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))  ### channel 数量每次减半
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):  # k = kernel_size, 依次为 3,5,7,11。 d=dilation=(1,3,5)
                self.resblocks.append(resblock_class(h, ch, k, d, activation=h.activation))

        # Post-conv
        activation_post = (Snake(ch, alpha_logscale=h.snake_logscale)  # snake 乃每个通道走自己独立的 alpha。F_snake = x + 1/alpha * (sin(alpha * x))^2
                        if h.activation == "snake"
                        else (
                            SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                            if h.activation == "snakebeta"
                            else None
                        )
            )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        # x.torch.Size([1, 80, 136]) = [bs, dim_of_mel=80, seq_len=136]
        # Pre-conv
        x = self.conv_pre(x) # 

        for i in range(self.num_upsamples):
            '''
            每循环一次，对于 x: channel 数减半，seq_len 加倍：
            torch.Size([1, 1024, 136]) = [bs, channel, seq_len]
            torch.Size([1, 1024, 680])
            torch.Size([1, 512, 1360])
            torch.Size([1, 256, 2720])
            torch.Size([1, 128, 5440])
            torch.Size([1, 64, 10880])
            torch.Size([1, 32, 32640]) 
            '''
            # Upsampling
            for i_up in range(len(self.ups[i])): # 实际这里循环只执行一次。乃单循环
                x = self.ups[i][i_up](x)

            '''
            up-sample 后 x.shape: 
            torch.Size([1, 1024, 680])
            torch.Size([1, 512, 1360])
            torch.Size([1, 256, 2720])
            torch.Size([1, 128, 5440])
            torch.Size([1, 64, 10880])
            torch.Size([1, 32, 32640]) 
            torch.Size([1, 16, 65280])
            '''
            # AMP blocks
            xs = None
            for j in range(self.num_kernels): # 实际这里循环 4 次：每一次的 kernel_size 分别是 3， 5， 7， 11. 
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x) # 每个 resblocks，也就是每个 AMPBlock 内部，会有 1,3 5 三种 dilation
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels 

        # Post-conv
        # input_x.shape = torch.Size([1, 16, 65280])
        x = self.activation_post(x)
        
        # input_x.shape = torch.Size([1, 16, 65280])
        x = self.conv_post(x)
        
        # Final tanh activation
        # input_x.shape = torch.Size([1, 1, 65280])
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        # x.shape = torch.Size([1, 1, 65280])
        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",  # Additional argument
        strict: bool = False,  # Additional argument
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="bigvgan_generator.pt",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        checkpoint_dict = torch.load(
            model_file, map_location=map_location, weights_only=True
        )

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model

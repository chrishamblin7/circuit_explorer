# Copyright 2020 The lucent_circuit Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import torch
import numpy as np
from lucent_circuit.optvis.param import color


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TORCH_VERSION = torch.__version__


def pixel_image(shape, sd=None,start_params=None,magic=None,device=device):
    sd = sd or 0.01

    if start_params is None:
        tensor = (torch.randn(*shape) * sd).to(device).requires_grad_(True)
    else:
        tensor = start_params.to(device).requires_grad_(True)
    return [tensor], lambda: tensor


# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)

def fft_image(shape, sd=None, magic=None, decay_power=1, start_params=None,device=device):
    import torch

    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, channels) + freqs.shape + (2,) # 2 for imaginary and real components
    sd = sd or 0.01
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)
    #import pdb; pdb.set_trace()

    if start_params is None:
        spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
    else:
        spectrum_real_imag_t = start_params
    # else:      #convert image to spectrum
    #     if TORCH_VERSION >= "1.7.0":
    #         import torch.fft
    #         spectrum_real_imag_t = (torch.fft.rfft(start_image,2,norm='ortho')).to(device).requires_grad_(True)
    #     else:
    #         import torch
    #         spectrum_real_imag_t = (torch.rfft(start_image, 2, normalized=True)).to(device).requires_grad_(True)

    magic = magic or 30.0
    

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if TORCH_VERSION >= "1.7.0":
            import torch.fft
            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        else:
            import torch
            image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        image = image[:batch, :channels, :h, :w]
        #if magic is None:
        #    magic = 4.0 # Magic constant from Lucid library; increasing this seems to reduce saturation
        image = image / magic
        image.retain_grad()
        return image
    return [spectrum_real_imag_t], inner


def image_2_fourier(pixel_image, saturation=4.0):
  '''
  pixel image is a torch tensor of shape (batch,channel,h,w),
  returns a fourier image that can be used as start_params
  '''
  device = pixel_image.device
  import torch
  TORCH_VERSION = torch.__version__
  shape = pixel_image.shape
  batch, channels, h, w = shape
  freqs = rfft2d_freqs(h, w)
  init_val_size = (batch, channels) + freqs.shape + (2,) # 2 for imaginary and real components
  scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h))
  scale = torch.tensor(scale).float()[None, None, ..., None].to(device)
  image = torch.tensor(pixel_image)
  image = torch.logit(image).permute(0, 2, 3, 1)
  image = torch.matmul(image,torch.inverse(torch.tensor(color.color_correlation_normalized.T).to(image.device)))
  image = image.permute(0,3,1,2)
  image = image * saturation
  if TORCH_VERSION >= "1.7.0":
      import torch.fft
      fourier_image = torch.fft.rfftn(image, norm='ortho',s=(h, w))
      fourier_image = torch.view_as_real(fourier_image)
  else:
      fourier_image = torch.fft.rfftn(image, 2, normalized=True, signal_sizes=(h, w))
  fourier_image = fourier_image/scale
  return fourier_image

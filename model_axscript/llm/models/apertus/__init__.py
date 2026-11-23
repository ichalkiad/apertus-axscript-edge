# SPDX-License-Identifier: Apache-2.0
"""Apertus 1.5 (swiss-ai): the PyTorch reference architecture plus its Metis kernel.

Registration is NOT an import side effect of this package (see the note in
`models.qwen3`). The reference model (`.apertus`, which needs torch) and the device kernel
(`.metis_impl`, which does not) are registered by `models.architecture.register_architectures`
and `implementation.register_kernels`, through the two names this port adds to
`_BUNDLED_ARCHITECTURES` and `_BUNDLED_KERNELS` (patches/register-apertus.patch).
`.xielu_lut` is the calibration-free g(m) = expm1(m) - m table both of them use.
"""

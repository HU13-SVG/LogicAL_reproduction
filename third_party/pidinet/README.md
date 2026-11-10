# PiDiNet (trimmed)

Source: official implementation of "Pixel Difference Networks for Efficient
Edge Detection" by Z. Su, W. Liu, Z. Yu, D. Hu, Q. Liao, Q. Tian,
M. Pietikäinen and L. Liu (ICCV 2021; TPAMI 2023 extension).

The following files are copied from the upstream repository:

```text
models/__init__.py
models/config.py
models/convert_pidinet.py
models/hed_fps.py
models/ops.py
models/ops_theta.py
models/pidinet.py
LICENSE
```

They are used to instantiate the Carv4 + SA + dilated PiDiNet used for
first-scale edge extraction.  No functional changes were made.

The upstream licence permits non-commercial research use; see `LICENSE`.

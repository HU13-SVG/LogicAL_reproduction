# PiDiNet checkpoint

`table5_pidinet.pth` is the official PiDiNet checkpoint
(`table5_pidinet.pth` from the PiDiNet release,
<https://github.com/zhuo-su/PiDiNet>) used by the paper for first-scale edge
extraction.  It is loaded with
`models.pidinet(SimpleNamespace(config="carv4", sa=True, dil=True))` and kept
frozen.

SHA-256:

```text
80860ac267258b5f27486e0ef152a211d0b08120f62aeb185a050acc30da486c
```

Verify with `sha256sum checkpoints/table5_pidinet.pth` (Git Bash/Linux) or
`Get-FileHash checkpoints/table5_pidinet.pth -Algorithm SHA256` (PowerShell).

The checkpoint and the vendored PiDiNet source are distributed for
non-commercial research use under the PiDiNet licence; see
`../third_party/pidinet/LICENSE`.

Paper: Z. Su, W. Liu, Z. Yu, D. Hu, Q. Liao, Q. Tian, M. Pietikäinen, L. Liu,
"Pixel Difference Networks for Efficient Edge Detection", ICCV 2021 (TPAMI
2023 extension).

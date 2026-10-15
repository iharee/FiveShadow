# 运行模型

- `local/`：完整修复管线需要的FaceParsingV2、四个修复模型及其浮点权重、图结构和几何模板。程序不读取已安装软件或隔壁研究目录。
- `detection/`：YuNet人脸检测、MediaPipe关键点模型包，以及YuNet许可文本；下载地址与SHA-256在 `detection_models.py` 固定。

这些运行参数在当前工作目录内保留。缺少自动定位参数时，可执行 `uv run stubble models` 从官方来源补齐；手动框选加三点模式不依赖定位模型。

模型资产不因项目代码的MIT许可而改变授权，见根目录THIRD_PARTY.md。

## `local/` 资产清单

拷贝、备份或替换 `local/` 内的文件后，可用下表核对字节数与 SHA-256（路径相对 `local/`）。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `dunnite_networks.json` | 1812049 | `9449c11efbc5dd1bc6588faa991d85a82fa7fcf3605c4adebffa40128b272807` |
| `beard_alignment_template.json` | 9108 | `55362372bcf0c313bbf8b6089d4af83bc58e4c0268a49e8e8e963b36a308edbf` |
| `parsing_models/dunnite_networks.json` | 155377 | `34ecb62b1dcedc06c916bd9fc89b82ee4f633405c1bb54ee9dd27f32f9a6635b` |
| `parsing_models/2102828093.decoded.bin` | 23048833 | `a9440779fe459e69f76b43655aa838fcaf46264a14b5c0b87e2096a8f9c14c00` |
| `decoded_models/3378116080.decoded.bin` | 7318717 | `409a96664aa232df1575a6964e5fe2fcf5c8913741dd11b6e4785e6f15edf015` |
| `recovered_weights/3378116080.float.npz` | 28249740 | `fa39dadd39800264ae5f6dc114e8f9ecc1f69f390795b6ed99d428c8c5e423a1` |
| `recovered_weights/3378116080.float.json` | 86034 | `f40ed493d6f2d74e2ad899bacc55f5391b088e0cf3012667273bf9d882cdc954` |
| `decoded_models/195031232.decoded.bin` | 7224497 | `ccb2e596e43c9cdef47dbd59d09a7193638994555673d5a257ed157285517fd9` |
| `recovered_weights/195031232.float.npz` | 27852520 | `4dccff5dba8197f498c915b13894203e2aeb1fef0eeacf7061555ac4acec7d3c` |
| `recovered_weights/195031232.float.json` | 131490 | `372d6769a92eaec9f32efc93f20ad3dded298c0154807a32cb388f9163f9e27d` |
| `decoded_models/4083080210.decoded.bin` | 7208025 | `d9d3e7705f3f64cb7b21f08e6444d735c754a3bb75dc56b6f41b798d9cfea144` |
| `recovered_weights/4083080210.float.npz` | 27807086 | `e8706e3901e8d5a5115dd2c7741adf52f7be8d7d12b8b5632462e099e953e21e` |
| `recovered_weights/4083080210.float.json` | 85754 | `412d656625de428040999f9fe70bdbf73952013f2853c96fdee10ac97f90cc49` |
| `decoded_models/2127140479.decoded.bin` | 7318717 | `14d0e59ad77ddffab284a104ea3b44a8c3a6de25e846805e2d1c2faa950b2473` |
| `recovered_weights/2127140479.float.npz` | 28249740 | `c3a1b5acac941e4382bc087054b08e7391f7d907940fb6eb8a3b3bf1cf6cab2c` |
| `recovered_weights/2127140479.float.json` | 86062 | `3bba08a6581bcfd992c08eca1ed6c107b6b87c15ea15732ac6856f11a2825f6e` |

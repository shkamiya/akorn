# AKOrN 実験結果：分析手法別比較 (2025年7月)

このドキュメントは、AKOrN (Artificial Kuramoto Oscillator Networks) のパラメータスイープ実験結果を、分析手法ごとに整理して比較表示したものです。

## 実験概要

- **データセット**: CIFAR-10
- **Gamma (γ)**: 0.01, 0.1, 1.0 (Euler stepの時間間隔)
- **T**: 3, 7, 15, 31, 63 (時間ステップ数)
- **層数**: 3層 (Layer 0, 1, 2)
- **分析手法**: 静的接続性分析、行列分解、ネットワーク特性

---

## 1. Connectivity Statistics

### γ=0.01, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/connectivity_statistics.png) |

### γ=0.01, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/connectivity_statistics.png) |

### γ=0.01, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/connectivity_statistics.png) |

### γ=0.01, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/connectivity_statistics.png) |

### γ=0.01, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/connectivity_statistics.png) |

### γ=0.1, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/connectivity_statistics.png) |

### γ=0.1, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/connectivity_statistics.png) |

### γ=0.1, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/connectivity_statistics.png) |

### γ=0.1, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/connectivity_statistics.png) |

### γ=0.1, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/connectivity_statistics.png) |

### γ=1.0, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/connectivity_statistics.png) |

### γ=1.0, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/connectivity_statistics.png) |

### γ=1.0, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/connectivity_statistics.png) |

### γ=1.0, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/connectivity_statistics.png) |

### γ=1.0, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/connectivity_statistics.png) |

---

## 2. Symmetric/Skew Decomposition

### γ=0.01, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/decomposition_symmetric_skew.png) |

### γ=0.01, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/decomposition_symmetric_skew.png) |

### γ=0.01, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/decomposition_symmetric_skew.png) |

### γ=0.01, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/decomposition_symmetric_skew.png) |

### γ=0.01, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/decomposition_symmetric_skew.png) |

### γ=0.1, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/decomposition_symmetric_skew.png) |

### γ=0.1, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/decomposition_symmetric_skew.png) |

### γ=0.1, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/decomposition_symmetric_skew.png) |

### γ=0.1, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/decomposition_symmetric_skew.png) |

### γ=0.1, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/decomposition_symmetric_skew.png) |

### γ=1.0, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/decomposition_symmetric_skew.png) |

### γ=1.0, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/decomposition_symmetric_skew.png) |

### γ=1.0, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/decomposition_symmetric_skew.png) |

### γ=1.0, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/decomposition_symmetric_skew.png) |

### γ=1.0, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/decomposition_symmetric_skew.png) |

---

## 3. Rotation/Symmetric Decomposition

### γ=0.01, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/decomposition_rotation_symmetric.png) |

### γ=0.01, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/decomposition_rotation_symmetric.png) |

### γ=0.01, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/decomposition_rotation_symmetric.png) |

### γ=0.01, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/decomposition_rotation_symmetric.png) |

### γ=0.01, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/decomposition_rotation_symmetric.png) |

### γ=0.1, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/decomposition_rotation_symmetric.png) |

### γ=0.1, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/decomposition_rotation_symmetric.png) |

### γ=0.1, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/decomposition_rotation_symmetric.png) |

### γ=0.1, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/decomposition_rotation_symmetric.png) |

### γ=0.1, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/decomposition_rotation_symmetric.png) |

### γ=1.0, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/decomposition_rotation_symmetric.png) |

### γ=1.0, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/decomposition_rotation_symmetric.png) |

### γ=1.0, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/decomposition_rotation_symmetric.png) |

### γ=1.0, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/decomposition_rotation_symmetric.png) |

### γ=1.0, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/decomposition_rotation_symmetric.png) |

---

## 4. Frobenius Norms - Channels

### γ=0.01, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/frob_norms_channels.png) |

### γ=0.01, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/frob_norms_channels.png) |

### γ=0.01, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/frob_norms_channels.png) |

### γ=0.01, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/frob_norms_channels.png) |

### γ=0.01, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/frob_norms_channels.png) |

### γ=0.1, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/frob_norms_channels.png) |

### γ=0.1, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/frob_norms_channels.png) |

### γ=0.1, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/frob_norms_channels.png) |

### γ=0.1, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/frob_norms_channels.png) |

### γ=0.1, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/frob_norms_channels.png) |

### γ=1.0, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/frob_norms_channels.png) |

### γ=1.0, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/frob_norms_channels.png) |

### γ=1.0, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/frob_norms_channels.png) |

### γ=1.0, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/frob_norms_channels.png) |

### γ=1.0, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/frob_norms_channels.png) |

---

## 5. Frobenius Norms - Kernel

### γ=0.01, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/frob_norms_kernel.png) |

### γ=0.01, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/frob_norms_kernel.png) |

### γ=0.01, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/frob_norms_kernel.png) |

### γ=0.01, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/frob_norms_kernel.png) |

### γ=0.01, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/frob_norms_kernel.png) |

### γ=0.1, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/frob_norms_kernel.png) |

### γ=0.1, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/frob_norms_kernel.png) |

### γ=0.1, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/frob_norms_kernel.png) |

### γ=0.1, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/frob_norms_kernel.png) |

### γ=0.1, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/frob_norms_kernel.png) |

### γ=1.0, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/frob_norms_kernel.png) |

### γ=1.0, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/frob_norms_kernel.png) |

### γ=1.0, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/frob_norms_kernel.png) |

### γ=1.0, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/frob_norms_kernel.png) |

### γ=1.0, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/frob_norms_kernel.png) |

---

## 6. Omega Distributions

### γ=0.01, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/omega_distributions.png) |

### γ=0.01, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/omega_distributions.png) |

### γ=0.01, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/omega_distributions.png) |

### γ=0.01, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/omega_distributions.png) |

### γ=0.01, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/omega_distributions.png) |

### γ=0.1, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/omega_distributions.png) |

### γ=0.1, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/omega_distributions.png) |

### γ=0.1, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/omega_distributions.png) |

### γ=0.1, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/omega_distributions.png) |

### γ=0.1, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/omega_distributions.png) |

### γ=1.0, T=3 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/omega_distributions.png) |

### γ=1.0, T=7 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/omega_distributions.png) |

### γ=1.0, T=15 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/omega_distributions.png) |

### γ=1.0, T=31 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/omega_distributions.png) |

### γ=1.0, T=63 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/omega_distributions.png) | ![omega_distributions](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/omega_distributions.png) |

---

## 分析手法別観察事項

### 1. Connectivity Statistics
- **低γ値**: 統計値にばらつきが見られる
- **高γ値**: より均一で安定した統計分布
- **高T値**: 統計値の収束性向上

### 2. Symmetric/Skew Decomposition
- **対称成分**: γ増加で構造化が進む
- **歪対称成分**: T値と共に安定化
- **層間差異**: Layer 2で最も顕著な変化

### 3. Rotation/Symmetric Decomposition
- **回転成分**: パラメータ依存性が明確
- **対称成分**: γ=1.0で最適化
- **Costa & Aguiar分解**: 理論との対応確認

### 4. Frobenius Norms
- **Channels**: γ=1.0, T=63で最小値
- **Kernel**: 層間で異なるパターン
- **最適化**: 高γ・高Tで顕著な改善

### 5. Omega Distributions
- **分布形状**: パラメータ変化で明確な違い
- **層間変化**: 特にLayer 2で顕著
- **収束性**: T増加で分布の安定化

### 最適条件の確認
**γ=1.0, T=63**が全ての分析手法で最も優れた特性を示し、理論と実装の一貫性を確認できる。

---

*作成日: 2025年7月11日*  
*実験データ: /results/sweep_connectivity_analysis_20250712/*
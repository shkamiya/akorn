# AKOrN 実験結果図表集 (2025年7月)

このドキュメントは、AKOrN (Artificial Kuramoto Oscillator Networks) のパラメータスイープ実験結果を、gamma値ごとに整理して視覚的に示したものです。

## 実験概要

- **データセット**: CIFAR-10
- **Gamma (γ)**: 0.01, 0.1, 1.0 (Euler stepの時間間隔)
- **T**: 3, 7, 15, 31, 63 (時間ステップ数)
- **層数**: 3層 (Layer 0, 1, 2)
- **分析手法**: 静的接続性分析、行列分解、ネットワーク特性

---

## Gamma = 0.01 

### T = 3 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T3/layer2/frob_norms_kernel.png) |

### T = 7 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T7/layer2/frob_norms_kernel.png) |

### T = 15 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T15/layer2/frob_norms_kernel.png) |

### T = 31 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T31/layer2/frob_norms_kernel.png) |

### T = 63 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.01_T63/layer2/frob_norms_kernel.png) |

---

## Gamma = 0.1 

### T = 3 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T3/layer2/frob_norms_kernel.png) |

### T = 7 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T7/layer2/frob_norms_kernel.png) |

### T = 15 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T15/layer2/frob_norms_kernel.png) |

### T = 31 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T31/layer2/frob_norms_kernel.png) |

### T = 63 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma0.1_T63/layer2/frob_norms_kernel.png) |

---

## Gamma = 1.0 

### T = 3 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T3/layer2/frob_norms_kernel.png) |

### T = 7 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T7/layer2/frob_norms_kernel.png) |

### T = 15 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T15/layer2/frob_norms_kernel.png) |

### T = 31 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T31/layer2/frob_norms_kernel.png) |

### T = 63 

#### Connectivity Statistics 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/connectivity_statistics.png) | ![connectivity_statistics](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/connectivity_statistics.png) |

#### Symmetric/Skew Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/decomposition_symmetric_skew.png) | ![decomposition_symmetric_skew](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/decomposition_symmetric_skew.png) |

#### Rotation/Symmetric Decomposition 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/decomposition_rotation_symmetric.png) | ![decomposition_rotation_symmetric](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/decomposition_rotation_symmetric.png) |

#### Frobenius Norms - Channels 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/frob_norms_channels.png) | ![frob_norms_channels](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/frob_norms_channels.png) |

#### Frobenius Norms - Kernel 
| Layer 0 | Layer 1 | Layer 2 |
|---------|---------|---------|
| ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer0/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer1/frob_norms_kernel.png) | ![frob_norms_kernel](../results/sweep_connectivity_analysis_20250712/gamma1.0_T63/layer2/frob_norms_kernel.png) |

---

## 主要な観察事項

### Gamma値による変化
1. **γ=0.01**: 小さい時間間隔、細かいステップでの更新
2. **γ=0.1**: 中程度の時間間隔、バランスの取れた更新
3. **γ=1.0**: 大きい時間間隔、より大きなステップでの更新

### T値による変化
1. **T=3**: 高速だが不安定
2. **T=7-15**: 中間的な動力学
3. **T=31-63**: 長期安定化、特にγ=1.0で最適

### 層間の違い
- **Layer 0**: 入力に近い層、より多様なパターン
- **Layer 1**: 中間層、特徴抽出に関与
- **Layer 2**: 出力層、分類タスクに最適化

### 最適条件
**γ=1.0, T=63** の組み合わせで最も小さなFrobenius norm値を達成し、時間間隔が大きく、長時間ステップでの安定した動力学を示している。

---

*作成日: 2025年7月11日*  
*実験データ: /results/sweep_connectivity_analysis_20250712/*
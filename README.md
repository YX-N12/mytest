# DCT-SVD Digital Watermarking

基于DCT-SVD变换的鲁棒数字水印算法实现与评估系统。本项目实现了经典的Sverdlov-Dexter-Eskicioglu DCT-SVD水印算法，并提供了纯DCT基线方法进行对比分析。

## 🌟 特性

- **DCT-SVD水印算法**：结合离散余弦变换(DCT)和奇异值分解(SVD)的鲁棒水印方法
- **12种攻击测试**：包括高斯模糊、噪声、像素化、JPEG压缩等常见图像处理攻击
- **对比实验**：与纯DCT基线方法进行性能对比
- **可视化分析**：提供多种图表和热力图展示水印提取效果
- **Python实现**：使用NumPy、OpenCV和Matplotlib等科学计算库

## 📊 性能评估指标

| 指标 | 描述 | 取值范围 |
|------|------|----------|
| **PSNR** | 峰值信噪比 (dB) | [0, ∞] |
| **SSIM** | 结构相似性指数 | [-1, 1] |
| **SV Correlation** | 奇异值相关度 | [-1, 1] |
| **Pixel Correlation** | 像素相关度 | [-1, 1] |

## 📁 项目结构

```
watermarking_dct_svd/
├── README.md                    # 项目文档
├── config.py                    # 配置文件
├── main.py                      # 主程序入口
├── watermark_algorithm.py       # DCT-SVD水印算法实现
├── baseline_dct.py              # 纯DCT基线算法实现
├── attacks.py                   # 攻击方法实现
├── metrics.py                   # 评估指标计算
├── utils.py                     # 工具函数
├── visualization.py             # 可视化模块
└── output/                      # 输出目录
    ├── watermarked/             # 含水印图像
    ├── attacks/                 # 受攻击图像
    ├── extracted/               # 提取的水印（各攻击）
    ├── baseline/                # 基线方法结果
    ├── plots/                   # 可视化图表
    └── tables/                  # 结果表格
```

## 🚀 快速开始

### 环境要求

```bash
# Python 3.6+
pip install numpy opencv-python matplotlib scikit-image pandas
```

### 运行实验

```bash
# 进入项目目录
cd watermarking_dct_svd

# 运行完整实验
python main.py
```

### 输入图像

项目需要以下输入图像（放在根目录）：
- `lena_640x480.png` - 主图像载体 (640×480)
- `watermark.png` - 水印图像（将被自动调整到240×320）

## 📈 实验结果

主要输出文件：

### 可视化图表
- `output/plots/all_attacks_grid.png` - DCT-SVD所有攻击下的提取水印合并图
- `output/plots/all_baseline_attacks_grid.png` - 基线方法所有攻击下的提取水印合并图
- `output/plots/baseline_comparison.png` - DCT-SVD vs 基线方法对比
- `output/plots/intensity_curves_all.png` - 鲁棒性vs攻击强度曲线
- `output/plots/heatmap_sv_corr.png` - 奇异值相关度热力图
- `output/plots/heatmap_pixel_corr.png` - 像素相关度热力图

### 数据表格
- `output/tables/sv_correlation_table.csv` - 奇异值相关度表格
- `output/tables/pixel_correlation_table.csv` - 像素相关度表格
- `output/tables/baseline_pixel_corr.csv` - 基线方法像素相关度
- `output/tables/psnr_ssim.csv` - PSNR和SSIM值

## 🔧 算法原理

### DCT-SVD 水印嵌入

1. **DCT变换**：对载体图像进行2D DCT变换
2. **四象限分割**：使用Zigzag扫描将DCT系数分为4个象限
3. **SVD分解**：对每个象限进行奇异值分解
4. **水印嵌入**：
   - 对水印进行DCT变换和SVD分解
   - 修改载体图像的奇异值：σ* = σ + α × σ_watermark
5. **重建图像**：重建各象限，进行逆DCT变换

### DCT-SVD 水印提取

1. **DCT变换**：对可能受攻击的含水印图像进行DCT变换
2. **四象限分割**：同样分为4个象限
3. **奇异值恢复**：σ_watermark^ = (σ* - σ_original) / α
4. **水印重建**：重建水印并逆DCT变换

## 🎯 攻击类型

| 攻击类型 | 参数 | 描述 |
|----------|------|------|
| Gaussian Blur | Kernel Size | 高斯模糊 |
| Gaussian Noise | Std Dev | 高斯噪声 |
| Pixelation | Block Size | 像素化 |
| JPEG Compression | Quality Factor | JPEG压缩 |
| JPEG2000 | Compression Ratio | JPEG2000压缩 |
| Sharpening | Strength | 锐化 |
| Rescaling | Scale Factor | 尺寸缩放 |
| Rotation | Angle (°) | 旋转 |
| Symmetric Cropping | Crop % | 对称裁剪 |
| Contrast Adjustment | Brightness Offset | 对比度调整 |
| Histogram Equalization | CLAHE Clip Limit | 直方图均衡化 |
| Gamma Correction | Gamma Value | 伽马校正 |

## ⚙️ 配置说明

在 `config.py` 中可以调整以下参数：

```python
# 嵌入强度（不同象限）
ALPHA = {'B1': 0.25, 'B2': 0.01, 'B3': 0.01, 'B4': 0.01}

# 图像尺寸
COVER_SHAPE = (480, 640)   # 主图像尺寸
QUAD_SHAPE  = (240, 320)   # 象限尺寸
WM_SHAPE    = (240, 320)   # 水印尺寸

# 攻击默认参数
ATTACK_DEFAULTS = {
    'gaussian_blur': 5,
    'gaussian_noise': 20,
    # ... 其他攻击参数
}
```

## 📊 结果分析

### 主要发现

1. **鲁棒性对比**：DCT-SVD方法在大多数攻击下比纯DCT方法更鲁棒
2. **象限差异**：B1象限（低频区域）的嵌入效果最好，但更容易被攻击影响
3. **攻击敏感性**：
   - 几何攻击（旋转、缩放）对水印提取影响最大
   - 噪声攻击相对影响较小
   - JPEG压缩在高质量因子下影响较小

### 可视化建议

1. **对比两张grid图**：DCT-SVD和Baseline的合并grid图直观显示算法差异
2. **分析热力图**：快速识别哪些攻击对哪些象限影响最大
3. **查看强度曲线**：理解水印鲁棒性随攻击强度的变化

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📚 参考文献

1. Sverdlov, A., Dexter, S., & Eskicioglu, A. M. (2005). DCT-SVD domain watermarking: Embedding data in singular values and an improved DCT-based method. In Security, Steganography, and Watermarking of Multimedia Contents VII (Vol. 5681, pp. 261-272).

2. Podilchuk, C. I., & Zeng, W. (1998). Image-adaptive watermarking using visual models. IEEE Journal on Selected Areas in Communications, 16(4), 525-539.
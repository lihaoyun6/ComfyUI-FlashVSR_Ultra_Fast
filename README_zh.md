# ComfyUI-FlashVSR_Ultra_Fast
在低显存环境下运行 FlashVSR，同时保持无伪影高质量输出。  
**[[📃English](./readme.md)]**

## 更新日志
#### 2025-10-24
- 新增长视频管道, 可显著降低长视频放大的显存用量  

#### 2025-10-21
- 项目首次发布, 引入了`tile_dit`等功能, 大幅度降低显存需求  

#### 2025-10-22
- 使用`Sparse_SageAttention`替换了`Block-Sparse-Attention`, 无需编译安装任何自定义内核, 开箱即用.  
- 支持在 RTX50 系列显卡上运行.  

## 预览
![](./img/preview.jpg)

## 使用说明
- **mode（模式）：**  
  `tiny` → 更快（默认）；`full` → 更高质量  
- **scale（放大倍数）：**  
  通常使用 `4` 效果更好；如果显存不足，可使用 `2`  
- **color_fix（颜色修正）：**  
  使用小波变换方法修正输出视频的颜色偏差。  
- **tiled_vae（VAE分块解码）：**  
  启用后可显著降低显存占用，但会降低解码速度。  
- **tiled_dit（DiT分块计算）：**  
  大幅减少显存占用，但会降低推理速度。  
- **tile_size / tile_overlap（分块大小与重叠）：**  
  控制输入视频在推理时的分块方式。  
- **unload_dit（卸载DiT模型）：**  
  解码前卸载 DiT 模型以降低显存峰值，但会略微降低速度。  

## 安装步骤

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Fast/requirements.txt
```
> 📢: 要在RTX20系或更早的GPU上运行, 请安装`triton<3.3.0`:  

> ```bash
# Windows
python -m pip install -U triton-windows<3.3.0
# Linux
python -m pip install -U triton<3.3.0
```

#### 模型下载:
- 从[这里](https://huggingface.co/JunhaoZhuang/FlashVSR)下载整个`FlashVSR`文件夹和它里面的所有文件, 并将其放到`ComfyUI/models`目录中。  

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## 致谢
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous

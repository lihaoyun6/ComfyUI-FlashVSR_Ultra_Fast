# ComfyUI-FlashVSR_Ultra_Fast
在低显存环境下运行 FlashVSR，同时保持无伪影高质量输出。  
**[[📃English](./readme.md)]**

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
⚠️ 预编译的`Block-Sparse-Attention`安装包仅支持 torch2.7+cu128 环境, 不支持 torch2.8!  
⚠️ 如果你正在使用 torch2.8 或更高版本, 请在下载本插件前自行编译安装`Block-Sparse-Attention`  
⚠️ 参考下方附录中的"编译 Block-Sparse-Attention"小节

```bash
#如果确定安装的是torch2.7+cu128, 请执行下列命令安装
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Fast/requirements.txt
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
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) @mit-han-lab
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous

## 附录
### 编译 Block-Sparse-Attention:

1. 首先确保你安装了 MSVC 编译环境和 CUDAToolkit  
2. 运行下列命令来进行编译安装:  

```bash
git clone https://github.com/lihaoyun6/Block-Sparse-Attention
cd Block-Sparse-Attention
pip install packaging
pip install ninja
set MAX_JOBS=4 #Linux用户请执行: export MAX_JOBS=4
python setup.py install
```

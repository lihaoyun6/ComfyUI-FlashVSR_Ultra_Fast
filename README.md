# ComfyUI-FlashVSR_Ultra_Fast
Running FlashVSR on lower VRAM without any artifacts.   
**[[中文版本](./readme_zh.md)]**

## Preview
![](./img/preview.jpg)

## Usage
- **mode:**  
`tiny` -> faster (default); `full` -> higher quality  
- **scale:**  
`4` is always better, unless you are low on VRAM then use `2`    
- **color_fix:**  
Use wavelet transform to correct the color of output video.  
- **tiled_vae:**  
Set to True for lower VRAM consumption during decoding at the cost of speed.  
- **tiled_dit:**  
Significantly reduces VRAM usage at the cost of speed.
- **tile\_size, tile\_overlap**:  
How to split the input video.  
- **unload_dit:**  
Unload DiT before decoding to reduce VRAM peak at the cost of speed.  

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Fast/requirements.txt
```

## Acknowledgments
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) @mit-han-lab
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous

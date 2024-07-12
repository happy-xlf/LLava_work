# LLava架构DIY

## LLava组装
第一步保存组装模型：build_model_show.ipynb
其中的线性层是随机初始化，因此测试VQA效果很差，这也是正常的～～
### 视觉层
视觉层选取**openai/clip-vit-large-patch14-336**
### 语言层
语言层选取**Qwen1.5-4B-Chat**

## 数据构建
### 数据集
liuhaotian/LLaVA-CC3M-Pretrain-595K
下载后可进行解压缩，查看图片
```bash
unzip -d images_dl images.zip
```
### 数据集构建
数据浏览可见：make_dataset.ipynb
Dataset构建可见train_llava文件夹

## 训练
### 训练方式
基于deepspeed-zero2，有lora训练、全量参数训练、冻结视觉层进行训练等方式。
### 训练策略

| 训练方式                         | 视觉层  | 转接层  | 语言层        | 效果评估     |
|------------------------------|------|------|------------|----------|
| `--train_type use_lora`      | 冻结🧊 | 训练🔥 | 训练🔥（部分参数） | 效果非常好 👍 |
| `--train_type none`          | 训练🔥 | 训练🔥 | 训练🔥       | 效果非常差👎  |
| `--train_type freeze_vision` | 冻结🧊 | 训练🔥 | 训练🔥（全量参数  | 尚未评估     |

1. 训练的时候，使用lora方式进行训练最好。在`run_zero2.sh`里面设置`--train_type use_lora`即可。
2. 全量参数训练，效果非常差。
### 启动
```bash
bash run_zero2.py
```
### 查看日志
```bash
tail -f llava_train_0712.log
```

## 推理
推理可见：llava_infer_lora.ipynb
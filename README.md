# BGE Reranker 微调项目

本项目用于使用 `FlagEmbedding` 微调 `BAAI/bge-reranker-v2-m3` 模型。

## 环境设置

本项目使用 `uv` 进行依赖管理。

1.  初始化环境并安装依赖：
    ```bash
    uv sync
    ```
    或者手动安装：
    ```bash
    uv add FlagEmbedding torch transformers peft deepspeed accelerate
    ```

## 数据准备

训练数据应为 JSONL 格式。示例文件已提供在 `data/toy_finetune_data.jsonl`。
格式如下：
```json
{"query": "查询文本", "pos": ["正例列表"], "neg": ["负例列表"], "prompt": "提示词"}
```

## 训练

要开始训练，请运行 `train.sh` 脚本：

```bash
./train.sh
```

该脚本使用 `torchrun` 在 4 张 GPU 上启动分布式训练。
它使用了 `FlagEmbedding.finetune.reranker.encoder_only.base` 模块，该模块适用于 `BAAI/bge-reranker-v2-m3` 模型（基于 XLM-RoBERTa 的 Encoder-Only 模型）。

## 推理验证

训练完成后，可以使用 `inference.py` 脚本验证模型效果：

```bash
python inference.py
```

## 配置说明

-   `stage1.json`: DeepSpeed Zero Stage 1 配置文件。
-   `train.sh`: 包含训练参数（学习率、批次大小等）的启动脚本。

# 项目概览

这个仓库现在以 `../train-guard-dashboard` 的 agent 协议为准。

## 核心结构

```text
train_guard/
  agent.py              训练侧主入口，负责拉配置和协调上传
  config.py             本地配置管理与 dashboard 协议适配
  metrics_collector.py  生成 dashboard 友好的扁平数值指标
  uploader.py           负责带鉴权头的 HTTP 上报
  cli.py                本地配置读写工具
examples/
  quick_start.py
  advanced_examples.py
tests/
  test_agent.py
```

## 当前协议

- 远程配置接口：`GET /api/agent/config`
- 指标上报接口：`POST /api/metrics/ingest`
- 请求头：`X-Access-Key-Id`、`X-Secret-Key`、`X-Project-Id`

## 设计重点

- `TrainingAgent` 继续保留低侵入式训练 API
- `MetricsCollector` 输出扁平标量事件，便于 dashboard worker 直接抽取时序点
- `MetricsUploader` 统一处理重试、异步上报和 agent 鉴权头
- `config.py` 允许代码传参和 CLI 持久化两种接入方式

## 验证命令

```bash
uv run python -m unittest -v tests.test_agent
```

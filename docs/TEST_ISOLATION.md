# 测试隔离说明

当前测试目标是两件事：

- 不读取用户本地真实凭证
- 不发出真实 HTTP 请求

## 风险点

如果直接实例化 `TrainingAgent()`，它会调用 `load_config()`，而用户本地配置里可能包含：

```json
{
  "backend": {
    "base_url": "http://localhost:8000",
    "access_key_id": "ak_xxx",
    "secret_key": "sk_xxx",
    "project_id": "project_abcd1234"
  }
}
```

这会触发真实的 `GET /api/agent/config`。

## 当前做法

- 在 `TestTrainingAgent` 和 `TestIntegration` 中 patch `train_guard.agent.load_config`
- 在 uploader 相关测试中 patch `requests.Session.post`
- 在远程配置测试中 patch `train_guard.config.requests.get`

这样测试既不会访问真实 dashboard，也不会泄露用户本地 AK/SK。

## 回归命令

```bash
uv run python -m unittest -v tests.test_agent
```

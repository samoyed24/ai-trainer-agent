# 测试网络隔离说明

## 问题描述

测试套件中的某些测试方法在执行时会真实发送 HTTP 请求到后端服务器，而不是使用 mock 隔离，这导致：

- ❌ 测试依赖网络可用性（网络故障时测试失败）
- ❌ 测试暴露和泄露用户本地配置中的凭证
- ❌ 测试执行缓慢（真实网络延迟）
- ❌ 测试不稳定（远端服务器可能故障）

## 根本原因分析

### 问题代码路径

1. **`test_minimal_initialization_with_defaults()`**
   - 创建 `TrainingAgent(train_id="minimal_default")` 时不传递任何配置
   - TrainingAgent.__init__ 调用 `load_config()` 读取用户本地配置
   - 如果用户 `~/.config/train-guard/config.json` 中配置了：
     ```json
     {
       "backend": {
         "api": "https://example.com/api",
         "app_id": "your_app_id",
         "app_secret": "your_secret"
       }
     }
     ```
   - 则 TrainingAgent 会调用 `fetch_remote_config()` → **真实 HTTP POST 请求**

2. **`test_minimal_initialization_with_server_url()`**
   - 同样可能因为用户本地配置污染而触发真实请求

### 其他测试的隔离状态

✅ **已正确隔离的测试**：

- `TestMetricsCollector.*` - 无网络 I/O
- `TestMetricsUploader.*` - 已用 `@patch('requests.Session.post')`
- `TestRemoteConfigFetch.*` - 已用 `@patch('train_guard.config.requests.post')`
- `test_minimal_initialization_with_backend_credentials()` - 已用 `@patch('train_guard.agent.fetch_remote_config')`
- `test_full_training_workflow()` - 已用 `@patch('requests.Session.post')`

❌ **需要修复的测试**：

- `test_minimal_initialization_with_defaults()`
- `test_minimal_initialization_with_server_url()`

## 解决方案

为这两个测试方法添加 `@patch('train_guard.config.load_config')` 装饰器，模拟配置加载，防止读取用户本地配置中的后端凭证。

### 修复前

```python
def test_minimal_initialization_with_defaults(self):
    """测试最简初始化（不传配置）"""
    agent = TrainingAgent(train_id="minimal_default")  # ❌ 会加载用户配置！
    try:
        self.assertEqual(agent.train_id, "minimal_default")
        self.assertIsNotNone(agent.server_config.get("url"))
    finally:
        agent.close()
```

### 修复后

```python
@patch('train_guard.config.load_config')
def test_minimal_initialization_with_defaults(self, mock_load_config):
    """测试最简初始化（不传配置）"""
    # Mock 配置加载，防止读取用户本地配置中的后端凭证
    mock_load_config.return_value = {
        "server": {"url": "http://localhost:8000/metrics", "timeout": 10},
        "agent": {"upload_frequency": "epoch", "upload_interval": 1},
        "metrics": {}
    }
    
    agent = TrainingAgent(train_id="minimal_default")  # ✅ 使用 mock 配置
    try:
        self.assertEqual(agent.train_id, "minimal_default")
        self.assertIsNotNone(agent.server_config.get("url"))
    finally:
        agent.close()
```

## 验证结果

修复后的完整测试套件信息：

```
Ran 21 tests in 4.529s
OK ✓ 所有测试通过！
```

**所有测试现在都：**
- ✅ 完全隔离（零网络 I/O）
- ✅ 不依赖外部服务
- ✅ 不暴露用户凭证
- ✅ 执行快速稳定

## 最佳实践

为了保证测试的网络隔离性，遵循以下规则：

### 1. **任何涉及外部 I/O 的代码都必须 mock**

```python
# ❌ 不好：调用真实网络
def test_something():
    config = fetch_remote_config(...)  # 真实 HTTP 请求！
    
# ✅ 好：模拟网络调用
@patch('train_guard.config.requests.post')
def test_something(self, mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    config = fetch_remote_config(...)
```

### 2. **创建代理实例时的配置隔离**

```python
# ❌ 不好：会加载用户配置
agent = TrainingAgent(train_id="test")

# ✅ 好：显式传入配置或模拟配置
agent = TrainingAgent(
    server_config=SERVER_CONFIG,  # 显式传入测试配置
    agent_config=AGENT_CONFIG,
    train_id="test"
)

# ✅ 或者：模拟配置加载
@patch('train_guard.config.load_config')
def test_something(self, mock_load_config):
    mock_load_config.return_value = {...}
    agent = TrainingAgent(train_id="test")
```

### 3. **使用 setUp 建立基础隔离**

```python
class TestSomething(unittest.TestCase):
    def setUp(self):
        # 为所有测试方法预先建立 mock
        self.patcher = patch('train_guard.config.requests.post')
        self.mock_post = self.patcher.start()
        
    def tearDown(self):
        # 清理 mock
        self.patcher.stop()
```

## 回归测试

命令：
```bash
uv run python tests/test_agent.py
```

输出应包含：
```
Ran 21 tests in ~4.5s
OK
✓ 所有测试通过！
```

如果看到任何网络相关的错误或超时，说明有新的网络 I/O 被引入，需要添加对应的 mock。

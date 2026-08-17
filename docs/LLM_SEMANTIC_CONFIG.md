# LLM语义分析配置指南

## 概述

LLM语义分析模块用于解决关键词匹配无法处理的复杂语义问题：
- 区分模拟盘与实盘
- 区分闲聊与正式观点
- 处理反讽、隐喻、暗示等复杂语义
- 提取真实交易意图

## 启用方式

### 方式一：使用OpenAI API（推荐）

```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key"
export ENABLE_LLM_ANALYSIS="true"

# 可选：使用自定义API地址（如代理）
export OPENAI_BASE_URL="https://your-proxy.com/v1"
```

### 方式二：使用Claude API

```bash
# 设置环境变量
export ANTHROPIC_API_KEY="your-api-key"
export ENABLE_LLM_ANALYSIS="true"
```

### 方式三：使用本地模型（Ollama）

```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型（推荐qwen2.5:7b，中文理解能力强）
ollama pull qwen2.5:7b

# 启动Ollama服务
ollama serve

# 设置环境变量
export ENABLE_LLM_ANALYSIS="true"
```

## 环境变量说明

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `ENABLE_LLM_ANALYSIS` | 是否启用LLM分析 | `true` |
| `OPENAI_API_KEY` | OpenAI API密钥 | - |
| `OPENAI_BASE_URL` | OpenAI API地址 | `https://api.openai.com/v1` |
| `ANTHROPIC_API_KEY` | Claude API密钥 | - |

## 分析策略

系统采用混合分析策略：

1. **关键词优先**：对于明确的买入/卖出信号（如"买了"、"清仓"），直接使用关键词匹配
2. **LLM兜底**：对于以下情况，使用LLM进行深度分析：
   - 置信度低于0.6
   - 检测到否定词
   - 内容包含假设性讨论（"如果...就..."）
   - 内容包含反讽/玩笑标记（"呵呵"、"笑死"）
   - 内容包含模拟盘标记（"模拟"、"虚拟"）
   - 同时存在买入和卖出关键词

## 场景类型

LLM分析会识别以下场景类型：

| 场景类型 | 说明 | 置信度调整 |
|----------|------|------------|
| `real_trading` | 实盘交易 | 不调整 |
| `simulation` | 模拟盘/假设 | ×0.5 |
| `discussion` | 市场讨论 | ×0.7 |
| `example` | 举例说明 | 降为watch |
| `review` | 复盘总结 | 不调整 |

## 测试

```bash
# 运行测试脚本
python3 test_llm_semantic.py
```

## 成本估算

- OpenAI gpt-4o-mini: 约$0.0001/条微博
- Claude Haiku: 约$0.0001/条微博
- 本地模型: 免费

## 示例

### 输入
```
"如果明天中际旭创突破前高，我就考虑买入"
```

### 关键词分析结果
```
信号: buy, 置信度: 1.00
原因: 关键词: 买入
```

### LLM分析结果
```
场景类型: simulation (假设性讨论)
真实意图: bullish (看多)
交易信号: watch (观望，非实际操作)
是否实盘: false
置信度: 0.5
原因: [模拟盘] 这是假设性讨论，博主并未实际买入
```

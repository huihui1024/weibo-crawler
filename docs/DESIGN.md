# 微博抓取系统 — 详细设计文档

> 版本: 1.0 | 最后更新: 2026-05-21

---

## 目录

1. [系统概述](#1-系统概述)
2. [系统架构](#2-系统架构)
3. [数据模型](#3-数据模型)
4. [爬虫引擎](#4-爬虫引擎)
5. [内容解析与清洗](#5-内容解析与清洗)
6. [图片处理与OCR](#6-图片处理与ocr)
7. [语义分析引擎](#7-语义分析引擎)
8. [股票关联分析](#8-股票关联分析)
9. [选股与预测模型](#9-选股与预测模型)
10. [模拟交易系统](#10-模拟交易系统)
11. [实时告警系统](#11-实时告警系统)
12. [通知推送系统](#12-通知推送系统)
13. [定时任务调度](#13-定时任务调度)
14. [Web管理后台](#14-web管理后台)
15. [代理与反爬策略](#15-代理与反爬策略)
16. [日志与监控](#16-日志与监控)
17. [部署架构](#17-部署架构)
18. [配置体系](#18-配置体系)
19. [数据流全景](#19-数据流全景)
20. [扩展性与未来规划](#20-扩展性与未来规划)

---

## 1. 系统概述

### 1.1 项目定位

微博抓取系统是一个面向金融舆情场景的端到端数据处理平台。系统从微博平台持续抓取大V和财经博主的发帖内容，经过解析、清洗、语义分析后，提取股票市场情绪信号，驱动选股模型和模拟交易，并通过多渠道实时推送告警。

### 1.2 核心能力

| 能力域 | 描述 |
|--------|------|
| 数据采集 | 多账号轮换、代理池、自动翻页、增量抓取 |
| 内容理解 | LLM语义分析、情感评分、关键词提取、图片OCR |
| 股票关联 | 股票代码/名称识别、情绪-股票映射、热度追踪 |
| 量化选股 | 多因子模型、情绪因子融合、回测框架 |
| 模拟交易 | 仓位管理、止盈止损、交易记录、收益统计 |
| 实时告警 | 语义告警检测、多级告警、去重与聚合 |
| 通知推送 | 微信(企业微信/Server酱)、邮件、Webhook |
| 运维管理 | Web Dashboard、定时调度、日志、健康检查 |

### 1.3 技术栈

- **语言**: Python 3.11+
- **Web框架**: FastAPI + Uvicorn
- **数据库**: SQLite (可迁移至 PostgreSQL)
- **ORM**: SQLAlchemy
- **HTTP客户端**: httpx (异步)
- **LLM**: OpenAI API 兼容接口 (支持 DeepSeek / 通义千问等)
- **OCR**: PaddleOCR
- **调度**: APScheduler
- **容器化**: Docker + Docker Compose
- **前端**: Jinja2 模板 + Bootstrap

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        微博抓取系统架构                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  微博API  │◄───│   爬虫引擎    │◄───│   代理池 & Cookie管理    │  │
│  │  (数据源) │    │  (Crawler)   │    │   (Proxy & Auth)        │  │
│  └──────────┘    └──────┬───────┘    └──────────────────────────┘  │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    内容解析与清洗 (Parser)                     │  │
│  │   HTML去标签 → 去重 → 关键词提取 → 股票实体识别               │  │
│  └──────────┬───────────────────────────────────┬───────────────┘  │
│             │                                   │                  │
│             ▼                                   ▼                  │
│  ┌────────────────────┐            ┌──────────────────────┐       │
│  │   图片处理 (Image)  │            │  语义分析 (Semantic)  │       │
│  │   下载 → OCR → 存储 │            │  LLM → 情感 → 摘要   │       │
│  └────────┬───────────┘            └──────────┬───────────┘       │
│           │                                    │                   │
│           └──────────────┬─────────────────────┘                   │
│                          ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      SQLite 数据库                            │  │
│  │   weibos | images | stock_mentions | alerts | trades | ...   │  │
│  └──────────┬───────────────────────────────────────────────────┘  │
│             │                                                       │
│     ┌───────┼───────────────────┐                                   │
│     ▼       ▼                   ▼                                   │
│  ┌──────┐ ┌──────────┐  ┌──────────────┐                          │
│  │ 选股  │ │ 模拟交易  │  │  告警 & 推送  │                          │
│  │ 模型  │ │ 系统     │  │  系统        │                          │
│  └──────┘ └──────────┘  └──────────────┘                          │
│                                     │                              │
│                                     ▼                              │
│                          ┌──────────────────┐                      │
│                          │  通知渠道         │                      │
│                          │  微信 | 邮件 | Webhook │                │
│                          └──────────────────┘                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Web Dashboard (FastAPI + Jinja2)                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              APScheduler 定时任务调度                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
main.py (入口)
  ├── config.py ← config.yaml
  ├── database.py ← models.py
  ├── scheduler.py
  │     ├── crawler/weibo.py
  │     │     ├── crawler/parser.py
  │     │     ├── crawler/image.py
  │     │     └── utils/proxy.py
  │     ├── analyzer/stock_analyzer.py
  │     ├── analyzer/llm_semantic.py
  │     ├── analyzer/enhanced_semantic.py
  │     ├── analyzer/image_ocr.py
  │     ├── notifier/notifier.py
  │     │     ├── notifier/wechat.py
  │     │     ├── notifier/email_notify.py
  │     │     └── notifier/webhook.py
  │     ├── notifier/realtime_alert.py
  │     ├── notifier/semantic_alert_detector.py
  │     ├── selector/stock_selector.py
  │     │     ├── selector/predictor.py
  │     │     ├── selector/multi_factor_model.py
  │     │     └── selector/backtest.py
  │     ├── trading/simulated_trading.py
  │     │     └── trading/position_manager.py
  │     └── reviewer/daily_reviewer.py
  └── web/routes.py
        └── web/templates/
```

---

## 3. 数据模型

### 3.1 ER关系图

```
┌─────────────┐       ┌──────────────┐       ┌──────────────────┐
│   WeiboPost  │1─────*│    Image     │       │  StockMention    │
│─────────────│       │──────────────│       │──────────────────│
│ id (PK)     │       │ id (PK)      │       │ id (PK)          │
│ weibo_id    │       │ weibo_id(FK) │       │ weibo_id (FK)    │
│ user_id     │       │ url          │       │ stock_code       │
│ user_name   │       │ local_path   │       │ stock_name       │
│ content     │       │ ocr_text     │       │ sentiment_score  │
│ raw_html    │       │ created_at   │       │ mention_type     │
│ post_time   │       └──────────────┘       │ confidence       │
│ reposts     │                              └──────────────────┘
│ comments    │
│ likes       │       ┌──────────────┐       ┌──────────────────┐
│ is_original │       │    Alert     │       │  Trade           │
│ source      │       │──────────────│       │──────────────────│
│ crawl_time  │       │ id (PK)      │       │ id (PK)          │
│ sentiment   │       │ weibo_id(FK) │       │ stock_code       │
│ keywords    │       │ alert_type   │       │ action           │
│ summary     │       │ level        │       │ price            │
│ is_analyzed │       │ title        │       │ quantity         │
│ is_alerted  │       │ content      │       │ amount           │
└─────────────┘       │ is_sent      │       │ trade_time       │
                      │ created_at   │       │ profit_loss      │
                      └──────────────┘       │ strategy         │
                                             └──────────────────┘
                      ┌──────────────┐
                      │  Position    │       ┌──────────────────┐
                      │──────────────│       │  DailyReview     │
                      │ id (PK)      │       │──────────────────│
                      │ stock_code   │       │ id (PK)          │
                      │ quantity     │       │ review_date      │
                      │ avg_cost     │       │ hot_stocks       │
                      │ current_price│       │ sentiment_summary│
                      │ profit_loss  │       │ top_posts        │
                      │ opened_at    │       │ market_view      │
                      │ strategy     │       │ created_at       │
                      └──────────────┘       └──────────────────┘
```

### 3.2 核心表结构

#### weibo_posts — 微博帖子主表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| weibo_id | TEXT UNIQUE | 微博原始ID，用于去重 |
| user_id | TEXT | 发帖用户ID |
| user_name | TEXT | 发帖用户昵称 |
| content | TEXT | 清洗后的纯文本内容 |
| raw_html | TEXT | 原始HTML（保留完整信息） |
| post_time | DATETIME | 发帖时间 |
| reposts | INTEGER | 转发数 |
| comments | INTEGER | 评论数 |
| likes | INTEGER | 点赞数 |
| is_original | BOOLEAN | 是否原创 |
| source | TEXT | 发帖来源（客户端） |
| crawl_time | DATETIME | 抓取时间 |
| sentiment | REAL | 情感分数 [-1, 1] |
| keywords | TEXT(JSON) | 提取的关键词列表 |
| summary | TEXT | LLM生成的摘要 |
| is_analyzed | BOOLEAN | 是否已完成语义分析 |
| is_alerted | BOOLEAN | 是否已触发告警 |

#### images — 图片表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| weibo_id | TEXT FK | 关联微博ID |
| url | TEXT | 原始图片URL |
| local_path | TEXT | 本地存储路径 |
| ocr_text | TEXT | OCR识别文本 |
| created_at | DATETIME | 记录创建时间 |

#### stock_mentions — 股票提及表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| weibo_id | TEXT FK | 关联微博ID |
| stock_code | TEXT | 股票代码（如 SH600519） |
| stock_name | TEXT | 股票名称（如 贵州茅台） |
| sentiment_score | REAL | 该提及的情感分数 |
| mention_type | TEXT | 提及类型（直接/间接/隐含） |
| confidence | REAL | 识别置信度 |

#### alerts — 告警表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| weibo_id | TEXT FK | 关联微博ID |
| alert_type | TEXT | 告警类型 |
| level | TEXT | 告警级别 (info/warning/critical) |
| title | TEXT | 告警标题 |
| content | TEXT | 告警内容 |
| is_sent | BOOLEAN | 是否已推送 |
| created_at | DATETIME | 创建时间 |

#### trades — 交易记录表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| stock_code | TEXT | 股票代码 |
| action | TEXT | 买入/卖出 |
| price | REAL | 成交价格 |
| quantity | INTEGER | 成交数量 |
| amount | REAL | 成交金额 |
| trade_time | DATETIME | 交易时间 |
| profit_loss | REAL | 盈亏（卖出时计算） |
| strategy | TEXT | 交易策略标识 |

#### positions — 持仓表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| stock_code | TEXT | 股票代码 |
| quantity | INTEGER | 持仓数量 |
| avg_cost | REAL | 平均成本价 |
| current_price | REAL | 当前价格 |
| profit_loss | REAL | 浮动盈亏 |
| opened_at | DATETIME | 开仓时间 |
| strategy | TEXT | 策略标识 |

#### daily_reviews — 每日复盘表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| review_date | DATE | 复盘日期 |
| hot_stocks | TEXT(JSON) | 热门股票列表 |
| sentiment_summary | TEXT | 情绪总结 |
| top_posts | TEXT(JSON) | 热门帖子列表 |
| market_view | TEXT | 市场观点 |
| created_at | DATETIME | 创建时间 |

### 3.3 索引设计

```sql
-- 高频查询索引
CREATE INDEX idx_weibo_user ON weibo_posts(user_id);
CREATE INDEX idx_weibo_time ON weibo_posts(post_time);
CREATE INDEX idx_weibo_sentiment ON weibo_posts(sentiment);
CREATE INDEX idx_weibo_analyzed ON weibo_posts(is_analyzed);
CREATE INDEX idx_stock_code ON stock_mentions(stock_code);
CREATE INDEX idx_alert_level ON alerts(level, is_sent);
CREATE INDEX idx_trade_stock ON trades(stock_code, trade_time);
CREATE INDEX idx_position_stock ON positions(stock_code);
```

---

## 4. 爬虫引擎

### 4.1 模块位置

`app/crawler/weibo.py`

### 4.2 功能概述

爬虫引擎负责从微博平台抓取指定用户的发帖数据，是整个系统的数据入口。

### 4.3 核心流程

```
启动爬虫
  │
  ├── 加载监控用户列表 (config.yaml → monitor_users)
  │
  ├── 初始化会话 (Cookie + 代理)
  │
  └── 对每个用户循环:
        │
        ├── 构造请求 (微博移动端API)
        │     └── URL: https://m.weibo.cn/api/container/getIndex
        │
        ├── 发起请求 (httpx.AsyncClient)
        │     ├── 设置 Headers (User-Agent, Referer, X-Requested-With)
        │     ├── 绑定代理 (从代理池获取)
        │     └── 绑定 Cookie (从Cookie池轮换)
        │
        ├── 解析响应 (JSON)
        │     ├── 提取卡片列表 (data.cards)
        │     ├── 过滤广告/推广 (card_type 判断)
        │     └── 提取帖子字段
        │
        ├── 去重检查 (weibo_id → 数据库查重)
        │     ├── 已存在 → 跳过
        │     └── 不存在 → 写入数据库
        │
        ├── 翻页处理
        │     ├── 提取 pagelet 下一页参数
        │     ├── 延迟翻页 (随机 2-5 秒)
        │     └── 达到上限或无新数据 → 停止
        │
        └── 错误处理
              ├── HTTP 429 → 切换代理+Cookie，指数退避
              ├── HTTP 403 → Cookie失效，切换账号
              ├── 网络超时 → 重试3次
              └── 解析异常 → 记录日志，跳过该条
```

### 4.4 请求参数

```python
# 微博移动端API请求参数
params = {
    "type": "uid",
    "value": user_id,          # 监控用户的UID
    "containerid": f"107603{user_id}",  # 用户帖子容器ID
    "page": page_number,       # 页码
}
```

### 4.5 数据提取字段

| 字段 | 来源路径 | 说明 |
|------|----------|------|
| weibo_id | mblog.id | 帖子唯一标识 |
| user_id | mblog.user.id | 用户ID |
| user_name | mblog.user.screen_name | 用户昵称 |
| content | mblog.text | 帖子内容(含HTML) |
| post_time | mblog.created_at | 发帖时间 |
| reposts | mblog.reposts_count | 转发数 |
| comments | mblog.comments_count | 评论数 |
| likes | mblog.attitudes_count | 点赞数 |
| is_original | mblog.retweeted_status | 是否原创 |
| source | mblog.source | 发帖来源 |
| images | mblog.pics | 图片列表 |

### 4.6 Cookie管理

```python
class CookieManager:
    """Cookie池管理，支持轮换和自动检测失效"""

    def __init__(self, cookies: list[str]):
        self.cookies = cookies      # Cookie字符串列表
        self.index = 0              # 当前使用索引
        self.fail_count = {}        # 失败计数

    def get(self) -> str:
        """获取当前Cookie"""

    def rotate(self):
        """轮换到下一个Cookie"""

    def mark_failed(self, cookie: str):
        """标记Cookie失效，连续失败3次则移除"""
```

### 4.7 速率控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| request_delay | 2-5秒 | 请求间隔（随机） |
| page_delay | 3-8秒 | 翻页间隔 |
| max_pages | 50 | 单次最大翻页数 |
| retry_times | 3 | 失败重试次数 |
| retry_delay | 5-15秒 | 重试延迟（指数退避） |
| concurrent_users | 3 | 并发抓取用户数 |

---

## 5. 内容解析与清洗

### 5.1 模块位置

`app/crawler/parser.py`

### 5.2 功能概述

将微博原始HTML内容解析为结构化数据，完成文本清洗、实体提取和去重。

### 5.3 处理流水线

```
原始HTML (mblog.text)
  │
  ├── HTML标签清理
  │     ├── 移除 <a> 标签（@用户链接）
  │     ├── 移除 <em> 标签（话题标签）
  │     ├── 移除其他HTML标签
  │     └── 保留纯文本内容
  │
  ├── URL处理
  │     ├── 识别微博短链接 (t.cn/xxx)
  │     ├── 识别完整URL
  │     └── 替换为 [链接] 占位符或保留
  │
  ├── 表情处理
  │     ├── 识别微博表情标签 <img alt="[xxx]">
  │     └── 替换为文字描述 [xxx]
  │
  ├── 话题提取
  │     ├── 识别 #话题# 格式
  │     └── 提取为独立字段
  │
  ├── @用户提取
  │     ├── 识别 @用户名 格式
  │     └── 提取为独立字段
  │
  ├── 股票实体识别
  │     ├── 正则匹配股票代码 (6位数字 + 市场)
  │     ├── 关键词匹配股票名称 (股票名称词典)
  │     └── 识别提及类型 (直接/间接/隐含)
  │
  └── 文本规范化
        ├── 全角→半角转换
        ├── 多余空白压缩
        └── 特殊字符清理
```

### 5.4 股票实体识别规则

```python
# 股票代码识别
STOCK_CODE_PATTERN = r'(SH|SZ|sh|sz)?(\d{6})'
# 有效范围: SH 600xxx-605xxx, SZ 000xxx-005xxx, SZ 300xxx

# 股票名称匹配
# 从 stock_name_dict 加载，支持:
#   - 全称: "贵州茅台"
#   - 简称: "茅台"
#   - 别名: "茅王" (网络用语映射)

# 提及类型判定
MENTION_TYPE = {
    "direct":   "明确提及股票代码或名称",
    "indirect": "提及行业/板块关联到股票",
    "implicit": "隐含提及（如'白酒龙头'→茅台）",
}
```

### 5.5 去重策略

| 维度 | 方法 | 说明 |
|------|------|------|
| 精确去重 | weibo_id唯一索引 | 同一条微博不重复入库 |
| 近似去重 | SimHash + 海明距离 | 相似度>90%视为重复 |
| 转发去重 | 原文weibo_id关联 | 转发帖关联到原帖 |

---

## 6. 图片处理与OCR

### 6.1 模块位置

- 图片下载: `app/crawler/image.py`
- OCR识别: `app/analyzer/image_ocr.py`

### 6.2 图片下载流程

```
微博图片列表 (mblog.pics)
  │
  ├── 提取图片URL
  │     ├── 原图: url.replace("thumb150", "large")
  │     └── 中图: url.replace("thumb150", "mw2000")
  │
  ├── 异步批量下载 (httpx.AsyncClient)
  │     ├── 并发控制: semaphore(5)
  │     ├── 超时: 30秒
  │     └── 重试: 2次
  │
  ├── 本地存储
  │     ├── 路径: data/images/{date}/{weibo_id}_{index}.jpg
  │     └── 关联写入 images 表
  │
  └── 错误处理
        ├── 下载失败 → 记录URL，不阻塞
        └── 格式异常 → 跳过
```

### 6.3 OCR识别流程

```
本地图片文件
  │
  ├── PaddleOCR 初始化
  │     ├── 语言: ch (中文)
  │     ├── 使用GPU: 按配置
  │     └── 检测+识别模型
  │
  ├── 文字区域检测
  │     └── 输出: 文字区域坐标 + 置信度
  │
  ├── 文字识别
  │     └── 输出: 识别文本 + 置信度
  │
  ├── 后处理
  │     ├── 低置信度过滤 (< 0.5)
  │     ├── 文本拼接 (按区域位置排序)
  │     └── 股票信息提取 (从OCR文本中)
  │
  └── 结果存储
        └── 写入 images.ocr_text 字段
```

### 6.4 OCR应用场景

| 场景 | 示例 | 处理方式 |
|------|------|----------|
| 股票截图 | 股票行情截图 | 提取股票代码和价格 |
| 文字长图 | 长文转图片 | OCR提取全文 |
| 数据表格 | 财报/数据表 | OCR+结构化提取 |
| 手写笔记 | 分析师手写 | OCR+LLM理解 |

---

## 7. 语义分析引擎

### 7.1 模块位置

- 基础语义: `app/analyzer/llm_semantic.py`
- 增强语义: `app/analyzer/enhanced_semantic.py`

### 7.2 基础语义分析 (LLM Semantic)

#### 分析流程

```
微博文本
  │
  ├── Prompt构造
  │     ├── 系统提示词 (角色定义)
  │     ├── 分析指令 (情感/关键词/摘要)
  │     └── 输入文本
  │
  ├── LLM调用
  │     ├── 模型: 配置的LLM (DeepSeek/Qwen等)
  │     ├── 温度: 0.3 (偏确定性)
  │     └── 最大Token: 1024
  │
  ├── 结果解析
  │     ├── 情感分数: [-1, 1] (看空→中性→看多)
  │     ├── 关键词列表: 5-10个
  │     ├── 内容摘要: 1-2句
  │     └── 市场影响: 高/中/低
  │
  └── 结果存储
        ├── weibo_posts.sentiment
        ├── weibo_posts.keywords
        ├── weibo_posts.summary
        └── weibo_posts.is_analyzed = True
```

#### Prompt模板

```
你是一位专业的金融分析师，请分析以下微博内容的情感倾向和市场影响。

分析要求:
1. 情感分数: -1(极度看空) 到 1(极度看多)，保留两位小数
2. 关键词: 提取5-10个关键词
3. 摘要: 用1-2句话概括核心观点
4. 市场影响: 评估对市场的潜在影响程度(高/中/低)

请以JSON格式输出:
{
  "sentiment": 0.75,
  "keywords": ["白酒", "涨价", "茅台"],
  "summary": "博主看好白酒板块，认为茅台有涨价预期",
  "market_impact": "中"
}

微博内容:
{content}
```

### 7.3 增强语义分析 (Enhanced Semantic)

在基础语义之上增加更深入的分析维度:

| 分析维度 | 说明 | 输出 |
|----------|------|------|
| 情感细分 | 区分对大盘/行业/个股的情感 | per_entity_sentiment |
| 观点强度 | 语气词、感叹号、表情等辅助判断 | intensity: 0-1 |
| 时效性判断 | 内容是即时观点还是回顾总结 | timeliness |
| 可信度评估 | 博主历史准确率、内容客观性 | credibility |
| 传播预测 | 预测该帖的传播影响力 | viral_potential |

### 7.4 LLM调用策略

| 策略 | 说明 |
|------|------|
| 批量处理 | 积累N条未分析帖子后批量调用 |
| 优先级 | 高粉丝用户 > 普通用户 |
| 降级 | LLM不可用时回退到规则引擎 |
| 缓存 | 相似内容复用分析结果 |
| 限流 | 控制API调用频率，避免超额 |

---

## 8. 股票关联分析

### 8.1 模块位置

`app/analyzer/stock_analyzer.py`

### 8.2 功能概述

将微博内容与A股市场建立关联，识别提及的股票并评估情绪影响。

### 8.3 分析流程

```
已分析的微博帖子
  │
  ├── 股票实体提取
  │     ├── 正则匹配 (股票代码)
  │     ├── 词典匹配 (股票名称/别名)
  │     ├── LLM辅助 (隐含提及)
  │     └── OCR文本补充 (图片中的股票信息)
  │
  ├── 情绪-股票映射
  │     ├── 帖子情感 → 各提及股票的情感
  │     ├── 上下文关联 (同一句中的情感指向)
  │     └── 否定处理 ("不看好茅台" → 茅台情感为负)
  │
  ├── 热度计算
  │     ├── 提及频次 (时间窗口内)
  │     ├── 互动加权 (转发+评论+点赞)
  │     ├── 博主权重 (粉丝数/影响力)
  │     └── 时效衰减 (指数衰减函数)
  │
  └── 结果输出
        ├── stock_mentions 表写入
        └── 股票热度排行更新
```

### 8.4 热度计算公式

```
heat_score = Σ(mention_count × interaction_weight × author_weight × time_decay)

其中:
  interaction_weight = log10(reposts + comments + likes + 1)
  author_weight = 基于博主影响力分级 (1-5)
  time_decay = e^(-λ × hours_since_post)   # λ=0.1, 半衰期约7小时
```

### 8.5 情绪聚合

```
# 单只股票的聚合情绪
stock_sentiment = Σ(individual_sentiment × weight) / Σ(weight)

weight = interaction_weight × author_weight × confidence

# 时间窗口: 默认24小时
# 输出: 综合情绪分数 + 情绪趋势(上升/下降/平稳)
```

---

## 9. 选股与预测模型

### 9.1 模块位置

- 选股器: `app/selector/stock_selector.py`
- 预测器: `app/selector/predictor.py`
- 多因子模型: `app/selector/multi_factor_model.py`
- 回测: `app/selector/backtest.py`

### 9.2 选股流程

```
股票情绪数据
  │
  ├── 因子计算
  │     ├── 情绪因子 (sentiment_factor)
  │     ├── 热度因子 (heat_factor)
  │     ├── 动量因子 (momentum_factor)
  │     ├── 波动因子 (volatility_factor)
  │     └── 资金因子 (capital_flow_factor)
  │
  ├── 多因子合成
  │     ├── 因子标准化 (Z-Score)
  │     ├── 因子加权 (可配置权重)
  │     └── 综合评分
  │
  ├── 股票筛选
  │     ├── 评分排名 (Top N)
  │     ├── 行业分散 (同行业最多2只)
  │     ├── 流动性过滤 (日均成交额>阈值)
  │     └── ST/停牌过滤
  │
  └── 输出推荐列表
        ├── 推荐股票代码+名称
        ├── 综合评分
        ├── 各因子得分
        └── 推荐理由
```

### 9.3 多因子模型

```python
class MultiFactorModel:
    """多因子选股模型"""

    factors = {
        "sentiment": {           # 情绪因子
            "weight": 0.30,
            "source": "微博情绪聚合",
            "calculation": "加权平均情绪分数",
        },
        "heat": {                # 热度因子
            "weight": 0.20,
            "source": "提及频次+互动量",
            "calculation": "标准化热度分数",
        },
        "momentum": {            # 动量因子
            "weight": 0.20,
            "source": "情绪变化趋势",
            "calculation": "情绪5日变化率",
        },
        "volatility": {          # 波动因子
            "weight": 0.15,
            "source": "情绪分歧度",
            "calculation": "情绪标准差(反向)",
        },
        "capital_flow": {        # 资金因子
            "weight": 0.15,
            "source": "大V一致性",
            "calculation": "高权重博主情绪一致性",
        },
    }
```

### 9.4 预测器

```
预测器 (Predictor)
  │
  ├── 输入特征
  │     ├── 情绪时间序列 (7日)
  │     ├── 热度时间序列 (7日)
  │     ├── 互动量时间序列
  │     └── 技术指标 (如有)
  │
  ├── 模型
  │     ├── 简单模型: 线性回归
  │     ├── 进阶模型: XGBoost
  │     └── LLM预测: 基于文本的走势判断
  │
  └── 输出
        ├── 涨跌方向预测
        ├── 预测置信度
        └── 建议操作 (买入/持有/卖出)
```

### 9.5 回测框架

```
回测引擎 (Backtest)
  │
  ├── 输入
  │     ├── 历史情绪数据
  │     ├── 历史价格数据
  │     └── 策略参数
  │
  ├── 模拟执行
  │     ├── 按时间顺序遍历
  │     ├── 信号触发 → 模拟交易
  │     └── 记录每笔交易
  │
  ├── 统计指标
  │     ├── 总收益率
  │     ├── 年化收益率
  │     ├── 最大回撤
  │     ├── 夏普比率
  │     ├── 胜率
  │     └── 盈亏比
  │
  └── 报告输出
        ├── 收益曲线
        ├── 回撤曲线
        └── 交易明细
```

---

## 10. 模拟交易系统

### 10.1 模块位置

- 交易引擎: `app/trading/simulated_trading.py`
- 仓位管理: `app/trading/position_manager.py`

### 10.2 交易流程

```
选股信号 / 告警触发
  │
  ├── 信号评估
  │     ├── 信号强度 (强/中/弱)
  │     ├── 情绪一致性
  │     └── 风险评估
  │
  ├── 交易决策
  │     ├── 买入信号 → 计算买入量
  │     ├── 卖出信号 → 计算卖出量
  │     └── 无信号 → 持仓不动
  │
  ├── 仓位管理
  │     ├── 单只股票仓位上限: 20%
  │     ├── 总仓位上限: 80%
  │     ├── 止损线: -7%
  │     └── 止盈线: +15%
  │
  ├── 模拟成交
  │     ├── 以当日收盘价成交
  │     ├── 计算手续费 (万三)
  │     ├── 计算印花税 (卖出千一)
  │     └── 记录交易
  │
  └── 持仓更新
        ├── 更新 positions 表
        ├── 更新 trades 表
        └── 计算浮动盈亏
```

### 10.3 仓位管理器

```python
class PositionManager:
    """仓位管理器"""

    # 风控参数
    MAX_POSITION_RATIO = 0.20    # 单只股票最大仓位
    MAX_TOTAL_RATIO = 0.80       # 总仓位上限
    STOP_LOSS = -0.07            # 止损线 -7%
    TAKE_PROFIT = 0.15           # 止盈线 +15%
    INITIAL_CAPITAL = 1000000    # 初始资金 100万

    def can_buy(self, stock_code, amount) -> bool:
        """检查是否可以买入"""

    def can_sell(self, stock_code, quantity) -> bool:
        """检查是否可以卖出"""

    def check_stop_loss(self, stock_code, current_price) -> bool:
        """止损检查"""

    def check_take_profit(self, stock_code, current_price) -> bool:
        """止盈检查"""

    def get_portfolio_summary(self) -> dict:
        """获取持仓汇总"""
```

### 10.4 交易记录

每笔交易记录包含:
- 交易时间、股票代码、买卖方向
- 成交价格、数量、金额
- 手续费、印花税
- 盈亏（卖出时）
- 触发策略标识

---

## 11. 实时告警系统

### 11.1 模块位置

- 实时告警: `app/notifier/realtime_alert.py`
- 语义告警检测: `app/notifier/semantic_alert_detector.py`

### 11.2 告警类型

| 告警类型 | 触发条件 | 级别 |
|----------|----------|------|
| 大V发帖 | 监控列表中的大V发布新帖 | info |
| 情绪突变 | 股票情绪分数24h变化>0.5 | warning |
| 热度飙升 | 股票提及量1h内增长>200% | warning |
| 负面爆发 | 多个博主同时看空同一股票 | critical |
| 重大事件 | 检测到重大利空/利好关键词 | critical |
| 止损触发 | 模拟持仓触及止损线 | warning |
| 选股信号 | 选股模型产生买入/卖出信号 | info |

### 11.3 语义告警检测

```
新微博帖子
  │
  ├── 关键词匹配 (快速通道)
  │     ├── 利空词库: "暴雷", "退市", "违规", "处罚"...
  │     ├── 利好词库: "涨停", "重组", "回购", "增持"...
  │     └── 匹配到 → 直接触发告警
  │
  ├── LLM语义判断 (深度通道)
  │     ├── 是否包含重大事件
  │     ├── 影响范围评估
  │     └── 紧急程度判断
  │
  ├── 告警去重
  │     ├── 同一事件30分钟内不重复告警
  │     ├── 同一股票1小时内聚合告警
  │     └── 告警指纹 (stock_code + alert_type + content_hash)
  │
  └── 告警输出
        ├── 写入 alerts 表
        └── 触发通知推送
```

### 11.4 告警级别

```
┌─────────┬──────────────────────────────────────────────┐
│  Level  │  行为                                         │
├─────────┼──────────────────────────────────────────────┤
│  info   │  记录到数据库，Dashboard展示，不推送通知        │
│ warning │  记录到数据库，微信/邮件推送                    │
│critical │  记录到数据库，全渠道推送(微信+邮件+Webhook)    │
└─────────┴──────────────────────────────────────────────┘
```

---

## 12. 通知推送系统

### 12.1 模块位置

- 通知调度: `app/notifier/notifier.py`
- 微信推送: `app/notifier/wechat.py`
- 邮件推送: `app/notifier/email_notify.py`
- Webhook: `app/notifier/webhook.py`

### 12.2 通知调度流程

```
告警事件
  │
  ├── 查询告警级别
  │
  ├── 选择通知渠道
  │     ├── info → 仅Dashboard
  │     ├── warning → 微信 + 邮件
  │     └── critical → 微信 + 邮件 + Webhook
  │
  ├── 并行推送
  │     ├── 微信推送 (异步)
  │     ├── 邮件推送 (异步)
  │     └── Webhook推送 (异步)
  │
  ├── 推送结果记录
  │     ├── 成功 → alerts.is_sent = True
  │     └── 失败 → 记录失败原因，重试1次
  │
  └── 推送限流
        ├── 同一用户 1分钟内最多5条
        └── 同一股票 10分钟内最多3条
```

### 12.3 微信推送

支持两种方式:

| 方式 | 实现 | 适用场景 |
|------|------|----------|
| 企业微信 | 企业微信机器人Webhook | 团队使用 |
| Server酱 | sct.ftqq.com API | 个人使用 |

```python
class WechatNotifier:
    """微信通知器"""

    async def send_enterprise_wechat(self, title, content):
        """企业微信机器人推送"""
        # Webhook URL: https://qyapi.weixin.qq.com/cgi-bin/webhook/send
        # 支持: text, markdown, image, news 类型

    async def send_serverchan(self, title, content):
        """Server酱推送"""
        # API: https://sct.ftqq.com/{key}.send
```

### 12.4 邮件推送

```python
class EmailNotifier:
    """邮件通知器"""

    # SMTP配置
    smtp_host: str       # SMTP服务器
    smtp_port: int       # 端口 (465/587)
    smtp_user: str       # 发件人
    smtp_pass: str       # 授权码
    recipients: list     # 收件人列表

    async def send(self, subject, body, html=False):
        """发送邮件通知"""
```

### 12.5 Webhook推送

```python
class WebhookNotifier:
    """Webhook通知器"""

    async def send(self, payload: dict):
        """发送Webhook通知"""
        # POST JSON到配置的Webhook URL
        # 支持自定义Header
        # 支持飞书/钉钉/自定义格式
```

---

## 13. 定时任务调度

### 13.1 模块位置

`app/scheduler.py`

### 13.2 调度任务列表

| 任务 | Cron表达式 | 间隔 | 说明 |
|------|-----------|------|------|
| 微博抓取 | - | 每30分钟 | 抓取监控用户的新帖 |
| 语义分析 | - | 每15分钟 | 分析未处理的帖子 |
| 股票分析 | - | 每30分钟 | 更新股票情绪和热度 |
| 选股计算 | 0 9 * * 1-5 | 每交易日9:00 | 运行选股模型 |
| 模拟交易 | 0 15 * * 1-5 | 每交易日15:00 | 执行模拟交易 |
| 每日复盘 | 30 16 * * 1-5 | 每交易日16:30 | 生成每日复盘报告 |
| 告警检查 | - | 每5分钟 | 检查告警条件 |
| 健康检查 | - | 每10分钟 | 检查系统健康状态 |
| 数据清理 | 0 2 * * * | 每天凌晨2:00 | 清理过期数据 |

### 13.3 调度器实现

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class TaskScheduler:
    """定时任务调度器"""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._register_tasks()

    def _register_tasks(self):
        """注册所有定时任务"""
        # 间隔任务
        self.scheduler.add_job(crawl_weibo, 'interval', minutes=30)
        self.scheduler.add_job(analyze_semantic, 'interval', minutes=15)
        self.scheduler.add_job(analyze_stocks, 'interval', minutes=30)
        self.scheduler.add_job(check_alerts, 'interval', minutes=5)
        self.scheduler.add_job(health_check, 'interval', minutes=10)

        # Cron任务
        self.scheduler.add_job(run_stock_selector, 'cron',
                             hour=9, minute=0, day_of_week='mon-fri')
        self.scheduler.add_job(execute_trading, 'cron',
                             hour=15, minute=0, day_of_week='mon-fri')
        self.scheduler.add_job(daily_review, 'cron',
                             hour=16, minute=30, day_of_week='mon-fri')
        self.scheduler.add_job(cleanup_data, 'cron',
                             hour=2, minute=0)

    async def start(self):
        """启动调度器"""

    async def shutdown(self):
        """优雅关闭"""
```

### 13.4 任务执行保障

| 机制 | 说明 |
|------|------|
| 任务锁 | 同一任务不会并发执行 (APScheduler misfire_grace_time) |
| 超时控制 | 每个任务设置最大执行时间 |
| 失败重试 | 任务失败自动重试1次 |
| 执行日志 | 记录每次任务的开始/结束/耗时/结果 |
| 手动触发 | Web Dashboard支持手动触发任务 |

---

## 14. Web管理后台

### 14.1 模块位置

- 路由: `app/web/routes.py`
- 模板: `app/web/templates/`

### 14.2 页面列表

| 页面 | 路由 | 功能 |
|------|------|------|
| 首页 | `/` | 系统概览，关键指标 |
| Dashboard | `/dashboard` | 详细数据面板 |
| 微博列表 | `/weibos` | 浏览抓取的微博 |
| 股票情绪 | `/stocks` | 股票情绪排行 |
| 告警列表 | `/alerts` | 查看告警历史 |
| 交易记录 | `/trades` | 模拟交易明细 |
| 持仓管理 | `/positions` | 当前持仓 |
| 任务管理 | `/tasks` | 定时任务状态 |

### 14.3 API接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/weibos` | 获取微博列表（分页、筛选） |
| GET | `/api/weibos/{id}` | 获取微博详情 |
| GET | `/api/stocks` | 获取股票情绪数据 |
| GET | `/api/stocks/{code}` | 获取单只股票详情 |
| GET | `/api/alerts` | 获取告警列表 |
| GET | `/api/trades` | 获取交易记录 |
| GET | `/api/positions` | 获取当前持仓 |
| GET | `/api/stats` | 获取系统统计 |
| POST | `/api/tasks/{name}/trigger` | 手动触发任务 |
| GET | `/api/health` | 健康检查 |

### 14.4 Dashboard功能

```
┌──────────────────────────────────────────────────────────┐
│  微博抓取系统 Dashboard                                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 今日抓取  │ │ 待分析   │ │ 今日告警  │ │ 持仓盈亏  │   │
│  │   1,234  │ │    56    │ │    12    │ │  +3.2%   │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  股票情绪排行                                       │  │
│  │  1. 贵州茅台  ████████████  0.85  ↑               │  │
│  │  2. 宁德时代  ██████████    0.72  ↑               │  │
│  │  3. 比亚迪    █████████     0.65  →               │  │
│  │  4. 中芯国际  ████████      0.58  ↓               │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  最新告警                                           │  │
│  │  🔴 大V"股社区"发布重大利空消息 - 茅台  5分钟前     │  │
│  │  🟡 白酒板块情绪突变 - 多空分歧加大      12分钟前   │  │
│  │  🔵 选股模型新增2只推荐标的               30分钟前   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  模拟交易收益曲线                                   │  │
│  │  📈 [收益曲线图]                                    │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 15. 代理与反爬策略

### 15.1 模块位置

`app/utils/proxy.py`

### 15.2 代理池管理

```
代理池 (ProxyPool)
  │
  ├── 代理来源
  │     ├── 配置文件 (静态代理列表)
  │     ├── 代理API (动态获取)
  │     └── 自建代理 (可选)
  │
  ├── 代理验证
  │     ├── 可用性检测 (定期ping)
  │     ├── 速度测试 (响应时间)
  │     └── 匿名度检测
  │
  ├── 代理调度
  │     ├── 随机选取
  │     ├── 加权选取 (按速度/成功率)
  │     └── 轮换策略 (每N次请求换代理)
  │
  └── 代理健康
        ├── 成功率统计
        ├── 失败自动剔除
        └── 定期补充
```

### 15.3 反爬策略

| 策略 | 实现 | 说明 |
|------|------|------|
| User-Agent轮换 | 随机UA池 | 每次请求随机选择 |
| 请求间隔 | 随机延迟2-5秒 | 模拟人类行为 |
| Cookie轮换 | Cookie池 | 多账号轮换 |
| 代理轮换 | 代理池 | 避免IP封禁 |
| 请求头模拟 | 完整Headers | 模拟浏览器行为 |
| 指数退避 | 429/403自动退避 | 避免触发限制 |
| 验证码处理 | 人工介入 | 检测到验证码暂停并通知 |

### 15.4 请求头配置

```python
HEADERS = {
    "User-Agent": "随机UA",
    "Referer": "https://m.weibo.cn/",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
```

---

## 16. 日志与监控

### 16.1 模块位置

`app/utils/logger.py`

### 16.2 日志体系

```
日志级别:
  DEBUG   → 开发调试信息
  INFO    → 正常运行信息 (抓取/分析/交易)
  WARNING → 异常但可恢复 (代理失败/重试)
  ERROR   → 严重错误 (数据库异常/LLM调用失败)

日志输出:
  ├── 控制台 (彩色格式化)
  ├── 文件 (按日期轮转)
  │     ├── logs/app.log         # 全量日志
  │     ├── logs/crawler.log     # 爬虫日志
  │     ├── logs/analyzer.log    # 分析日志
  │     └── logs/trading.log     # 交易日志
  └── 结构化日志 (JSON格式，供后续分析)
```

### 16.3 健康检查

```python
async def health_check():
    """系统健康检查"""
    checks = {
        "database": check_db_connection(),
        "crawler": check_crawler_status(),
        "llm_api": check_llm_availability(),
        "proxy_pool": check_proxy_health(),
        "disk_space": check_disk_usage(),
        "last_crawl": get_last_crawl_time(),
    }
    return HealthStatus(checks)
```

### 16.4 监控指标

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| 抓取成功率 | 成功请求数/总请求数 | < 80% |
| 分析延迟 | 帖子发布到分析完成的耗时 | > 30分钟 |
| 代理可用率 | 可用代理数/总代理数 | < 50% |
| LLM调用成功率 | 成功调用/总调用 | < 90% |
| 数据库大小 | SQLite文件大小 | > 1GB |
| 磁盘使用率 | 数据目录磁盘占用 | > 80% |

---

## 17. 部署架构

### 17.1 Docker部署

```yaml
# docker-compose.yml
services:
  weibo-crawler:
    build: .
    container_name: weibo-crawler
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data        # 数据持久化
      - ./logs:/app/logs        # 日志持久化
      - ./config.yaml:/app/config.yaml  # 配置文件
    env_file:
      - .env
    environment:
      - TZ=Asia/Shanghai
```

### 17.2 Dockerfile

```dockerfile
FROM python:3.11-slim

# 安装系统依赖 (PaddleOCR需要)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "-m", "app.main"]
```

### 17.3 目录结构

```
weibo_crawler/
├── app/
│   ├── __init__.py
│   ├── main.py                 # 入口
│   ├── config.py               # 配置加载
│   ├── database.py             # 数据库连接
│   ├── models.py               # 数据模型
│   ├── scheduler.py            # 任务调度
│   ├── crawler/
│   │   ├── weibo.py            # 微博爬虫
│   │   ├── parser.py           # 内容解析
│   │   └── image.py            # 图片下载
│   ├── analyzer/
│   │   ├── stock_analyzer.py   # 股票分析
│   │   ├── llm_semantic.py     # LLM语义分析
│   │   ├── enhanced_semantic.py# 增强语义
│   │   └── image_ocr.py        # 图片OCR
│   ├── notifier/
│   │   ├── notifier.py         # 通知调度
│   │   ├── wechat.py           # 微信推送
│   │   ├── email_notify.py     # 邮件推送
│   │   ├── webhook.py          # Webhook推送
│   │   ├── realtime_alert.py   # 实时告警
│   │   └── semantic_alert_detector.py  # 语义告警
│   ├── selector/
│   │   ├── stock_selector.py   # 选股器
│   │   ├── predictor.py        # 预测器
│   │   ├── multi_factor_model.py  # 多因子模型
│   │   └── backtest.py         # 回测
│   ├── trading/
│   │   ├── simulated_trading.py # 模拟交易
│   │   └── position_manager.py  # 仓位管理
│   ├── reviewer/
│   │   └── daily_reviewer.py   # 每日复盘
│   ├── web/
│   │   ├── routes.py           # Web路由
│   │   └── templates/          # 页面模板
│   └── utils/
│       ├── proxy.py            # 代理管理
│       ├── logger.py           # 日志工具
│       └── helpers.py          # 通用工具
├── migrations/
│   └── 001_init.sql            # 数据库初始化
├── data/                       # 数据目录
│   ├── weibo.db                # SQLite数据库
│   └── images/                 # 图片存储
├── logs/                       # 日志目录
├── config.yaml                 # 配置文件
├── .env.example                # 环境变量模板
├── requirements.txt            # Python依赖
├── Dockerfile                  # Docker构建
├── docker-compose.yml          # Docker编排
└── README.md                   # 项目说明
```

---

## 18. 配置体系

### 18.1 配置文件结构 (config.yaml)

```yaml
# 系统配置
app:
  name: "微博抓取系统"
  debug: false
  log_level: "INFO"

# 数据库配置
database:
  path: "data/weibo.db"

# 爬虫配置
crawler:
  monitor_users:              # 监控用户列表
    - uid: "1234567890"
      name: "大V用户1"
      weight: 5               # 影响力权重
    - uid: "0987654321"
      name: "财经博主2"
      weight: 3
  request_delay: [2, 5]       # 请求延迟范围(秒)
  max_pages: 50               # 最大翻页数
  concurrent_users: 3         # 并发用户数
  cookies:                    # Cookie池
    - "SUB=xxx; SUBP=xxx"
    - "SUB=yyy; SUBP=yyy"

# 代理配置
proxy:
  enabled: true
  pool:                      # 代理列表
    - "http://proxy1:8080"
    - "http://proxy2:8080"
  api_url: ""                # 代理API地址
  rotate_interval: 10        # 轮换间隔(请求次数)

# LLM配置
llm:
  api_base: "https://api.deepseek.com/v1"
  api_key: "${LLM_API_KEY}"  # 从环境变量读取
  model: "deepseek-chat"
  temperature: 0.3
  max_tokens: 1024

# OCR配置
ocr:
  enabled: true
  use_gpu: false
  language: "ch"

# 通知配置
notification:
  wechat:
    enabled: true
    type: "enterprise"       # enterprise | serverchan
    webhook_url: "${WECHAT_WEBHOOK}"
    serverchan_key: "${SERVERCHAN_KEY}"
  email:
    enabled: true
    smtp_host: "smtp.qq.com"
    smtp_port: 465
    smtp_user: "${EMAIL_USER}"
    smtp_pass: "${EMAIL_PASS}"
    recipients:
      - "user@example.com"
  webhook:
    enabled: false
    url: ""
    headers: {}

# 选股配置
selector:
  factors:
    sentiment: 0.30
    heat: 0.20
    momentum: 0.20
    volatility: 0.15
    capital_flow: 0.15
  top_n: 10                  # 推荐股票数量
  min_daily_volume: 50000000 # 最低日均成交额

# 交易配置
trading:
  enabled: true
  initial_capital: 1000000
  max_position_ratio: 0.20
  max_total_ratio: 0.80
  stop_loss: -0.07
  take_profit: 0.15
  commission_rate: 0.0003
  stamp_tax: 0.001

# 告警配置
alert:
  sentiment_change_threshold: 0.5
  heat_surge_threshold: 2.0
  negative_burst_count: 3
  dedup_window_minutes: 30

# 调度配置
scheduler:
  crawl_interval: 30         # 抓取间隔(分钟)
  analyze_interval: 15       # 分析间隔(分钟)
  alert_check_interval: 5    # 告警检查间隔(分钟)
```

### 18.2 环境变量 (.env)

```bash
# LLM API
LLM_API_KEY=sk-xxx

# 微信通知
WECHAT_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
SERVERCHAN_KEY=SCTxxx

# 邮件通知
EMAIL_USER=xxx@qq.com
EMAIL_PASS=xxx

# 代理API (可选)
PROXY_API_URL=https://proxy-api.example.com/get
PROXY_API_KEY=xxx
```

### 18.3 配置加载优先级

```
环境变量 > .env文件 > config.yaml > 代码默认值
```

---

## 19. 数据流全景

### 19.1 主数据流

```
[微博平台]
    │
    │ HTTP请求 (代理+Cookie)
    ▼
[爬虫引擎] ──── 抓取JSON响应
    │
    │ 解析+清洗
    ▼
[内容解析器] ──── 纯文本+元数据
    │
    ├──→ [图片下载器] ──→ [OCR引擎] ──→ OCR文本
    │                                        │
    │◄───────────────────────────────────────┘
    │
    │ 写入数据库
    ▼
[weibo_posts表] (is_analyzed=False)
    │
    │ 定时任务触发
    ▼
[语义分析引擎] ──→ LLM API
    │
    │ 情感+关键词+摘要
    ▼
[weibo_posts表] (is_analyzed=True)
    │
    │ 股票实体提取
    ▼
[股票关联分析] ──→ [stock_mentions表]
    │
    │ 热度+情绪聚合
    ▼
[股票情绪数据]
    │
    ├──→ [选股模型] ──→ [推荐列表] ──→ [模拟交易] ──→ [trades/positions表]
    │
    ├──→ [告警检测] ──→ [alerts表] ──→ [通知推送] ──→ [微信/邮件/Webhook]
    │
    └──→ [每日复盘] ──→ [daily_reviews表]
```

### 19.2 数据生命周期

| 数据 | 保留策略 | 清理规则 |
|------|----------|----------|
| 微博帖子 | 永久保留 | 超过1年的raw_html可清理 |
| 图片文件 | 90天 | 超期图片删除，OCR文本保留 |
| 告警记录 | 180天 | 超期告警归档 |
| 交易记录 | 永久保留 | - |
| 日志文件 | 30天 | 按日期轮转，超期删除 |

---

## 20. 扩展性与未来规划

### 20.1 短期优化

| 方向 | 说明 |
|------|------|
| 数据库迁移 | SQLite → PostgreSQL，支持更大并发 |
| 消息队列 | 引入Redis/RabbitMQ，解耦爬虫和分析 |
| 缓存层 | Redis缓存热点数据，减轻数据库压力 |
| API限流 | Web API增加限流和认证 |

### 20.2 中期扩展

| 方向 | 说明 |
|------|------|
| 多平台采集 | 扩展至雪球、东方财富股吧、知乎 |
| 实时行情 | 接入实时股票行情数据 |
| 高级模型 | 训练专属情感分析模型 |
| 分布式爬虫 | Scrapy集群，提升采集能力 |
| 移动端 | 小程序/App查看告警和持仓 |

### 20.3 长期愿景

| 方向 | 说明 |
|------|------|
| 实盘对接 | 对接券商API，从模拟走向实盘 |
| 知识图谱 | 构建股票-概念-事件知识图谱 |
| 多模态分析 | 视频内容理解、语音转文字 |
| 协作平台 | 多用户、多策略、跟单系统 |

---

## 附录A: 关键接口定义

### 微博移动端API

```
GET https://m.weibo.cn/api/container/getIndex

参数:
  type: "uid"
  value: {user_id}
  containerid: "107603{user_id}"
  page: {page_number}

响应:
{
  "ok": 1,
  "data": {
    "cards": [
      {
        "card_type": 9,
        "mblog": {
          "id": "5123456789012345",
          "text": "<html content>",
          "user": {
            "id": 1234567890,
            "screen_name": "用户名"
          },
          "created_at": "01-15",
          "reposts_count": 100,
          "comments_count": 50,
          "attitudes_count": 200,
          "pics": [{"url": "..."}]
        }
      }
    ]
  }
}
```

## 附录B: 错误码定义

| 错误码 | 说明 | 处理方式 |
|--------|------|----------|
| E001 | Cookie失效 | 切换Cookie |
| E002 | IP被封禁 | 切换代理 |
| E003 | 请求频率过高 | 指数退避 |
| E004 | 数据解析异常 | 跳过并记录 |
| E005 | LLM调用失败 | 重试/降级 |
| E006 | 数据库写入失败 | 重试3次 |
| E007 | OCR处理失败 | 跳过该图片 |
| E008 | 通知推送失败 | 重试1次 |
| E009 | 代理不可用 | 切换代理 |
| E010 | 磁盘空间不足 | 告警+清理 |

## 附录C: 性能基准

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 抓取速度 | 100条/分钟 | 单线程 |
| 分析速度 | 50条/分钟 | LLM批处理 |
| OCR速度 | 10张/分钟 | CPU模式 |
| 告警延迟 | < 5分钟 | 从发帖到推送 |
| API响应 | < 200ms | Web API |
| 内存占用 | < 512MB | 常驻内存 |
| 磁盘增长 | ~50MB/天 | 含图片 |

# 舆情数据与实盘行情数据对接 — 专项审计报告

> 审计日期: 2026-05-21
> 审计范围: 全部源码、配置文件、数据库Schema、依赖清单
> 审计结论: **问题确认，且严重程度超出预期**

---

## 一、审计结论

**系统完全缺失实盘行情数据源。** 不存在任何股票价格、成交量、资金流向的实时或历史数据接入。选股模型的5个因子中，3个因子（动量、波动、资金）的计算依赖行情数据但无数据来源，其计算逻辑要么是存根，要么基于微博舆情数据做了不合理的替代。模拟交易系统的成交价格、持仓盈亏计算均无真实数据支撑。

这不是"可能存在"的问题，而是**确定性的系统级缺陷**。

---

## 二、逐层审计证据

### 2.1 外部依赖审计

**requirements.txt — 无任何行情数据SDK**

```
# 期望存在但实际不存在的依赖:
tushare          # ✗ 未引入
akshare          # ✗ 未引入
baostock         # ✗ 未引入
yfinance         # ✗ 未引入
jqdatasdk        # ✗ 未引入
xtquant          # ✗ 未引入
```

全项目无任何行情数据Python包的import语句。

### 2.2 配置文件审计

**config.yaml — 无行情数据配置段**

```yaml
# 实际存在的配置段:
app:          ✓ 系统配置
database:     ✓ 数据库配置
crawler:      ✓ 爬虫配置
proxy:        ✓ 代理配置
llm:          ✓ LLM配置
ocr:          ✓ OCR配置
notification: ✓ 通知配置
selector:     ✓ 选股配置
trading:      ✓ 交易配置
alert:        ✓ 告警配置
scheduler:    ✓ 调度配置

# 完全缺失的配置段:
market_data:  ✗ 无行情数据源配置
stock_api:    ✗ 无股票API配置
price_feed:   ✗ 无价格数据源配置
```

**.env.example — 无行情API密钥**

```bash
# 实际存在的环境变量:
LLM_API_KEY=          ✓ LLM密钥
WECHAT_WEBHOOK=       ✓ 微信Webhook
SERVERCHAN_KEY=       ✓ Server酱密钥
EMAIL_USER=           ✓ 邮件账号
EMAIL_PASS=           ✓ 邮件密码
PROXY_API_URL=        ✓ 代理API
PROXY_API_KEY=        ✓ 代理密钥

# 完全缺失的环境变量:
TUSHARE_TOKEN=        ✗ 无Tushare令牌
AKSHARE无密钥但需配置  ✗ 无
```

### 2.3 数据库Schema审计

**migrations/001_init.sql — 无行情数据表**

| 表名 | 存在 | 用途 |
|------|:---:|------|
| weibo_posts | ✓ | 微博帖子 |
| images | ✓ | 图片记录 |
| stock_mentions | ✓ | 股票提及 |
| alerts | ✓ | 告警记录 |
| trades | ✓ | 交易记录 |
| positions | ✓ | 持仓记录 |
| daily_reviews | ✓ | 每日复盘 |
| **stock_prices** | ✗ | **不存在 — 无K线/行情表** |
| **stock_daily** | ✗ | **不存在 — 无日线数据表** |
| **capital_flows** | ✗ | **不存在 — 无资金流向表** |
| **stock_info** | ✗ | **不存在 — 无股票基本信息表** |

`trades` 表有 `price` 字段，`positions` 表有 `current_price` 和 `avg_cost` 字段，但这些字段**没有任何数据源来填充真实值**。

### 2.4 HTTP调用审计

对 `app/selector/` 和 `app/trading/` 目录下所有Python文件搜索外部HTTP调用:

```
# 搜索: httpx / aiohttp / requests / fetch / api / url / endpoint
# 结果: 无任何行情数据相关的HTTP请求代码
```

这两个模块不发起任何网络请求获取行情数据。

---

## 三、受影响模块逐个剖析

### 3.1 多因子模型 (`multi_factor_model.py`) — 5个因子中3个无数据源

| 因子 | 权重 | 所需行情数据 | 实际数据来源 | 状态 |
|------|:---:|-------------|-------------|:---:|
| sentiment (情绪) | 30% | 不需要行情数据 | 微博情绪聚合 | ✓ 有数据 |
| heat (热度) | 20% | 不需要行情数据 | 微博提及频次+互动量 | ✓ 有数据 |
| **momentum (动量)** | **20%** | **情绪变化趋势需价格验证** | **仅用情绪5日变化率替代** | **⚠ 替代** |
| **volatility (波动)** | **15%** | **股价历史波动率** | **用情绪标准差替代** | **⚠ 替代** |
| **capital_flow (资金)** | **15%** | **主力资金净流入/北向资金** | **用大V情绪一致性替代** | **⚠ 替代** |

**核心问题: 后3个因子用舆情数据替代了行情数据，这不是"近似"，而是概念偷换。**

详细分析:

**momentum因子 — 用"情绪5日变化率"替代"价格动量"**

- 金融领域的动量因子定义: 过去N日股价收益率，反映价格趋势
- 系统的实现: 过去5日情绪分数的变化率
- 问题: 情绪变化率 ≠ 价格动量。情绪可能持续看多但股价横盘（利好出尽），也可能情绪中性但股价因资金推动上涨。两者相关性未经检验，直接替代无依据。

**volatility因子 — 用"情绪标准差"替代"股价波动率"**

- 金融领域的波动因子定义: 过去N日股价收益率的标准差
- 系统的实现: 情绪分数的标准差（反向，即分歧度）
- 问题: 情绪分歧度反映的是市场观点的分散程度，而非价格的实际波动。一只股票可能情绪高度一致但价格剧烈波动（如突发消息导致一致看涨但价格跳空），也可能情绪分歧但价格平稳。

**capital_flow因子 — 用"大V情绪一致性"替代"资金流向"**

- 金融领域的资金因子定义: 主力资金净流入额、北向资金净买入额
- 系统的实现: 高权重博主的情绪一致性
- 问题: 这完全是两个不同的概念。大V看多不代表有资金买入，大V看空也不代表有资金卖出。微博大V的影响力主要在散户端，而A股的定价权在机构资金。

### 3.2 模拟交易系统 (`simulated_trading.py`) — 成交价格无来源

`execute_buy()` 和 `execute_sell()` 函数需要成交价格来计算交易金额和盈亏，但:

- 无行情API调用获取实时/收盘价格
- 无数据库查询获取历史价格
- `price` 参数从调用方传入，但调用方本身也没有价格数据
- 交易记录中的 `price` 字段**没有可信的数据填充路径**

### 3.3 仓位管理器 (`position_manager.py`) — 止损止盈无法执行

```python
def check_stop_loss(self, stock_code, current_price) -> bool:
    """止损检查 — 需要current_price参数"""

def check_take_profit(self, stock_code, current_price) -> bool:
    """止盈检查 — 需要current_price参数"""
```

这两个函数需要 `current_price` 参数，但:

- 系统中没有任何地方能获取 `current_price` 的真实值
- `positions` 表的 `current_price` 字段无更新机制
- 无定时任务刷新持仓的当前价格
- 止损止盈检查形同虚设 — 没有价格数据，检查永远无法触发

### 3.4 选股器 (`stock_selector.py`) — 流动性过滤无数据

```python
def filter_by_liquidity(self, candidates) -> list:
    """流动性过滤 — 需要日均成交额数据"""
```

配置中设定 `min_daily_volume: 50000000`（最低日均成交额5000万），但:

- 无成交额数据来源
- 该过滤条件无法实际执行
- 可能退化为不过滤，导致推荐出流动性极差的股票

### 3.5 预测器 (`predictor.py`) — 预测无基准

预测器声称输出"涨跌方向预测"，但:

- 无历史价格数据作为训练标签
- 无实际收益率数据验证预测准确率
- 预测的"涨跌"无法与任何真实走势对照
- `predict()` 函数实现度低，核心逻辑是存根

### 3.6 回测框架 (`backtest.py`) — 无法运行

回测需要:
- 历史价格数据（开高低收）→ ✗ 无
- 基准收益率（沪深300等）→ ✗ 无
- 交易滑点数据 → ✗ 无

**没有价格数据，回测根本无法执行。** 即使代码框架完整，也是空转。

### 3.7 每日复盘 (`daily_reviewer.py`) — 市场观点无依据

`daily_reviewer.py` 生成每日复盘报告，包含 `market_view`（市场观点）字段，但:

- 无当日市场涨跌数据
- 无板块轮动数据
- 无北向资金数据
- "市场观点"仅基于微博情绪，与实际市场走势可能完全脱节

---

## 四、影响链路图

```
缺失: 实盘行情数据源
  │
  ├──→ 多因子模型
  │      ├── momentum因子 → 用情绪变化率替代价格动量 → 因子失真
  │      ├── volatility因子 → 用情绪标准差替代价格波动率 → 因子失真
  │      └── capital_flow因子 → 用大V一致性替代资金流向 → 概念偷换
  │      └── 综合评分 → 3/5因子失真 → 选股结果不可信
  │
  ├──→ 模拟交易
  │      ├── 成交价格 → 无来源 → 交易记录价格不可信
  │      ├── 盈亏计算 → 基于不可信价格 → 盈亏不可信
  │      └── 收益曲线 → 基于不可信盈亏 → 曲线无意义
  │
  ├──→ 仓位管理
  │      ├── 止损检查 → 无current_price → 永远不触发
  │      ├── 止盈检查 → 无current_price → 永远不触发
  │      └── 浮动盈亏 → 无法更新 → 持仓状态过时
  │
  ├──→ 回测框架
  │      └── 无价格数据 → 无法运行 → 回测结果为空
  │
  ├──→ 预测器
  │      └── 无历史收益标签 → 无法训练/验证 → 预测无基准
  │
  └──→ 每日复盘
         └── 无市场实际走势 → 复盘与真实市场脱节
```

**结论: 缺失行情数据源的影响不是局部的，而是系统性的。它从底层腐蚀了选股、交易、风控、回测、复盘5个核心模块的可信度。**

---

## 五、严重程度评估

| 评估维度 | 评级 | 说明 |
|----------|:---:|------|
| 影响范围 | **致命** | 波及5个核心模块，非局部问题 |
| 影响深度 | **致命** | 因子失真导致选股结果从根本上不可信 |
| 修复难度 | **中等** | 接入免费行情API（AKShare/Tushare）技术难度不高 |
| 修复工作量 | **中等** | 需新增数据表、数据拉取模块、因子重写、交易逻辑修正 |
| 紧迫度 | **P0** | 不修复则整个系统的投资决策输出无价值 |

---

## 六、修复方案

### 6.1 接入行情数据源（推荐AKShare，免费无需Token）

```python
# 新增模块: app/market/data_fetcher.py
import akshare as ak

class MarketDataFetcher:
    """A股行情数据获取"""

    async def get_daily_price(self, stock_code: str, days: int = 30) -> pd.DataFrame:
        """获取日线行情 (开高低收量)"""
        # ak.stock_zh_a_hist(symbol="600519", period="daily", adjust="qfq")

    async def get_realtime_price(self, stock_code: str) -> float:
        """获取实时价格"""
        # ak.stock_zh_a_spot_em()

    async def get_capital_flow(self, stock_code: str) -> dict:
        """获取资金流向"""
        # ak.stock_individual_fund_flow(stock="600519", market="sh")

    async def get_sector_data(self) -> pd.DataFrame:
        """获取板块数据"""
        # ak.stock_board_industry_name_em()
```

### 6.2 新增数据库表

```sql
-- 日线行情表
CREATE TABLE stock_daily (
    id INTEGER PRIMARY KEY,
    stock_code TEXT NOT NULL,
    trade_date DATE NOT NULL,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL,
    volume INTEGER,
    amount REAL,
    turnover_rate REAL,
    pct_change REAL,
    UNIQUE(stock_code, trade_date)
);

-- 实时行情快照
CREATE TABLE stock_realtime (
    id INTEGER PRIMARY KEY,
    stock_code TEXT UNIQUE NOT NULL,
    current_price REAL,
    change_pct REAL,
    volume INTEGER,
    amount REAL,
    updated_at DATETIME
);

-- 资金流向表
CREATE TABLE capital_flows (
    id INTEGER PRIMARY KEY,
    stock_code TEXT NOT NULL,
    trade_date DATE NOT NULL,
    main_net_inflow REAL,
    north_net_inflow REAL,
    UNIQUE(stock_code, trade_date)
);
```

### 6.3 因子模型修正

| 因子 | 当前实现 | 修正后实现 |
|------|----------|-----------|
| momentum | 情绪5日变化率 | **过去5/10/20日股价收益率** |
| volatility | 情绪标准差 | **过去20日股价日收益率标准差** |
| capital_flow | 大V情绪一致性 | **主力资金净流入 + 北向资金净买入** |

修正后，舆情数据（情绪+热度）作为独立的Alpha因子，行情数据（动量+波动+资金）作为独立的Beta因子，两者正交组合，这才是专业的多因子建模方法。

### 6.4 交易系统修正

- `execute_buy/sell` 从 `MarketDataFetcher` 获取实时/收盘价格
- `PositionManager` 增加定时价格刷新任务（每分钟更新一次 `current_price`）
- 止损止盈检查基于真实价格触发
- 交易手续费基于真实成交金额计算

### 6.5 配置新增

```yaml
# config.yaml 新增段
market_data:
  provider: "akshare"           # akshare | tushare | baostock
  tushare_token: "${TUSHARE_TOKEN}"  # Tushare需要Token
  refresh_interval: 60          # 实时行情刷新间隔(秒)
  daily_sync_time: "15:30"      # 日线数据同步时间
  history_days: 365             # 默认拉取历史天数
```

---

## 七、总结

| 问题 | 确认状态 |
|------|:---:|
| 系统无实盘行情数据源 | **确认** — 无任何行情API/SDK/配置 |
| 选股因子依赖静态/替代数据 | **确认** — 3/5因子用舆情数据替代行情数据 |
| 模拟交易成交价格无来源 | **确认** — price参数无真实数据填充路径 |
| 止损止盈无法实际触发 | **确认** — current_price无更新机制 |
| 回测框架无法运行 | **确认** — 无历史价格数据 |
| 预测器无训练/验证基准 | **确认** — 无历史收益率标签 |

**这不是一个"可以后续优化"的问题，而是系统核心链路的基础设施缺失。** 就像建了一栋楼，水电管道都设计好了，但从未接入市政管网——所有水龙头都拧不出水。

修复优先级: **P0 — 在任何投资决策功能可用之前，必须先完成。**

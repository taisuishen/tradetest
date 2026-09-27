# OKX 合约行情分析 + 15 分钟级缠论模拟交易

| 文件 | 作用 |
|---|---|
| `方法论.md` | 从《基础分享课程》整理出的分析方法论，外加合约数据层（订单簿 / OI / 资金费率 / 多空比 / CVD） |
| `okx_analyzer.py` | 单次深度分析：拉数据 → 计算 → 汇总关键位 → 草拟情景 → 调用 DeepSeek 写报告 → 程序复算盈亏比 |
| `trader15.py` | **主策略**：15 分钟级缠论模拟交易，每 5 分钟运行一次，严格止损，记录胜率和利润 |
| `ta.py` | 课程里的指标和形态识别 |
| `okx_client.py` | OKX 公共行情接口（不需要 API Key） |
| `config.example.json` | 配置模板。复制为 `config.json` 并填 Key，或用环境变量 `LLM_API_KEY` |
| `deploy/install.sh` | Linux 服务器一键部署（systemd 定时器 + 看板） |

依赖：`pip install -r requirements.txt`

## 深度分析报告

```bash
python okx_analyzer.py ETH-USDT-SWAP
python okx_analyzer.py SOL-USDT-SWAP --no-llm   # 只算数据，不调用大模型
```

近期有 FOMC、CPI、非农等事件时，可写进 `config.json` 的 `events`。

## 模拟交易（主策略：15 分钟级缠论，`trader15.py`）

只模拟，不下任何真实订单，也不需要 OKX 账户。规则来自 `chan15_lab.py` 的控制变量测试（区间套 + 大方向过滤 + 只做三类买卖点）：

- **入场**：15m 新确认的三买做多、三卖做空，按当前价成交，每次 1 倍仓位（ETH 1 个 / BTC 0.03 个），不加仓。
- **过滤**：
  - 大方向：1H（0.6）+ 4H（0.4）加权趋势分不能明显相反。
  - 区间套：1H 缠论方向（走势 + 近期买卖点）不能相反。
- **止损**：买卖点价外 0.5 倍 ATR(15m)。用 1 分钟 K 线逐根判定，按止损价原价成交（不计滑点）。
- **离场**：止损，或持仓中 15m 出现反向买卖点就按当时价格平仓；不设固定止盈。
- **手续费**：每笔平仓每 1 倍扣 2U。
- **记录**：数据存在 `trader15.db`，看板在 `web/index.html`（每 60 秒自动刷新）。

```bash
python trader15.py run       # 执行一次（定时任务每 5 分钟调用）
python trader15.py report    # 胜率、利润、持仓、最近成交
```

**回测参考**（最近 120 天，ETH + BTC，已扣手续费，未计资金费率）：123 笔，胜率 19.5%，净利 +565U，组合最大回撤 246U。
按两个合约的名义本金约 4,018U（不加杠杆）计，年化约 +43%，但每个月差异很大（9 月为 −111U）。
这个策略胜率低，利润靠少数大赚的单子，不能只看胜率。

## 研究工具

| 脚本 | 用途 |
|---|---|
| `chan15_lab.py 120` | 15m 缠论控制变量实验台：一次只改一个因素，并按前后两半检验（前一半相当于样本外） |
| `bt_annual.py 120` | 主策略的年化、回撤和按月净利 |
| `chan.py` | 缠论实现：包含处理 → 分型 → 笔 → 笔中枢 → 趋势 / 盘整背驰 → 三类买卖点 |
| `paper_trader.py` / `backtest.py` | 备选：小时级马丁分批挂单（0.2 / 0.4 / 0.8 / 1.6 / 3.2 倍）及其回测 |
| `backtest_chan.py` / `chan_strategy.py` | 1H 级缠论单仓策略回测 |
| `bt_parallel.py` | 回测多进程加速（16 核下 60 天回测从约 15 分钟降到约 35 秒） |

> 马丁分批挂单的 60 天回测胜率 89–93%，但净亏 −741U。高胜率不代表赚钱，详见 `backtest.py`。

## 部署到 Linux 服务器

```bash
git clone https://github.com/taisuishen/tradetest.git && cd tradetest
sudo bash deploy/install.sh 8080
```

装好后每 5 分钟自动运行一次。看板地址是 `http://服务器IP:8080/`，需要在云厂商安全组放行 8080 端口。
服务器需要能访问 www.okx.com：美国 IP 可能被 OKX 限制，建议用香港、新加坡或日本的机房。

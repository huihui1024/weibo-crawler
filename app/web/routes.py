"""
Web路由模块
"""
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, Form, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..market import get_market_client, MarketType
from ..trading import (
    SimulatedTrading, BrokerConfig, OrderExecutor, PositionManager,
    Order, OrderType, OrderDirection,
)

from ..config import get_config
from ..database import (
    init_db, get_all_bloggers, get_bloggers_with_stats,
    get_blogger, add_blogger, update_blogger, delete_blogger,
    get_weibo_posts, get_weibo_posts_count,
    search_weibo_posts, search_weibo_posts_count,
    get_sync_db, Blogger, WeiboPost
)
from ..scheduler import get_scheduler, init_scheduler, init_stock_scheduler
from ..crawler.weibo import get_crawler, set_global_cookie, get_global_cookie, clear_global_cookie
from ..database import save_cookie as db_save_cookie, load_cookie as db_load_cookie, delete_cookie as db_delete_cookie
from ..analyzer.stock_analyzer import StockAnalyzer


# 创建FastAPI应用
app = FastAPI(title="微博博主内容抓取", description="抓取并展示微博博主的内容")

# 模板配置
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    await init_db()
    await sync_bloggers_from_config()

    # 从数据库加载Cookie
    cookie = await db_load_cookie()
    if cookie:
        set_global_cookie(cookie)
        print(f"从数据库加载Cookie，长度: {len(cookie)}")

    await init_scheduler()
    await init_stock_scheduler()


async def sync_bloggers_from_config():
    """从配置文件同步博主到数据库"""
    config = get_config()
    for blogger_config in config.bloggers:
        existing = await get_blogger_by_uid(blogger_config.uid)
        if not existing:
            blogger = Blogger(
                weibo_uid=blogger_config.uid,
                nickname=blogger_config.nickname,
                crawl_interval=blogger_config.interval,
                enabled=blogger_config.enabled
            )
            await add_blogger(blogger)


# ============ 页面路由 ============

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/bloggers", response_class=HTMLResponse)
async def bloggers_page(request: Request):
    """博主管理页面（兼容旧路由）"""
    return RedirectResponse(url="/")


@app.get("/history/{blogger_id}", response_class=HTMLResponse)
async def history_page(request: Request, blogger_id: int):
    """历史页面（兼容旧路由）"""
    return RedirectResponse(url="/")


# ============ API路由 ============

@app.get("/api/bloggers")
async def api_get_bloggers():
    """获取博主列表"""
    bloggers_with_stats = await get_bloggers_with_stats()
    scheduler = get_scheduler()
    jobs_info = {j['job_id']: j for j in scheduler.get_jobs_info()}

    for item in bloggers_with_stats:
        job_id = f"crawl_blogger_{item.blogger.id}"
        item.next_run_time = jobs_info.get(job_id, {}).get('next_run_time', '未调度')

    return {"bloggers": [item.model_dump() for item in bloggers_with_stats]}


@app.get("/api/posts")
async def api_get_posts(
    blogger_id: Optional[int] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """获取微博列表"""
    posts = []
    blogger_names = {}

    if blogger_id:
        posts = await get_weibo_posts(blogger_id, limit=limit, offset=offset)
        blogger = await get_blogger(blogger_id)
        blogger_names[blogger_id] = blogger.nickname if blogger else f"博主{blogger_id}"
    else:
        # 获取所有博主的微博
        bloggers = await get_all_bloggers()
        for b in bloggers:
            blogger_names[b.id] = b.nickname or b.weibo_uid
            b_posts = await get_weibo_posts(b.id, limit=limit, offset=offset)
            posts.extend(b_posts)
        # 按时间排序
        posts.sort(key=lambda x: x.post.publish_time or datetime.min, reverse=True)
        posts = posts[:limit]

    # 构建返回数据
    result = []
    for item in posts:
        bid = item.post.blogger_id
        if bid not in blogger_names:
            b = await get_blogger(bid)
            blogger_names[bid] = b.nickname if b and b.nickname else (b.weibo_uid if b else f"博主{bid}")

        result.append({
            "post": item.post.model_dump(),
            "images": [img.model_dump() for img in item.images],
            "blogger_name": blogger_names.get(bid, "未知博主")
        })

    return {"posts": result}


@app.get("/api/search")
async def api_search_posts(
    keyword: str = Query(..., min_length=1),
    blogger_id: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """搜索微博内容"""
    posts = await search_weibo_posts(keyword, blogger_id=blogger_id, limit=limit, offset=offset)
    total = await search_weibo_posts_count(keyword, blogger_id=blogger_id)

    # 获取博主名称
    blogger_names = {}
    result = []
    for item in posts:
        bid = item.post.blogger_id
        if bid not in blogger_names:
            b = await get_blogger(bid)
            blogger_names[bid] = b.nickname if b and b.nickname else (b.weibo_uid if b else f"博主{bid}")

        result.append({
            "post": item.post.model_dump(),
            "images": [img.model_dump() for img in item.images],
            "blogger_name": blogger_names.get(bid, "未知博主")
        })

    return {"posts": result, "total": total, "keyword": keyword}


@app.get("/api/cookie")
async def api_get_cookie():
    """获取Cookie状态"""
    cookie = await db_load_cookie()

    if not cookie:
        return {"has_cookie": False, "cookie": None, "status": "未设置", "is_valid": False}

    # 检查Cookie是否有效（通过发送测试请求）
    is_valid = await check_cookie_validity(cookie)

    if is_valid:
        return {
            "has_cookie": True,
            "cookie": cookie,
            "status": "有效",
            "is_valid": True,
            "cookie_length": len(cookie)
        }
    else:
        return {
            "has_cookie": True,
            "cookie": cookie,
            "status": "已过期",
            "is_valid": False,
            "cookie_length": len(cookie)
        }


async def check_cookie_validity(cookie: str) -> bool:
    """检查Cookie是否有效"""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # 使用微博API测试Cookie有效性
            response = await client.get(
                "https://weibo.com/ajax/statuses/mymblog",
                params={"uid": "7310477910", "page": 1},  # 使用一个测试UID
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Cookie": cookie,
                    "Referer": "https://weibo.com/"
                }
            )

            if response.status_code == 200:
                data = response.json()
                # 检查返回状态
                if data.get("ok") == 1:
                    return True
                elif data.get("ok") == -100:
                    # Cookie无效
                    return False
            return False
    except Exception as e:
        print(f"检查Cookie有效性失败: {e}")
        return False


@app.post("/api/cookie")
async def api_set_cookie(data: dict = Body(...)):
    """设置Cookie"""
    cookie = data.get("cookie", "")
    # 保存到数据库
    await db_save_cookie(cookie)
    # 同时设置全局变量
    set_global_cookie(cookie)
    print(f"Cookie已保存，长度: {len(cookie) if cookie else 0}")
    return {"success": True}


@app.delete("/api/cookie")
async def api_delete_cookie():
    """删除Cookie"""
    await db_delete_cookie()
    clear_global_cookie()
    print("Cookie已清除")
    return {"success": True}


# ============ 博主操作路由 ============

@app.post("/bloggers/add")
async def add_blogger_handler(
    uid: str = Form(...),
    nickname: str = Form(""),
    interval: int = Form(60),
    enabled: bool = Form(True)
):
    """添加博主"""
    blogger = Blogger(
        weibo_uid=uid,
        nickname=nickname,
        crawl_interval=interval,
        enabled=enabled
    )

    try:
        blogger_id = await add_blogger(blogger)
        if enabled:
            scheduler = get_scheduler()
            await scheduler.add_blogger_job(blogger_id, interval)
        return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bloggers/{blogger_id}/edit")
async def edit_blogger_handler(
    blogger_id: int,
    nickname: str = Form(""),
    interval: int = Form(60),
    enabled: bool = Form(True)
):
    """编辑博主"""
    blogger = await get_blogger(blogger_id)
    if not blogger:
        raise HTTPException(status_code=404, detail="博主不存在")

    blogger.nickname = nickname
    blogger.crawl_interval = interval
    blogger.enabled = enabled

    await update_blogger(blogger)

    scheduler = get_scheduler()
    scheduler.update_blogger_job(blogger_id, interval, enabled)

    return RedirectResponse(url="/", status_code=303)


@app.post("/bloggers/{blogger_id}/delete")
async def delete_blogger_handler(blogger_id: int):
    """删除博主"""
    scheduler = get_scheduler()
    scheduler.remove_blogger_job(blogger_id)
    await delete_blogger(blogger_id)
    return RedirectResponse(url="/", status_code=303)


# ============ 图片路由 ============

@app.get("/images/{weibo_id}/{filename}")
async def get_image(weibo_id: str, filename: str):
    """获取图片"""
    config = get_config()
    image_path = Path(config.storage.image_dir) / weibo_id / filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")

    return FileResponse(image_path)


# ============ 爬取路由 ============

@app.post("/crawl/all")
async def crawl_all():
    """手动爬取所有博主"""
    # 检查Cookie是否设置
    cookie = get_global_cookie()
    if not cookie:
        return {"success": False, "msg": "未设置Cookie，请先在设置中配置微博Cookie"}

    from ..crawler.weibo import WeiboCrawler

    crawler = WeiboCrawler()
    results = await crawler.crawl_all_bloggers()

    # 检查是否有Cookie相关的错误
    cookie_errors = []
    for uid, result in results.items():
        if not result.get('success') and result.get('error'):
            if 'Cookie' in result['error'] or '登录' in result['error'] or '过期' in result['error']:
                cookie_errors.append(uid)

    if cookie_errors and len(cookie_errors) == len(results):
        return {"success": False, "msg": "Cookie无效或已过期，请重新设置"}

    return {"success": True, "results": results}


@app.post("/crawl/{blogger_id}")
async def manual_crawl(blogger_id: int):
    """手动触发爬取"""
    blogger = await get_blogger(blogger_id)
    if not blogger:
        raise HTTPException(status_code=404, detail="博主不存在")

    # 检查Cookie是否设置
    cookie = get_global_cookie()
    if not cookie:
        return {"success": False, "msg": "未设置Cookie，请先在设置中配置微博Cookie"}

    crawler = await get_crawler()
    try:
        new_count = await crawler.crawl_blogger(blogger)
        return {"success": True, "new_count": new_count}
    except Exception as e:
        error_msg = str(e)
        if "Cookie" in error_msg or "登录" in error_msg or "过期" in error_msg:
            return {"success": False, "msg": "Cookie无效或已过期，请重新设置"}
        raise HTTPException(status_code=500, detail=error_msg)


# ============ 分析路由 ============

@app.get("/analyze/{blogger_id}")
async def analyze_blogger(blogger_id: int):
    """分析博主"""
    blogger = await get_blogger(blogger_id)
    if not blogger:
        raise HTTPException(status_code=404, detail="博主不存在")

    # 获取该博主的所有微博
    posts = await get_weibo_posts(blogger_id, limit=1000)

    if not posts:
        return {"success": False, "message": "该博主暂无微博数据，请先爬取"}

    # 获取配置
    config = get_config()
    image_dir = config.storage.image_dir

    # 转换为分析器需要的格式（包含图片路径）
    posts_data = []
    for item in posts:
        # 构建图片路径列表
        image_paths = []
        if item.images:
            for img in item.images:
                img_path = Path(image_dir) / item.post.weibo_id / Path(img.local_path).name
                if img_path.exists():
                    image_paths.append(str(img_path))

        posts_data.append({
            'content': item.post.content,
            'publish_time': item.post.publish_time,
            'is_repost': item.post.is_repost,
            'weibo_id': item.post.weibo_id,
            'image_paths': image_paths
        })

    # 执行分析
    analyzer = StockAnalyzer()
    result = analyzer.analyze_posts(posts_data, blogger.nickname or blogger.weibo_uid)

    # 构建仓位配置字符串
    position_config = ""
    if result.position_allocation:
        total = sum(result.position_allocation.values())
        parts = []
        for sector, ratio in sorted(result.position_allocation.items(), key=lambda x: x[1], reverse=True):
            parts.append(f"{sector}{ratio}%")
        if total < 100:
            parts.append(f"现金{100-total}%")
        position_config = "、".join(parts)

    return {
        "success": True,
        "blogger_name": result.blogger_name,
        "analysis_text": result.analysis_text,
        "current_positions": result.current_positions,
        "watch_positions": result.watch_positions,
        "closed_positions": result.closed_positions,
        "likely_positions": result.likely_positions,
        "bearish_positions": result.bearish_positions,
        "mentioned_positions": result.mentioned_positions,
        "industry_distribution": result.industry_distribution,
        "selection_factors": result.selection_factors,
        "selection_formula": result.selection_formula,
        "trading_cycle": result.trading_cycle,
        "trading_cycle_detail": result.trading_cycle_detail,
        "stop_profit": result.stop_profit,
        "stop_loss": result.stop_loss,
        "discipline_score": result.discipline_score,
        "common_mistakes": result.common_mistakes,
        "market_type": result.market_type,
        "position_style": result.position_style,
        "hot_industries": result.hot_industries,
        "avoid_stocks": result.avoid_stocks,
        "position_allocation": result.position_allocation,
        "position_config": position_config,
        "no_position_sectors": result.no_position_sectors,
        "attack_sector": result.attack_sector,
        "defense_sector": result.defense_sector,
        "k_pattern": result.kline_patterns[0] if result.kline_patterns else "",
        "watch_stocks": _build_watch_stocks(result),
    }


def _build_watch_stocks(result) -> list:
    """构建关注股票列表（美股 -> A股映射）"""
    watch_stocks = []
    us_stock_mapping = {
        'NVIDIA': {'a_stocks': '中际旭创、天孚通信、新易盛', 'count': 3},
        'ASML': {'a_stocks': '北方华创、中微公司、福晶科技', 'count': 3},
        'AMD': {'a_stocks': '海光信息、中际旭创、紫光国微', 'count': 3},
        'Intel': {'a_stocks': '中芯国际、北方华创、韦尔股份', 'count': 3},
        'Micron': {'a_stocks': '中芯国际、北方华创、澜起科技', 'count': 3},
        'Corning': {'a_stocks': '长飞光纤、亨通光电、中天科技', 'count': 3},
        'Tesla': {'a_stocks': '宁德时代、比亚迪、德赛西威', 'count': 3},
    }

    for pos in result.watch_positions:
        stock_name = pos.get('name', '')
        if stock_name in us_stock_mapping:
            info = us_stock_mapping[stock_name]
            watch_stocks.append({
                'us_stock': stock_name,
                'count': info['count'],
                'a_stocks': info['a_stocks']
            })

    return watch_stocks[:5]  # 最多返回5个


# 导入需要的函数
from ..database import get_blogger_by_uid


# ============ 智能选股路由 ============

@app.get("/stock", response_class=HTMLResponse)
async def stock_page(request: Request):
    """智能选股页面"""
    return templates.TemplateResponse("stock.html", {"request": request})


@app.get("/api/stock-pool")
async def api_get_stock_pool(blogger_name: Optional[str] = Query(None)):
    """获取股票池（含多因子量化数据）"""
    import sqlite3
    import json

    conn = get_sync_db()
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_pool'")
    if not cursor.fetchone():
        conn.close()
        return {"stocks": [], "blogger_name": None}

    # 获取股票池数据（含因子列）
    if blogger_name:
        cursor.execute("""
            SELECT stock_code, stock_name, sector, concept, price, change_pct,
                   style_match_score, selection_source, position_weight, selection_reasons,
                   technical_indicators, added_date, blogger_name,
                   fundamental_score, valuation_score, capital_flow_score, sentiment_score,
                   pe_percentile, pb_percentile, market_regime, factor_detail,
                   cognitive_gap_score, supply_chain_score, short_interest_ratio,
                   volume_surge_ratio, industry_chain
            FROM stock_pool
            WHERE status = 'active' AND (blogger_name = ? OR blogger_name IS NULL)
            ORDER BY
                CASE WHEN selection_source = '市场热点' THEN 0
                     WHEN selection_source = '行业方向' OR selection_source = '基本面驱动' THEN 1
                     WHEN selection_source = '量化确认' OR selection_source = '系统推荐' OR selection_source = '量化策略' THEN 2
                     ELSE 3 END,
                style_match_score DESC
        """, (blogger_name,))
        actual_blogger_name = blogger_name
    else:
        cursor.execute("""
            SELECT stock_code, stock_name, sector, concept, price, change_pct,
                   style_match_score, selection_source, position_weight, selection_reasons,
                   technical_indicators, added_date, blogger_name,
                   fundamental_score, valuation_score, capital_flow_score, sentiment_score,
                   pe_percentile, pb_percentile, market_regime, factor_detail,
                   cognitive_gap_score, supply_chain_score, short_interest_ratio,
                   volume_surge_ratio, industry_chain
            FROM stock_pool
            WHERE status = 'active'
            ORDER BY
                blogger_name,
                CASE WHEN selection_source = '市场热点' THEN 0
                     WHEN selection_source = '行业方向' OR selection_source = '基本面驱动' THEN 1
                     WHEN selection_source = '量化确认' OR selection_source = '系统推荐' OR selection_source = '量化策略' THEN 2
                     ELSE 3 END,
                style_match_score DESC
        """)
        actual_blogger_name = None

    rows = cursor.fetchall()
    conn.close()

    stocks = []
    for row in rows:
        stock = {
            "code": row[0],
            "name": row[1],
            "sector": row[2],
            "concept": row[3],
            "price": row[4],
            "change_pct": row[5],
            "style_match_score": row[6],
            "selection_source": row[7] if len(row) > 7 and row[7] else "市场热点",
            "position_weight": row[8] if len(row) > 8 else 0,
            "selection_reasons": json.loads(row[9]) if len(row) > 9 and row[9] else [],
            "technical_indicators": json.loads(row[10]) if len(row) > 10 and row[10] else {},
            "added_date": row[11] if len(row) > 11 else None,
            "blogger_name": row[12] if len(row) > 12 else None,
        }
        # 多因子量化数据（兼容旧数据无因子列）
        if len(row) > 13:
            stock["fundamental_score"] = row[13] if row[13] else 0
            stock["valuation_score"] = row[14] if row[14] else 0
            stock["capital_flow_score"] = row[15] if row[15] else 0
            stock["sentiment_score"] = row[16] if row[16] else 0
            stock["pe_percentile"] = row[17] if row[17] else 0
            stock["pb_percentile"] = row[18] if row[18] else 0
            stock["market_regime"] = row[19] if row[19] else ''
            stock["factor_detail"] = json.loads(row[20]) if len(row) > 20 and row[20] else {}
            # 新增因子列（产业链、认知差、轧空等）
            stock["cognitive_gap_score"] = row[21] if len(row) > 21 and row[21] else 0
            stock["supply_chain_score"] = row[22] if len(row) > 22 and row[22] else 0
            stock["short_interest_ratio"] = row[23] if len(row) > 23 and row[23] else 0
            stock["volume_surge_ratio"] = row[24] if len(row) > 24 and row[24] else 0
            stock["industry_chain"] = row[25] if len(row) > 25 and row[25] else ''
        else:
            stock["fundamental_score"] = 0
            stock["valuation_score"] = 0
            stock["capital_flow_score"] = 0
            stock["sentiment_score"] = 0
            stock["pe_percentile"] = 0
            stock["pb_percentile"] = 0
            stock["market_regime"] = ''
            stock["factor_detail"] = {}
            stock["cognitive_gap_score"] = 0
            stock["supply_chain_score"] = 0
            stock["short_interest_ratio"] = 0
            stock["volume_surge_ratio"] = 0
            stock["industry_chain"] = ''
        stocks.append(stock)

    # 补充预测数据（从predictions表关联）
    try:
        stock_codes = [s["code"] for s in stocks]
        if stock_codes:
            conn2 = get_sync_db()
            cursor2 = conn2.cursor()
            placeholders = ",".join(["?"] * len(stock_codes))
            cursor2.execute(f"""
                SELECT stock_code, expected_close_change, expected_trend, expected_strength,
                       expected_amplitude, expected_hold_days, expected_hold_return
                FROM predictions
                WHERE stock_code IN ({placeholders})
                AND prediction_date >= date('now', '-1 day')
                ORDER BY prediction_time DESC
            """, stock_codes)
            pred_map = {}
            for row in cursor2.fetchall():
                code = row[0]
                if code not in pred_map:  # 取最新一条
                    pred_map[code] = {
                        "expected_close_change": row[1],
                        "expected_trend": row[2],
                        "expected_strength": row[3],
                        "expected_amplitude": row[4],
                        "expected_hold_days": row[5],
                        "expected_hold_return": row[6],
                    }
            conn2.close()
            for stock in stocks:
                pred = pred_map.get(stock["code"])
                if pred:
                    stock["prediction"] = pred
                else:
                    stock["prediction"] = None
    except Exception:
        for stock in stocks:
            stock["prediction"] = None

    # 添加实盘模拟验证状态
    simulation_status = {}
    try:
        from ..selector.stock_selector import get_simulation_status, get_dynamic_win_rate
        simulation_status = get_simulation_status(blogger_name=actual_blogger_name)
    except Exception:
        simulation_status = {"status": "unknown", "warning": None}

    # 动态胜率风险提示
    dyn = {}
    try:
        dyn = get_dynamic_win_rate()
        wr = dyn.get('win_rate', 0) or dyn.get('backtest_win_rate', 64.0)
        signal = dyn.get('signal_strength', 'normal')
        if dyn.get('total_samples', 0) > 0:
            risk_warning = f"系统近期胜率{wr}%，非确定性预测。单笔建议仓位≤5%，严格止损。"
        else:
            risk_warning = f"系统回测胜率{wr}%（无实盘验证），非确定性预测。单笔建议仓位≤5%，严格止损。"
        if signal == 'suspended':
            risk_warning += " ⚠️ 荐股信号已暂停，请勿跟投。"
        elif signal == 'degraded':
            risk_warning += " ⚠️ 荐股信号降级，建议减仓观望。"
    except Exception:
        risk_warning = "系统非确定性预测。单笔建议仓位≤5%，严格止损。"

    # P2: 强制实盘模拟警告 — 系统尚未完成3个月实盘验证
    sim_days = simulation_status.get("days", 0)
    if sim_days < 90:
        risk_warning += f" ⚠️ 系统尚未完成3个月实盘模拟验证(当前{sim_days}天/需90天)，所有推荐仅供参考，请勿直接跟投。"
    else:
        risk_warning += " 系统已完成3个月实盘模拟验证。"

    if simulation_status.get("warning"):
        risk_warning += " " + simulation_status["warning"]

    # 大盘择时信息
    market_timing = {}
    try:
        from ..selector.market_timing import MarketTiming
        mt_engine = MarketTiming()
        mt_result = mt_engine.get_timing()
        market_timing = {
            "timing": mt_result['timing'],
            "label": mt_result['config']['label'],
            "reason": mt_result['reason'],
            "trend": mt_result['trend'],
            "max_position_pct": mt_result['config']['max_position'] * 100,
            "pool_ratio": mt_result['config']['pool_ratio'],
            "allow_hot_stocks": mt_result['config']['allow_hot_stocks'],
        }
        if not mt_result['config']['allow_hot_stocks']:
            risk_warning += " ⚠️ 大盘走弱，纯热点题材已屏蔽。"
    except Exception:
        market_timing = {"timing": "neutral", "label": "震荡", "reason": "择时数据不可用"}

    return {
        "stocks": stocks,
        "blogger_name": actual_blogger_name,
        "simulation_status": simulation_status,
        "risk_warning": risk_warning,
        "dynamic_win_rate": dyn if 'dyn' in dir() else {},
        "market_timing": market_timing,
    }


@app.get("/api/predictions")
async def api_get_predictions(
    date: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    blogger_name: Optional[str] = Query(None),
    limit: int = Query(20)
):
    """获取预测记录"""
    import sqlite3
    import json

    conn = get_sync_db()
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    if not cursor.fetchone():
        conn.close()
        return {"predictions": []}

    query = """
        SELECT stock_code, stock_name, prediction_date, prediction_time,
               current_price, expected_close_change, actual_change, expected_amplitude,
               expected_trend, expected_hold_days, expected_hold_return,
               expected_buy_point, expected_sell_point, expected_sell_price,
               selection_reason, risk_warning, status, blogger_name
        FROM predictions
        WHERE 1=1
    """
    params = []

    if date:
        query += " AND prediction_date = ?"
        params.append(date)
    if status:
        query += " AND status = ?"
        params.append(status)
    if blogger_name:
        query += " AND blogger_name = ?"
        params.append(blogger_name)

    query += " ORDER BY prediction_date DESC, prediction_time DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    predictions = []
    for row in rows:
        predictions.append({
            "code": row[0],
            "name": row[1],
            "prediction_date": row[2],
            "prediction_time": row[3],
            "current_price": row[4],
            "expected_change": row[5],
            "actual_change": row[6],
            "expected_amplitude": row[7],
            "expected_trend": row[8],
            "expected_hold_days": row[9],
            "expected_hold_return": row[10],
            "buy_point": row[11],
            "sell_point": row[12],
            "sell_price": row[13],
            "selection_reason": row[14],
            "risk_warning": row[15],
            "status": row[16],
            "blogger_name": row[17] if len(row) > 17 else None
        })

    return {"predictions": predictions}


@app.get("/api/reviews")
async def api_get_reviews(
    date: Optional[str] = Query(None),
    limit: int = Query(10)
):
    """获取复盘记录"""
    import sqlite3
    import json

    conn = get_sync_db()
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_reviews'")
    if not cursor.fetchone():
        conn.close()
        return {"reviews": []}

    query = """
        SELECT review_date, review_time, total_predictions, correct_predictions,
               accuracy_rate, avg_change_diff, avg_amplitude_diff, avg_prediction_score,
               lessons_learned, improvement_suggestions
        FROM daily_reviews
        WHERE 1=1
    """
    params = []

    if date:
        query += " AND review_date = ?"
        params.append(date)

    query += " ORDER BY review_date DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    reviews = []
    for row in rows:
        reviews.append({
            "review_date": row[0],
            "review_time": row[1],
            "total_predictions": row[2],
            "correct_predictions": row[3],
            "accuracy_rate": row[4],
            "avg_change_diff": row[5],
            "avg_amplitude_diff": row[6],
            "avg_prediction_score": row[7],
            "lessons_learned": json.loads(row[8]) if row[8] else [],
            "improvement_suggestions": json.loads(row[9]) if row[9] else []
        })

    return {"reviews": reviews}


@app.post("/api/refresh-pool")
async def api_refresh_pool(blogger_name: str = Query(None)):
    """刷新股票池（手动触发选股）"""
    try:
        # 异步执行选股任务
        from ..scheduler import run_stock_selection_task
        await run_stock_selection_task(blogger_name)

        # 获取模拟验证状态
        from ..selector.stock_selector import get_simulation_status
        sim = get_simulation_status(blogger_name=blogger_name)

        message = f"股票池已刷新（基于博主: {blogger_name or '默认'}）"
        if sim.get('warning'):
            message += f"。⚠️ {sim['warning']}"

        # 动态胜率风险提示
        dyn = {}
        try:
            dyn = get_dynamic_win_rate()
            wr = dyn.get('win_rate', 0) or dyn.get('backtest_win_rate', 64.0)
            signal = dyn.get('signal_strength', 'normal')
            if dyn.get('total_samples', 0) > 0:
                risk_msg = f"系统近期胜率{wr}%，非确定性预测。单笔建议仓位≤5%，严格止损。"
            else:
                risk_msg = f"系统回测胜率{wr}%（无实盘验证），非确定性预测。单笔建议仓位≤5%，严格止损。"
            if signal == 'suspended':
                risk_msg += " ⚠️ 荐股信号已暂停。"
            elif signal == 'degraded':
                risk_msg += " ⚠️ 荐股信号降级。"
        except Exception:
            risk_msg = "系统非确定性预测。单笔建议仓位≤5%，严格止损。"

        # P2: 强制实盘模拟警告
        sim_days = sim.get("days", 0)
        if sim_days < 90:
            risk_msg += f" ⚠️ 系统尚未完成3个月实盘模拟验证(当前{sim_days}天/需90天)，所有推荐仅供参考，请勿直接跟投。"

        return {
            "success": True,
            "message": message,
            "simulation_status": sim,
            "risk_warning": risk_msg,
            "dynamic_win_rate": dyn,
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/run-review")
async def api_run_review():
    """手动触发复盘"""
    import asyncio

    try:
        from ..scheduler import run_daily_review_task
        await run_daily_review_task()
        return {"success": True, "message": "复盘已完成"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/api/review-details")
async def api_get_review_details(date: Optional[str] = Query(None)):
    """获取复盘详情，按日期分组"""
    import sqlite3

    conn = get_sync_db()
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='review_results'")
    if not cursor.fetchone():
        conn.close()
        return {"details": [], "grouped": {}}

    if date:
        cursor.execute("""
            SELECT stock_code, stock_name, review_date, expected_change, actual_change,
                   change_diff, expected_amplitude, actual_amplitude, amplitude_diff,
                   expected_trend, actual_trend, trend_match, diff_reason,
                   improvement_suggestion, prediction_score
            FROM review_results
            WHERE review_date = ?
            ORDER BY prediction_score DESC
        """, (date,))
    else:
        cursor.execute("""
            SELECT stock_code, stock_name, review_date, expected_change, actual_change,
                   change_diff, expected_amplitude, actual_amplitude, amplitude_diff,
                   expected_trend, actual_trend, trend_match, diff_reason,
                   improvement_suggestion, prediction_score
            FROM review_results
            ORDER BY review_date DESC, prediction_score DESC
            LIMIT 100
        """)

    rows = cursor.fetchall()
    conn.close()

    details = []
    grouped = {}  # 按日期分组

    for row in rows:
        review_date = row[2] if len(row) > 2 else ''
        item = {
            "code": row[0],
            "name": row[1],
            "review_date": review_date,
            "expected_change": row[3],
            "actual_change": row[4],
            "change_diff": row[5],
            "expected_amplitude": row[6],
            "actual_amplitude": row[7],
            "amplitude_diff": row[8],
            "expected_trend": row[9],
            "actual_trend": row[10],
            "trend_match": bool(row[11]),
            "diff_reason": row[12],
            "improvement": row[13],
            "score": row[14]
        }
        details.append(item)

        # 按日期分组
        if review_date not in grouped:
            grouped[review_date] = []
        grouped[review_date].append(item)

    return {"details": details, "grouped": grouped}


# ============ 多因子模型和回测API ============

@app.get("/api/factor-score/{code}")
async def api_get_factor_score(code: str):
    """获取单只股票的多因子评分"""
    try:
        from ..selector.multi_factor_model import MultiFactorModel, prepare_stock_data_for_scoring
        from ..market import get_market_client

        market_client = get_market_client()
        model = MultiFactorModel()

        # 准备数据
        data = prepare_stock_data_for_scoring(code, market_client)
        name = data.get('name', code)

        # 评分
        score = model.score_stock(code, name, data)

        return {
            "code": code,
            "name": name,
            "total_score": score.total_score,
            "recommendation": score.recommendation,
            "confidence": score.confidence,
            "factor_scores": [
                {
                    "factor": fs.factor_name,
                    "raw_value": fs.raw_value,
                    "normalized_score": fs.normalized_score,
                    "weight": fs.weight,
                    "weighted_score": fs.weighted_score
                }
                for fs in score.factor_scores
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/run-backtest")
async def api_run_backtest(
    days: int = Body(30, embed=True),
    initial_capital: float = Body(1000000.0, embed=True),
    strategy: str = Body("prediction", embed=True)
):
    """
    运行回测

    Args:
        days: 回测天数
        initial_capital: 初始资金
        strategy: 策略类型 (prediction/multi_factor)
    """
    try:
        from ..selector.backtest import (
            BacktestRunner,
            PredictionBasedStrategy,
            MultiFactorStrategy,
        )
        from datetime import date, timedelta
        import math

        def _clean_float(v):
            """将inf/nan替换为安全的JSON值"""
            if isinstance(v, float):
                if math.isinf(v):
                    return 999.99 if v > 0 else -999.99
                if math.isnan(v):
                    return 0.0
            return v

        def _clean_dict(d):
            """递归清理dict中的inf/nan浮点值"""
            if isinstance(d, dict):
                return {k: _clean_dict(v) for k, v in d.items()}
            if isinstance(d, list):
                return [_clean_dict(v) for v in d]
            return _clean_float(d)

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        runner = BacktestRunner()

        if strategy == "multi_factor":
            strat = MultiFactorStrategy(score_threshold=65)
        else:
            strat = PredictionBasedStrategy(score_threshold=60)

        result = runner.run_backtest(strat, start_date, end_date, initial_capital)

        # 构建净值曲线数据
        equity_curve = []
        drawdown_series = []
        peak = initial_capital
        for dv in result.daily_values:
            total_value = dv['total_value']
            daily_return_pct = ((total_value - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0
            equity_curve.append({
                "date": str(dv['date']),
                "total_value": total_value,
                "cash": dv['cash'],
                "market_value": dv.get('market_value', 0),
                "positions_count": dv.get('positions', 0),
                "return_pct": round(daily_return_pct, 2),
            })
            # 计算回撤
            if total_value > peak:
                peak = total_value
            dd = (peak - total_value) / peak * 100 if peak > 0 else 0
            drawdown_series.append({
                "date": str(dv['date']),
                "drawdown_pct": round(dd, 2),
            })

        # 计算每日收益率分布
        daily_returns = []
        if len(result.daily_values) > 1:
            values = [dv['total_value'] for dv in result.daily_values]
            for i in range(1, len(values)):
                prev = values[i-1]
                r = (values[i] - prev) / prev * 100 if prev > 0 else 0
                daily_returns.append(round(r, 2))

        # 所有交易记录
        trades = [
            {
                "code": t.code,
                "name": t.name,
                "direction": t.direction,
                "date": str(t.date),
                "price": _clean_float(t.price),
                "shares": _clean_float(t.shares),
                "amount": _clean_float(t.amount),
                "commission": _clean_float(t.commission),
                "reason": t.reason,
            }
            for t in result.trades
        ]

        response_data = {
            "success": True,
            "result": {
                "start_date": str(result.start_date),
                "end_date": str(result.end_date),
                "initial_capital": _clean_float(result.initial_capital),
                "final_capital": _clean_float(result.final_capital),
                "total_return": _clean_float(result.total_return),
                "annual_return": _clean_float(result.annual_return),
                "benchmark_return": _clean_float(result.benchmark_return),
                "max_drawdown": _clean_float(result.max_drawdown),
                "volatility": _clean_float(result.volatility),
                "sharpe_ratio": _clean_float(result.sharpe_ratio),
                "sortino_ratio": _clean_float(result.sortino_ratio),
                "total_trades": result.total_trades,
                "win_trades": result.win_trades,
                "lose_trades": result.lose_trades,
                "win_rate": _clean_float(result.win_rate),
                "avg_profit_trade": _clean_float(result.avg_profit_trade),
                "avg_loss_trade": _clean_float(result.avg_loss_trade),
                "profit_factor": _clean_float(result.profit_factor),
                "avg_hold_days": _clean_float(result.avg_hold_days),
                "max_hold_days": _clean_float(result.max_hold_days),
                "trades": trades,
                "equity_curve": equity_curve,
                "drawdown_series": drawdown_series,
                "daily_returns": daily_returns,
            }
        }
        # 全量清理确保无inf/nan
        return JSONResponse(content=_clean_dict(response_data))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/backtest/charts")
async def api_backtest_charts(
    days: int = Query(30, ge=7, le=365),
    strategy: str = Query("prediction"),
    initial_capital: float = Query(1000000.0, ge=100000),
):
    """获取回测图表数据（净值曲线/回撤/收益分布）"""
    try:
        import math
        from ..selector.backtest import (
            BacktestRunner,
            PredictionBasedStrategy,
            MultiFactorStrategy,
        )
        from datetime import date, timedelta

        def _cf(v):
            if isinstance(v, float):
                if math.isinf(v): return 999.99 if v > 0 else -999.99
                if math.isnan(v): return 0.0
            return v

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        runner = BacktestRunner()
        if strategy == "multi_factor":
            strat = MultiFactorStrategy(score_threshold=65)
        else:
            strat = PredictionBasedStrategy(score_threshold=60)

        result = runner.run_backtest(strat, start_date, end_date, initial_capital)

        equity_curve = []
        drawdown_series = []
        peak = initial_capital
        for dv in result.daily_values:
            total_value = dv['total_value']
            ret = (total_value - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0
            equity_curve.append({"date": str(dv['date']), "value": _cf(total_value), "return_pct": round(ret, 2)})
            if total_value > peak:
                peak = total_value
            dd = (peak - total_value) / peak * 100 if peak > 0 else 0
            drawdown_series.append({"date": str(dv['date']), "drawdown": round(dd, 2)})

        daily_returns = []
        if len(result.daily_values) > 1:
            vals = [dv['total_value'] for dv in result.daily_values]
            for i in range(1, len(vals)):
                prev = vals[i-1]
                r = (vals[i] - prev) / prev * 100 if prev > 0 else 0
                daily_returns.append(round(r, 2))

        # 收益率分布直方图
        bins = [-5, -3, -2, -1, 0, 1, 2, 3, 5]
        return_distribution = {}
        for lower in bins:
            upper = lower + 1
            count = sum(1 for r in daily_returns if lower <= r < upper)
            label = f"{lower}%~{upper}%"
            return_distribution[label] = count
        # 极端值
        return_distribution["<-5%"] = sum(1 for r in daily_returns if r < -5)
        return_distribution[">5%"] = sum(1 for r in daily_returns if r >= 5)

        return JSONResponse(content={
            "success": True,
            "summary": {
                "total_return": _cf(result.total_return),
                "annual_return": _cf(result.annual_return),
                "max_drawdown": _cf(result.max_drawdown),
                "sharpe_ratio": _cf(result.sharpe_ratio),
                "win_rate": _cf(result.win_rate),
                "total_trades": result.total_trades,
                "final_capital": _cf(result.final_capital),
            },
            "equity_curve": equity_curve,
            "drawdown_series": drawdown_series,
            "return_distribution": return_distribution,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/backtest-report")
async def api_get_backtest_report(days: int = 30):
    """获取回测报告（文本格式）"""
    try:
        from ..selector.backtest import run_prediction_backtest, generate_backtest_report

        result = run_prediction_backtest(days=days)
        report = generate_backtest_report(result)

        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/strategy-performance")
async def api_get_strategy_performance(days: int = 30):
    """获取策略表现统计"""
    import sqlite3
    from datetime import date, timedelta

    try:
        conn = get_sync_db()
        cursor = conn.cursor()

        start_date = (date.today() - timedelta(days=days)).isoformat()

        # 获取预测准确率
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN trend_match = 1 THEN 1 ELSE 0 END) as correct,
                AVG(prediction_score) as avg_score,
                AVG(ABS(change_diff)) as avg_diff
            FROM review_results
            WHERE review_date >= ?
        """, (start_date,))

        row = cursor.fetchone()
        conn.close()

        if row and row[0] > 0:
            return {
                "total_predictions": row[0],
                "correct_predictions": row[1] or 0,
                "accuracy_rate": (row[1] or 0) / row[0] * 100 if row[0] > 0 else 0,
                "avg_score": row[2] or 0,
                "avg_diff": row[3] or 0
            }
        else:
            return {
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy_rate": 0,
                "avg_score": 0,
                "avg_diff": 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# ── 多市场 & 交易 API ──────────────────────

@app.get("/api/market/quotes")
async def api_market_quotes(market: str = Query("A", description="市场类型 A/HK/US")):
    """获取指定市场的实时行情"""
    try:
        market_type = MarketType(market)
        client = get_market_client(market_type)
        quotes = client.get_realtime_quotes(limit=50)
        return {"market": market, "quotes": quotes, "count": len(quotes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/market/overview")
async def api_market_overview(market: str = Query("A", description="市场类型 A/HK/US")):
    """获取指定市场概览"""
    try:
        market_type = MarketType(market)
        client = get_market_client(market_type)
        overview = client.get_market_overview()
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/market/search")
async def api_market_search(keyword: str = Query(...), market: str = Query("A")):
    """搜索股票"""
    try:
        market_type = MarketType(market)
        client = get_market_client(market_type)
        results = client.search_stock(keyword)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/stock-pool/by-market")
async def api_stock_pool_by_market(market: str = Query("A")):
    """按市场获取股票池"""
    try:
        import sqlite3
        conn = get_sync_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT stock_code, stock_name, sector, concept, price, change_pct,
                   style_match_score, selection_source, market
            FROM stock_pool WHERE status = 'active' AND market = ?
            ORDER BY style_match_score DESC
        """, (market,))
        results = []
        for row in cursor.fetchall():
            results.append({
                'code': row[0], 'name': row[1], 'sector': row[2],
                'concept': row[3], 'price': row[4], 'change_pct': row[5],
                'style_match_score': row[6], 'selection_source': row[7], 'market': row[8],
            })
        conn.close()
        return {"market": market, "stocks": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# ── 交易 API ──────────────────────────────

_trading_instance = None


def _get_trading() -> SimulatedTrading:
    global _trading_instance
    if _trading_instance is None:
        config = BrokerConfig()
        _trading_instance = SimulatedTrading(config)
    return _trading_instance


@app.get("/api/trading/balance")
async def api_trading_balance():
    """获取账户资金"""
    trading = _get_trading()
    balance = await trading.get_balance()
    return balance.to_dict()


@app.get("/api/trading/positions")
async def api_trading_positions():
    """获取持仓"""
    trading = _get_trading()
    positions = await trading.get_positions()
    return [p.to_dict() for p in positions]


@app.get("/api/trading/orders")
async def api_trading_orders(status: Optional[str] = Query(None)):
    """获取订单列表"""
    trading = _get_trading()
    from ..trading import OrderStatus
    order_status = OrderStatus(status) if status else None
    orders = await trading.get_orders(order_status)
    return [o.to_dict() for o in orders]


@app.post("/api/trading/order")
async def api_trading_place_order(data: dict = Body(...)):
    """下单"""
    try:
        trading = _get_trading()
        order = Order(
            order_id=data.get('order_id', ''),
            account_id=0,
            stock_code=data['stock_code'],
            stock_name=data.get('stock_name', ''),
            market=data.get('market', 'A'),
            direction=OrderDirection(data.get('direction', 'buy')),
            order_type=OrderType(data.get('order_type', 'market')),
            price=data.get('price', 0),
            quantity=data.get('quantity', 100),
            reason=data.get('reason', ''),
        )
        result = await trading.place_order(order)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/trading/signal")
async def api_trading_signal(data: dict = Body(...)):
    """信号下单"""
    try:
        executor = OrderExecutor(trading=_get_trading())
        result = await executor.execute_signal(data)
        if result:
            return result.to_dict()
        return {"status": "skipped", "reason": "非交易时间"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/trading/cancel/{order_id}")
async def api_trading_cancel(order_id: str):
    """撤单"""
    trading = _get_trading()
    success = await trading.cancel_order(order_id)
    return {"success": success}


# ── 风险度量 API ──────────────────────────────

@app.get("/api/simulation-status")
async def api_simulation_status(blogger_name: Optional[str] = Query(None)):
    """获取实盘模拟验证状态"""
    try:
        from ..selector.stock_selector import get_simulation_status
        status = get_simulation_status(blogger_name=blogger_name)
        return {"success": True, **status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/dynamic-win-rate")
async def api_dynamic_win_rate(recent_days: int = Query(30)):
    """获取动态胜率（实盘滚动计算）"""
    try:
        from ..selector.stock_selector import get_dynamic_win_rate
        result = get_dynamic_win_rate(recent_days=recent_days)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/portfolio")
async def api_get_portfolio(blogger_name: Optional[str] = Query(None)):
    """获取最新投资组合"""
    try:
        from ..selector.portfolio_constructor import get_latest_portfolio
        portfolio = get_latest_portfolio(blogger_name=blogger_name)
        if not portfolio:
            return {"success": True, "portfolio": None, "message": "暂无组合数据，请先运行选股"}
        return {"success": True, "portfolio": portfolio}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/trading-decision")
async def api_trading_decision(blogger_name: Optional[str] = Query(None)):
    """获取最新交易决策报告"""
    try:
        from ..selector.trading_decision import get_latest_decision_report
        report = get_latest_decision_report(blogger_name=blogger_name)
        if not report:
            return {"success": True, "report": None, "message": "暂无决策报告，请先运行选股"}
        return {"success": True, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/run-trading-decision")
async def api_run_trading_decision(blogger_name: Optional[str] = Query(None)):
    """手动触发交易决策分析"""
    try:
        from ..selector.trading_decision import TradingDecisionEngine
        from ..selector.stock_selector import get_stock_pool_by_blogger

        # 获取当前股票池作为候选
        pool = get_stock_pool_by_blogger(blogger_name)
        candidate_codes = [s['code'] for s in pool[:20]]  # 最多20只

        engine = TradingDecisionEngine()
        report = engine.generate_decision_report(
            candidate_codes=candidate_codes,
            regime=None,  # 自动检测
        )
        engine.save_decision_report(report, blogger_name=blogger_name)

        return {
            "success": True,
            "report": {
                'report_date': report.date,
                'regime': report.regime,
                'strategy_name': report.strategy_name,
                'buy_signals': [{
                    'code': s.code, 'name': s.name,
                    'signal_type': s.signal_type,
                    'timing_score': s.timing_score,
                    'entry_price': s.entry_price,
                    'stop_loss_price': s.stop_loss_price,
                    'take_profit_price': s.take_profit_price,
                    'position_pct': s.position_pct,
                    'reasons': s.reasons,
                    'confidence': s.confidence,
                } for s in report.buy_signals],
                'sell_signals': [{
                    'code': s.code, 'name': s.name,
                    'signal_type': s.signal_type,
                    'urgency': s.urgency,
                    'reasons': s.reasons,
                } for s in report.sell_signals],
                'portfolio_actions': report.portfolio_actions,
                'risk_warnings': report.risk_warnings,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/factor-health")
async def api_factor_health():
    """获取因子健康度报告(IC衰减监控)"""
    try:
        from ..selector.ic_decay_monitor import get_latest_factor_health
        health = get_latest_factor_health()
        if not health:
            return {"success": True, "health": None, "message": "暂无因子健康度数据"}
        return {"success": True, "health": health}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/run-ic-analysis")
async def api_run_ic_analysis(weeks: int = Query(8)):
    """手动触发IC衰减分析"""
    try:
        from ..selector.ic_decay_monitor import ICDecayMonitor
        monitor = ICDecayMonitor()
        report = monitor.run_weekly_ic_analysis(weeks=weeks)

        return {
            "success": True,
            "report": {
                'report_date': report.report_date,
                'summary': report.summary,
                'factors': [{
                    'name': f.factor_name,
                    'current_ic': f.current_ic,
                    'ic_trend': f.ic_trend,
                    'health_status': f.health_status,
                    'consecutive_low_weeks': f.consecutive_low_weeks,
                    'consecutive_neg_weeks': f.consecutive_neg_weeks,
                } for f in report.factors],
                'weight_adjustments': report.weight_adjustments,
                'actions': report.actions_taken,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/strategy-profile")
async def api_strategy_profile():
    """获取当前自适应策略配置"""
    try:
        from ..selector.adaptive_strategy import AdaptiveStrategyManager, STRATEGY_MAP
        from ..selector.stock_selector import StockSelector
        selector = StockSelector()
        regime = selector._detect_market_regime()
        asm = AdaptiveStrategyManager()
        guidance = asm.get_position_guidance(regime)

        # 返回所有策略配置 + 当前激活策略
        profiles = {}
        for key, profile in STRATEGY_MAP.items():
            profiles[key] = {
                'name': profile.name,
                'stop_loss_pct': profile.stop_loss_pct,
                'take_profit_pct': profile.take_profit_pct,
                'trailing_stop_pct': profile.trailing_stop_pct,
                'max_total_position': profile.max_total_position,
                'single_position': profile.single_position,
                'max_positions': profile.max_positions,
                'min_hold_days': profile.min_hold_days,
                'max_hold_days': profile.max_hold_days,
                'description': profile.description,
            }

        return {
            "success": True,
            "current_regime": regime,
            "current_strategy": guidance,
            "all_strategies": profiles,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# ── 持仓跟踪API ──

@app.get("/api/tracked-positions")
async def api_tracked_positions(state: Optional[str] = Query(None)):
    """获取跟踪持仓列表"""
    try:
        from ..selector.position_tracker import PositionTracker, ALL_STATES
        tracker = PositionTracker()

        if state and state in ALL_STATES + ['expired']:
            if state in ('sold', 'expired'):
                positions = tracker.get_sold_positions(days=30)
                positions = [p for p in positions if p.state == state]
            else:
                positions = tracker.get_active_positions()
                positions = [p for p in positions if p.state == state]
        else:
            positions = tracker.get_active_positions()

        result = []
        for p in positions:
            result.append({
                'id': p.id,
                'code': p.code,
                'name': p.name,
                'sector': p.sector,
                'state': p.state,
                'recommend_date': p.recommend_date,
                'recommend_score': p.recommend_score,
                'recommend_source': p.recommend_source,
                'recommend_price': p.recommend_price,
                'entry_date': p.entry_date,
                'entry_price': p.entry_price,
                'current_price': p.current_price,
                'profit_pct': p.profit_pct,
                'hold_days': p.hold_days,
                'stop_loss_price': p.stop_loss_price,
                'take_profit_price': p.take_profit_price,
                'timing_score': p.timing_score,
                'sell_date': p.sell_date,
                'sell_price': p.sell_price,
                'sell_reason': p.sell_reason,
                'actual_return_pct': p.actual_return_pct,
            })

        summary = tracker.get_position_summary()

        return {
            "success": True,
            "positions": result,
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/tracked-positions/summary")
async def api_tracked_positions_summary():
    """获取持仓概览统计"""
    try:
        from ..selector.position_tracker import PositionTracker
        tracker = PositionTracker()
        summary = tracker.get_position_summary()
        return {"success": True, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/tracked-positions/sync")
async def api_sync_tracked_positions():
    """从股票池同步到跟踪系统"""
    try:
        from ..selector.position_tracker import PositionTracker
        tracker = PositionTracker()
        result = tracker.sync_from_stock_pool()
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/tracked-positions/update")
async def api_update_tracked_positions():
    """更新所有持仓状态（价格、止损止盈检查等）"""
    try:
        from ..selector.position_tracker import PositionTracker
        tracker = PositionTracker()
        result = tracker.update_all_positions()
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/tracked-positions/{code}/buy")
async def api_mark_bought(code: str,
                          entry_price: float = Body(..., embed=True),
                          stop_loss: float = Body(0, embed=True),
                          take_profit: float = Body(0, embed=True),
                          position_pct: float = Body(0, embed=True)):
    """标记股票为已买入"""
    try:
        from ..selector.position_tracker import PositionTracker
        tracker = PositionTracker()
        success = tracker.mark_bought(code, entry_price, stop_loss, take_profit, position_pct)
        if success:
            return {"success": True, "message": f"{code} 已标记为买入"}
        else:
            return {"success": False, "message": f"{code} 未找到待买入记录"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/tracked-positions/{code}/sell")
async def api_mark_sold(code: str,
                        sell_price: float = Body(..., embed=True),
                        reason: str = Body('', embed=True)):
    """标记股票为已卖出"""
    try:
        from ..selector.position_tracker import PositionTracker
        tracker = PositionTracker()
        success = tracker.mark_sold(code, sell_price, reason)
        if success:
            return {"success": True, "message": f"{code} 已标记为卖出"}
        else:
            return {"success": False, "message": f"{code} 未找到持仓记录"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# ── 推荐质量反馈API ──

@app.get("/api/quality-dashboard")
async def api_quality_dashboard(days: int = Query(30)):
    """获取推荐质量仪表盘"""
    try:
        from ..selector.recommendation_feedback import RecommendationFeedback
        fb = RecommendationFeedback()
        dashboard = fb.get_quality_dashboard()
        return {"success": True, **dashboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/quality-review")
async def api_run_quality_review(days: int = Body(30, embed=True)):
    """执行推荐质量回顾"""
    try:
        from ..selector.recommendation_feedback import RecommendationFeedback
        fb = RecommendationFeedback()
        report = fb.run_batch_review(days=days)
        return {
            "success": True,
            "system_health": report.system_health,
            "degradation_reasons": report.degradation_reasons,
            "total_reviews": report.total_reviews,
            "direction_accuracy": round(report.direction_accuracy, 3),
            "avg_quality_score": round(report.avg_quality_score, 1),
            "win_rate": round(report.win_rate, 3),
            "avg_magnitude_error": round(report.avg_magnitude_error, 2),
            "grade_distribution": report.grade_distribution,
            "source_stats": report.source_stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/quality-report/latest")
async def api_latest_quality_report():
    """获取最新质量报告"""
    try:
        from ..selector.recommendation_feedback import RecommendationFeedback
        fb = RecommendationFeedback()
        report = fb.get_latest_quality_report()
        if report:
            return {"success": True, "report": report}
        else:
            return {"success": False, "message": "暂无质量报告"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/risk/report")
async def api_risk_report(blogger_name: Optional[str] = Query(None)):
    """获取最新的组合风险报告（优先从缓存加载）"""
    try:
        if blogger_name is not None and blogger_name.strip() == '':
            blogger_name = None
        # 1. 先尝试从数据库加载已有报告
        from ..selector.stock_selector import get_risk_report_from_db
        cached = get_risk_report_from_db(blogger_name=blogger_name, days=7)
        if cached and len(cached) > 0:
            # 将DB扁平格式转换为前端期望的嵌套格式
            r = cached[0]
            report = {
                'portfolio_value': r['portfolio_value'],
                'calculation_time': r['report_date'],
                'risk_rating': r['risk_rating'],
                'overall_risk_score': r['risk_score'],
                'sharpe_ratio': r['sharpe_ratio'],
                'sortino_ratio': r['sortino_ratio'],
                'max_drawdown': r['max_drawdown'],
                'calmar_ratio': r['calmar_ratio'],
                'stress_test_results': r.get('stress_test', {}),
                'risk_contribution': r.get('risk_contribution', {}),
                'marginal_var': {},
                'portfolio_var': {
                    'var_value': r['var_value'],
                    'var_absolute': r['var_absolute'],
                    'cvar': r['cvar_value'],
                    'confidence_level': r['confidence_level'],
                    'time_horizon': 1,
                    'method': 'historical',
                    'risk_level': r['risk_level'],
                    'risk_score': r['risk_score'],
                    'mean_return': 0,
                    'std_return': 0,
                    'skewness': 0,
                    'kurtosis': 0,
                },
                'individual_risks': {},
            }
            return {"success": True, "report": report, "source": "cache"}

        # 2. 缓存中没有最近报告，实时生成
        from ..selector.stock_selector import generate_risk_report
        result = generate_risk_report(blogger_name=blogger_name)
        if result:
            return {"success": True, "report": result, "source": "generated"}
        return {"success": False, "message": "数据不足，无法生成风险报告"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/risk/history")
async def api_risk_history(
    blogger_name: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365)
):
    """获取历史风险报告"""
    try:
        from ..selector.stock_selector import get_risk_report_from_db
        reports = get_risk_report_from_db(blogger_name=blogger_name, days=days)
        return {"success": True, "reports": reports, "count": len(reports)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/api/risk/generate")
async def api_risk_generate(blogger_name: Optional[str] = Query(None)):
    """手动生成风险报告"""
    try:
        from ..selector.stock_selector import generate_risk_report
        result = generate_risk_report(blogger_name=blogger_name)
        if result:
            return {"success": True, "report": result}
        return {"success": False, "message": "股票池为空或数据不足"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/api/market-timing")
async def api_market_timing():
    """获取大盘择时判断"""
    try:
        from ..selector.market_timing import MarketTiming, TIMING_CONFIG
        engine = MarketTiming()
        result = engine.get_timing()
        return {
            "success": True,
            "timing": result['timing'],
            "label": result['config']['label'],
            "reason": result['reason'],
            "ma5": result['ma5'],
            "ma20": result['ma20'],
            "trend": result['trend'],
            "ma5_slope": result['ma5_slope'],
            "vol_ratio": result['vol_ratio'],
            "recent_5d_change": result['recent_5d_change'],
            "recent_20d_change": result['recent_20d_change'],
            "config": {
                "pool_ratio": result['config']['pool_ratio'],
                "max_position": result['config']['max_position'],
                "single_position": result['config']['single_position'],
                "allow_hot_stocks": result['config']['allow_hot_stocks'],
                "min_composite_score": result['config']['min_composite_score'],
            },
            "should_trade": engine.should_trade(),
            "max_position_pct": engine.get_max_position() * 100,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulated-trading/check-stops")
async def api_check_auto_stops():
    """检查模拟盘所有持仓的自动止损止盈条件"""
    try:
        from ..trading import SimulatedTrading
        sim = SimulatedTrading()
        actions = await sim.check_auto_stop_loss_profit()
        # 获取所有持仓风险状态
        positions_status = {}
        for key, pos in sim._positions.items():
            if pos.quantity > 0:
                positions_status[key] = {
                    'code': pos.stock_code,
                    'name': pos.stock_name,
                    'quantity': pos.quantity,
                    'avg_cost': pos.avg_cost,
                    'current_price': pos.current_price,
                    'pnl_pct': round(pos.profit_loss_pct, 2),
                    'risk_status': sim.get_position_risk_status(key),
                }
        return {
            "success": True,
            "actions": actions,
            "positions_status": positions_status,
            "total_actions": len(actions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
数据库操作模块
"""
import aiosqlite
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from .models import Blogger, WeiboPost, WeiboImage, WeiboPostWithImages, BloggerWithStats
from .config import get_config

DB_TIMEOUT = 30  # SQLite busy timeout in seconds


async def get_db_path() -> str:
    """获取数据库路径"""
    config = get_config()
    db_path = Path(config.storage.database)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)


def _ensure_columns(conn):
    """确保表包含新列（旧数据库迁移）"""
    # market 列
    for table in ('stock_pool', 'predictions'):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN market TEXT DEFAULT 'A'")
        except sqlite3.OperationalError:
            pass
    # ML预测相关列
    for col, default in [('confidence', '0.0'), ('confidence_interval', 'NULL'), ('prediction_source', "'rule_based'")]:
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} TEXT DEFAULT {default}")
        except sqlite3.OperationalError:
            pass
    # 多因子量化列
    factor_columns = [
        ('fundamental_score', 'REAL DEFAULT 0'),
        ('valuation_score', 'REAL DEFAULT 0'),
        ('capital_flow_score', 'REAL DEFAULT 0'),
        ('sentiment_score', 'REAL DEFAULT 0'),
        ('pe_percentile', 'REAL DEFAULT 0'),
        ('pb_percentile', 'REAL DEFAULT 0'),
        ('market_regime', "TEXT DEFAULT ''"),
        ('factor_detail', "TEXT DEFAULT '{}'"),
        # 产业链推理 + 认知差 + 轧空信号
        ('cognitive_gap_score', 'REAL DEFAULT 0'),
        ('supply_chain_score', 'REAL DEFAULT 0'),
        ('short_interest_ratio', 'REAL DEFAULT 0'),
        ('volume_surge_ratio', 'REAL DEFAULT 0'),
        ('industry_chain', "TEXT DEFAULT ''"),
    ]
    for col, dtype in factor_columns:
        try:
            conn.execute(f"ALTER TABLE stock_pool ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass


def get_sync_db(db_path: str = None) -> sqlite3.Connection:
    """获取同步数据库连接（WAL模式 + busy timeout）

    用于调度器中的选股、预测、复盘等同步写入场景。
    """
    if db_path is None:
        config = get_config()
        db_path = str(Path(config.storage.database))
    conn = sqlite3.connect(db_path, timeout=DB_TIMEOUT)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout={}".format(DB_TIMEOUT * 1000))
    _ensure_columns(conn)
    return conn


async def init_db():
    """初始化数据库表"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        # 启用WAL模式，允许并发读写
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(f"PRAGMA busy_timeout={DB_TIMEOUT * 1000}")

        # 博主表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bloggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                weibo_uid TEXT UNIQUE NOT NULL,
                nickname TEXT DEFAULT '',
                crawl_interval INTEGER DEFAULT 60,
                enabled INTEGER DEFAULT 1,
                last_crawl_time DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 微博内容表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS weibo_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blogger_id INTEGER NOT NULL,
                weibo_id TEXT UNIQUE NOT NULL,
                content TEXT,
                is_repost INTEGER DEFAULT 0,
                repost_content TEXT,
                repost_author TEXT,
                publish_time DATETIME,
                is_pinned INTEGER DEFAULT 0,
                crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (blogger_id) REFERENCES bloggers(id)
            )
        """)

        # 图片表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS weibo_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                original_url TEXT NOT NULL,
                local_path TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES weibo_posts(id)
            )
        """)

        # 创建索引
        await db.execute("CREATE INDEX IF NOT EXISTS idx_posts_blogger_id ON weibo_posts(blogger_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_posts_publish_time ON weibo_posts(publish_time)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_images_post_id ON weibo_images(post_id)")

        # Cookie存储表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # P1: 预警存储表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                level TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                title TEXT DEFAULT '',
                message TEXT DEFAULT '',
                detail TEXT DEFAULT '{}',
                stock_code TEXT DEFAULT '',
                stock_name TEXT DEFAULT '',
                blogger_id TEXT DEFAULT '',
                current_value REAL DEFAULT 0,
                threshold_value REAL DEFAULT 0,
                change_pct REAL DEFAULT 0,
                trigger_time DATETIME NOT NULL,
                expire_time DATETIME,
                notified_channels TEXT DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_trigger_time ON alerts(trigger_time)")

        # ── 多市场支持：为stock_pool和predictions添加market字段 ──
        # stock_pool表（如果不存在则创建，含market字段）
        await db.execute("""
            CREATE TABLE IF NOT EXISTS stock_pool (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                market TEXT DEFAULT 'A',
                sector TEXT DEFAULT '',
                concept TEXT DEFAULT '',
                price REAL DEFAULT 0,
                change_pct REAL DEFAULT 0,
                style_match_score REAL DEFAULT 0,
                selection_source TEXT DEFAULT '市场热点',
                position_weight REAL DEFAULT 0,
                selection_reasons TEXT DEFAULT '[]',
                technical_indicators TEXT DEFAULT '{}',
                added_date TEXT DEFAULT '',
                blogger_name TEXT DEFAULT '',
                fundamental_score REAL DEFAULT 0,
                valuation_score REAL DEFAULT 0,
                capital_flow_score REAL DEFAULT 0,
                sentiment_score REAL DEFAULT 0,
                pe_percentile REAL DEFAULT 0,
                pb_percentile REAL DEFAULT 0,
                market_regime TEXT DEFAULT '',
                factor_detail TEXT DEFAULT '{}',
                cognitive_gap_score REAL DEFAULT 0,
                supply_chain_score REAL DEFAULT 0,
                short_interest_ratio REAL DEFAULT 0,
                volume_surge_ratio REAL DEFAULT 0,
                industry_chain TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_stock_pool_market ON stock_pool(market)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_stock_pool_status ON stock_pool(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_stock_pool_blogger ON stock_pool(blogger_name)")

        # predictions表（含market字段）
        await db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                market TEXT DEFAULT 'A',
                prediction_date TEXT NOT NULL,
                prediction_time TEXT DEFAULT '',
                current_price REAL DEFAULT 0,
                expected_close_change REAL DEFAULT 0,
                actual_change REAL,
                expected_amplitude REAL DEFAULT 0,
                expected_trend TEXT DEFAULT '',
                expected_hold_days INTEGER DEFAULT 0,
                expected_hold_return REAL DEFAULT 0,
                expected_buy_point REAL,
                expected_sell_point REAL,
                expected_sell_price REAL,
                selection_reason TEXT DEFAULT '',
                risk_warning TEXT DEFAULT '',
                blogger_name TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_market ON predictions(market)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)")

        # ── 交易模块表 ──
        # 交易账户表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trading_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                broker TEXT DEFAULT '',
                market TEXT DEFAULT 'A',
                mode TEXT DEFAULT 'simulated',
                api_key TEXT DEFAULT '',
                api_secret TEXT DEFAULT '',
                commission_rate REAL DEFAULT 0.0003,
                slippage REAL DEFAULT 0.001,
                initial_capital REAL DEFAULT 1000000,
                available_capital REAL DEFAULT 1000000,
                total_assets REAL DEFAULT 1000000,
                enabled INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 交易订单表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trading_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                order_id TEXT UNIQUE NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                market TEXT DEFAULT 'A',
                direction TEXT NOT NULL,
                order_type TEXT DEFAULT 'limit',
                price REAL DEFAULT 0,
                quantity INTEGER DEFAULT 0,
                filled_quantity INTEGER DEFAULT 0,
                filled_price REAL DEFAULT 0,
                filled_amount REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                source TEXT DEFAULT 'manual',
                reason TEXT DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES trading_accounts(id)
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_orders_account ON trading_orders(account_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON trading_orders(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_orders_stock ON trading_orders(stock_code)")

        # 持仓记录表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trading_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                market TEXT DEFAULT 'A',
                quantity INTEGER DEFAULT 0,
                available_quantity INTEGER DEFAULT 0,
                avg_cost REAL DEFAULT 0,
                current_price REAL DEFAULT 0,
                market_value REAL DEFAULT 0,
                profit_loss REAL DEFAULT 0,
                profit_loss_pct REAL DEFAULT 0,
                stop_profit_price REAL,
                stop_loss_price REAL,
                first_buy_date TEXT,
                last_update_date TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES trading_accounts(id)
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_positions_account ON trading_positions(account_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_positions_stock ON trading_positions(stock_code)")

        # ── 为已存在的stock_pool/predictions表添加market列（兼容旧数据） ──
        try:
            await db.execute("ALTER TABLE stock_pool ADD COLUMN market TEXT DEFAULT 'A'")
        except Exception:
            pass  # 列已存在
        try:
            await db.execute("ALTER TABLE predictions ADD COLUMN market TEXT DEFAULT 'A'")
        except Exception:
            pass  # 列已存在

        await db.commit()


# ============ Cookie相关操作 ============

async def save_cookie(cookie: str) -> None:
    """保存Cookie到数据库"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT OR REPLACE INTO settings (key, value, updated_at)
               VALUES ('weibo_cookie', ?, ?)""",
            (cookie, datetime.now())
        )
        await db.commit()


async def load_cookie() -> Optional[str]:
    """从数据库加载Cookie"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT value FROM settings WHERE key='weibo_cookie'"
        )
        row = await cursor.fetchone()
        return row[0] if row else None


async def delete_cookie() -> None:
    """删除Cookie"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM settings WHERE key='weibo_cookie'")
        await db.commit()


# ============ 博主相关操作 ============

async def add_blogger(blogger: Blogger) -> int:
    """添加博主"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """INSERT INTO bloggers (weibo_uid, nickname, crawl_interval, enabled)
               VALUES (?, ?, ?, ?)""",
            (blogger.weibo_uid, blogger.nickname, blogger.crawl_interval, blogger.enabled)
        )
        await db.commit()
        return cursor.lastrowid


async def update_blogger(blogger: Blogger) -> bool:
    """更新博主信息"""
    if blogger.id is None:
        return False
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """UPDATE bloggers SET nickname=?, crawl_interval=?, enabled=?
               WHERE id=?""",
            (blogger.nickname, blogger.crawl_interval, blogger.enabled, blogger.id)
        )
        await db.commit()
        return True


async def delete_blogger(blogger_id: int) -> bool:
    """删除博主"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM bloggers WHERE id=?", (blogger_id,))
        await db.commit()
        return True


async def get_blogger(blogger_id: int) -> Optional[Blogger]:
    """获取单个博主"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM bloggers WHERE id=?", (blogger_id,))
        row = await cursor.fetchone()
        if row:
            return Blogger(
                id=row['id'],
                weibo_uid=row['weibo_uid'],
                nickname=row['nickname'],
                crawl_interval=row['crawl_interval'],
                enabled=bool(row['enabled']),
                last_crawl_time=row['last_crawl_time'],
                created_at=row['created_at']
            )
        return None


async def get_blogger_by_uid(uid: str) -> Optional[Blogger]:
    """根据微博UID获取博主"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM bloggers WHERE weibo_uid=?", (uid,))
        row = await cursor.fetchone()
        if row:
            return Blogger(
                id=row['id'],
                weibo_uid=row['weibo_uid'],
                nickname=row['nickname'],
                crawl_interval=row['crawl_interval'],
                enabled=bool(row['enabled']),
                last_crawl_time=row['last_crawl_time'],
                created_at=row['created_at']
            )
        return None


async def get_all_bloggers(enabled_only: bool = False) -> List[Blogger]:
    """获取所有博主"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if enabled_only:
            cursor = await db.execute("SELECT * FROM bloggers WHERE enabled=1 ORDER BY id")
        else:
            cursor = await db.execute("SELECT * FROM bloggers ORDER BY id")
        rows = await cursor.fetchall()
        return [
            Blogger(
                id=row['id'],
                weibo_uid=row['weibo_uid'],
                nickname=row['nickname'],
                crawl_interval=row['crawl_interval'],
                enabled=bool(row['enabled']),
                last_crawl_time=row['last_crawl_time'],
                created_at=row['created_at']
            )
            for row in rows
        ]


async def update_blogger_crawl_time(blogger_id: int) -> None:
    """更新博主最后采集时间"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE bloggers SET last_crawl_time=? WHERE id=?",
            (datetime.now(), blogger_id)
        )
        await db.commit()


async def get_bloggers_with_stats() -> List[BloggerWithStats]:
    """获取博主列表(含统计信息)"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT b.*,
                   COUNT(p.id) as post_count,
                   MAX(p.publish_time) as last_post_time
            FROM bloggers b
            LEFT JOIN weibo_posts p ON b.id = p.blogger_id
            GROUP BY b.id
            ORDER BY b.id
        """)
        rows = await cursor.fetchall()
        return [
            BloggerWithStats(
                blogger=Blogger(
                    id=row['id'],
                    weibo_uid=row['weibo_uid'],
                    nickname=row['nickname'],
                    crawl_interval=row['crawl_interval'],
                    enabled=bool(row['enabled']),
                    last_crawl_time=row['last_crawl_time'],
                    created_at=row['created_at']
                ),
                post_count=row['post_count'],
                last_post_time=row['last_post_time']
            )
            for row in rows
        ]


# ============ 微博内容相关操作 ============

async def add_weibo_post(post: WeiboPost) -> Optional[int]:
    """添加微博内容，如果已存在则返回None"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        try:
            cursor = await db.execute(
                """INSERT INTO weibo_posts
                   (blogger_id, weibo_id, content, is_repost, repost_content, repost_author, publish_time, is_pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (post.blogger_id, post.weibo_id, post.content, post.is_repost,
                 post.repost_content, post.repost_author, post.publish_time, post.is_pinned)
            )
            await db.commit()
            return cursor.lastrowid
        except aiosqlite.IntegrityError:
            return None


async def weibo_post_exists(weibo_id: str) -> bool:
    """检查微博是否已存在"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT 1 FROM weibo_posts WHERE weibo_id=?", (weibo_id,))
        return await cursor.fetchone() is not None


async def get_weibo_posts(blogger_id: int, limit: int = 50, offset: int = 0) -> List[WeiboPostWithImages]:
    """获取博主的微博列表(含图片)"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row

        # 获取微博列表
        cursor = await db.execute(
            """SELECT * FROM weibo_posts WHERE blogger_id=?
               ORDER BY is_pinned DESC, publish_time DESC
               LIMIT ? OFFSET ?""",
            (blogger_id, limit, offset)
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            post = WeiboPost(
                id=row['id'],
                blogger_id=row['blogger_id'],
                weibo_id=row['weibo_id'],
                content=row['content'],
                is_repost=bool(row['is_repost']),
                repost_content=row['repost_content'] or "",
                repost_author=row['repost_author'] or "",
                publish_time=row['publish_time'],
                is_pinned=bool(row['is_pinned']),
                crawl_time=row['crawl_time']
            )

            # 获取图片
            img_cursor = await db.execute(
                "SELECT * FROM weibo_images WHERE post_id=?", (post.id,)
            )
            img_rows = await img_cursor.fetchall()
            images = [
                WeiboImage(
                    id=img['id'],
                    post_id=img['post_id'],
                    original_url=img['original_url'],
                    local_path=img['local_path'],
                    created_at=img['created_at']
                )
                for img in img_rows
            ]

            results.append(WeiboPostWithImages(post=post, images=images))

        return results


async def get_weibo_posts_count(blogger_id: int) -> int:
    """获取博主的微博数量"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM weibo_posts WHERE blogger_id=?", (blogger_id,))
        result = await cursor.fetchone()
        return result[0] if result else 0


# ============ 图片相关操作 ============

async def add_weibo_image(image: WeiboImage) -> int:
    """添加微博图片"""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO weibo_images (post_id, original_url, local_path) VALUES (?, ?, ?)",
            (image.post_id, image.original_url, image.local_path)
        )
        await db.commit()
        return cursor.lastrowid


# ============ 搜索相关操作 ============

async def search_weibo_posts(keyword: str, blogger_id: Optional[int] = None,
                              limit: int = 50, offset: int = 0) -> List[WeiboPostWithImages]:
    """搜索微博内容

    Args:
        keyword: 搜索关键词
        blogger_id: 指定博主ID，为None时搜索所有博主
        limit: 返回数量限制
        offset: 偏移量

    Returns:
        List[WeiboPostWithImages]: 匹配的微博列表
    """
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row

        # 构建搜索SQL
        search_pattern = f"%{keyword}%"

        if blogger_id:
            cursor = await db.execute(
                """SELECT * FROM weibo_posts
                   WHERE blogger_id=? AND (content LIKE ? OR repost_content LIKE ?)
                   ORDER BY is_pinned DESC, publish_time DESC
                   LIMIT ? OFFSET ?""",
                (blogger_id, search_pattern, search_pattern, limit, offset)
            )
        else:
            cursor = await db.execute(
                """SELECT * FROM weibo_posts
                   WHERE content LIKE ? OR repost_content LIKE ?
                   ORDER BY is_pinned DESC, publish_time DESC
                   LIMIT ? OFFSET ?""",
                (search_pattern, search_pattern, limit, offset)
            )

        rows = await cursor.fetchall()

        results = []
        for row in rows:
            post = WeiboPost(
                id=row['id'],
                blogger_id=row['blogger_id'],
                weibo_id=row['weibo_id'],
                content=row['content'],
                is_repost=bool(row['is_repost']),
                repost_content=row['repost_content'] or "",
                repost_author=row['repost_author'] or "",
                publish_time=row['publish_time'],
                is_pinned=bool(row['is_pinned']),
                crawl_time=row['crawl_time']
            )

            # 获取图片
            img_cursor = await db.execute(
                "SELECT * FROM weibo_images WHERE post_id=?", (post.id,)
            )
            img_rows = await img_cursor.fetchall()
            images = [
                WeiboImage(
                    id=img['id'],
                    post_id=img['post_id'],
                    original_url=img['original_url'],
                    local_path=img['local_path'],
                    created_at=img['created_at']
                )
                for img in img_rows
            ]

            results.append(WeiboPostWithImages(post=post, images=images))

        return results


async def search_weibo_posts_count(keyword: str, blogger_id: Optional[int] = None) -> int:
    """搜索微博数量

    Args:
        keyword: 搜索关键词
        blogger_id: 指定博主ID，为None时搜索所有博主

    Returns:
        int: 匹配的微博数量
    """
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        search_pattern = f"%{keyword}%"

        if blogger_id:
            cursor = await db.execute(
                """SELECT COUNT(*) FROM weibo_posts
                   WHERE blogger_id=? AND (content LIKE ? OR repost_content LIKE ?)""",
                (blogger_id, search_pattern, search_pattern)
            )
        else:
            cursor = await db.execute(
                """SELECT COUNT(*) FROM weibo_posts
                   WHERE content LIKE ? OR repost_content LIKE ?""",
                (search_pattern, search_pattern)
            )

        result = await cursor.fetchone()
        return result[0] if result else 0


async def search_weibo_posts_by_date(start_date: datetime, end_date: datetime,
                                       blogger_id: Optional[int] = None,
                                       limit: int = 50, offset: int = 0) -> List[WeiboPostWithImages]:
    """按日期范围搜索微博

    Args:
        start_date: 开始日期
        end_date: 结束日期
        blogger_id: 指定博主ID，为None时搜索所有博主
        limit: 返回数量限制
        offset: 偏移量

    Returns:
        List[WeiboPostWithImages]: 匹配的微博列表
    """
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row

        if blogger_id:
            cursor = await db.execute(
                """SELECT * FROM weibo_posts
                   WHERE blogger_id=? AND publish_time BETWEEN ? AND ?
                   ORDER BY is_pinned DESC, publish_time DESC
                   LIMIT ? OFFSET ?""",
                (blogger_id, start_date, end_date, limit, offset)
            )
        else:
            cursor = await db.execute(
                """SELECT * FROM weibo_posts
                   WHERE publish_time BETWEEN ? AND ?
                   ORDER BY is_pinned DESC, publish_time DESC
                   LIMIT ? OFFSET ?""",
                (start_date, end_date, limit, offset)
            )

        rows = await cursor.fetchall()

        results = []
        for row in rows:
            post = WeiboPost(
                id=row['id'],
                blogger_id=row['blogger_id'],
                weibo_id=row['weibo_id'],
                content=row['content'],
                is_repost=bool(row['is_repost']),
                repost_content=row['repost_content'] or "",
                repost_author=row['repost_author'] or "",
                publish_time=row['publish_time'],
                is_pinned=bool(row['is_pinned']),
                crawl_time=row['crawl_time']
            )

            # 获取图片
            img_cursor = await db.execute(
                "SELECT * FROM weibo_images WHERE post_id=?", (post.id,)
            )
            img_rows = await img_cursor.fetchall()
            images = [
                WeiboImage(
                    id=img['id'],
                    post_id=img['post_id'],
                    original_url=img['original_url'],
                    local_path=img['local_path'],
                    created_at=img['created_at']
                )
                for img in img_rows
            ]

            results.append(WeiboPostWithImages(post=post, images=images))

        return results

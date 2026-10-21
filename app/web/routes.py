"""
Web路由模块 - 仅微博抓取功能
"""
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, Form, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import get_config
from ..database import (
    init_db, get_all_bloggers, get_bloggers_with_stats,
    get_blogger, get_blogger_by_uid, add_blogger, update_blogger, delete_blogger,
    get_weibo_posts, get_weibo_posts_count,
    search_weibo_posts, search_weibo_posts_count,
    Blogger, WeiboPost
)
from ..scheduler import get_scheduler, init_scheduler
from ..crawler.weibo import get_crawler, set_global_cookie, get_global_cookie, clear_global_cookie
from ..database import save_cookie as db_save_cookie, load_cookie as db_load_cookie, delete_cookie as db_delete_cookie


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

"""
微博爬虫核心模块
"""
import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Set
import httpx
from ..config import get_config
from ..database import (
    add_weibo_post, add_weibo_image, weibo_post_exists,
    update_blogger_crawl_time, get_blogger_by_uid, get_weibo_posts_count
)
from ..models import WeiboPost, WeiboImage, Blogger
from .parser import parse_weibo_json, ParsedWeibo, parse_weibo_time
from .image import download_images


# 全局Cookie存储
_global_cookie: Optional[str] = None


def set_global_cookie(cookie: str):
    """设置全局Cookie"""
    global _global_cookie
    _global_cookie = cookie


def get_global_cookie() -> Optional[str]:
    """获取全局Cookie"""
    return _global_cookie


def clear_global_cookie():
    """清除全局Cookie"""
    global _global_cookie
    _global_cookie = None


class WeiboCrawler:
    """微博爬虫"""

    def __init__(self):
        self.config = get_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._cookie: Optional[str] = None

    def set_cookie(self, cookie: str):
        """设置Cookie"""
        self._cookie = cookie

    def clear_cookie(self):
        """清除Cookie"""
        self._cookie = None

    def _get_effective_cookie(self) -> Optional[str]:
        """获取有效的Cookie（优先使用实例Cookie，其次使用全局Cookie）"""
        if self._cookie:
            return self._cookie
        return _global_cookie

    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.crawler.timeout,
                follow_redirects=True
            )
        return self._client

    async def close(self):
        """关闭HTTP客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_weibo_detail(self, weibo_id: str) -> Optional[dict]:
        """获取单条微博的完整内容（用于长文本）"""
        client = await self._get_client()

        # 使用带 isGetLongText 参数的API
        url = f"https://weibo.com/ajax/statuses/show"
        params = {
            'id': weibo_id,
            'isGetLongText': 1
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': f'https://weibo.com/',
        }

        effective_cookie = self._get_effective_cookie()
        if effective_cookie:
            headers['Cookie'] = effective_cookie

        try:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get('id') or data.get('ok') == 1:
                    return data
            return None
        except Exception as e:
            print(f"获取微博详情失败: {e}")
            return None

    async def fetch_user_weibo(self, uid: str, page: int = 1) -> Optional[dict]:
        """获取用户微博列表 - 使用PC端API"""
        client = await self._get_client()

        # 使用PC端API，更稳定
        url = "https://weibo.com/ajax/statuses/mymblog"
        params = {
            'uid': uid,
            'page': page,
            'feature': 0
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': f'https://weibo.com/u/{uid}',
        }

        # 添加Cookie
        effective_cookie = self._get_effective_cookie()
        if effective_cookie:
            headers['Cookie'] = effective_cookie
            print(f"使用Cookie进行请求 (长度: {len(effective_cookie)})")

        try:
            response = await client.get(url, params=params, headers=headers)
            print(f"请求微博列表，状态码: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    ok_code = data.get('ok', 0)
                    if ok_code == 1:
                        return data
                    elif ok_code == -100:
                        print("错误: Cookie无效或已过期，请重新登录微博获取新Cookie")
                        return {'ok': -100, 'msg': 'Cookie无效或已过期'}
                    else:
                        print(f"API返回错误: {data}")
                        return None
                except json.JSONDecodeError:
                    print(f"JSON解析失败，响应内容: {response.text[:500]}")
                    return None
            else:
                print(f"请求失败，状态码: {response.status_code}")
                return None

        except httpx.RequestError as e:
            print(f"请求异常: {e}")
            return None

    async def parse_weibo_list(self, data: dict) -> List[ParsedWeibo]:
        """解析微博列表 - 支持PC端API格式"""
        results = []

        try:
            # PC端API格式: data.list
            weibo_list = data.get('data', {}).get('list', [])

            if weibo_list:
                print(f"PC端API返回 {len(weibo_list)} 条微博")
                for item in weibo_list:
                    parsed = self._parse_pc_weibo(item)
                    if parsed:
                        results.append(parsed)
            else:
                # 兼容移动端API格式
                cards = data.get('data', {}).get('cards', [])
                print(f"移动端API返回 {len(cards)} 个卡片")

                for card in cards:
                    card_type = card.get('card_type', 0)
                    if card_type == 9:
                        mblog = card.get('mblog', {})
                        if mblog:
                            parsed = parse_weibo_json(mblog)
                            if parsed:
                                results.append(parsed)

                    card_group = card.get('card_group', [])
                    if card_group:
                        for item in card_group:
                            if item.get('card_type') == 9:
                                mblog = item.get('mblog', {})
                                if mblog:
                                    parsed = parse_weibo_json(mblog)
                                    if parsed:
                                        results.append(parsed)

            print(f"解析到 {len(results)} 条微博")

        except Exception as e:
            print(f"解析微博列表失败: {e}")
            import traceback
            traceback.print_exc()

        return results

    def _parse_pc_weibo(self, item: dict) -> Optional[ParsedWeibo]:
        """解析PC端API返回的微博数据"""
        try:
            weibo_id = str(item.get('id', ''))
            if not weibo_id:
                return None

            # 内容 - 优先使用text_raw获取完整内容
            content = item.get('text_raw', '') or item.get('text', '')
            content = re.sub(r'<[^>]+>', '', content).strip()

            # 如果内容被截断（包含...展开），尝试获取完整内容
            if '...展开' in content or '展开全文' in content:
                # 检查是否有isLongText标记
                if item.get('isLongText') or item.get('is_long_text'):
                    # 尝试从longText获取
                    long_text = item.get('longText', {}).get('longTextContent', '')
                    if long_text:
                        content = long_text
                    # 或者直接使用完整字段
                    full_text = item.get('text_full', '') or item.get('full_text', '')
                    if full_text:
                        content = re.sub(r'<[^>]+>', '', full_text).strip()

            # 发布时间
            created_at = item.get('created_at', '')
            publish_time = parse_weibo_time(created_at)

            # 是否置顶
            is_pinned = item.get('isTop', 0) == 1 or item.get('is_top', 0) == 1

            # 图片
            image_urls = []
            pics = item.get('pic_ids', [])
            if pics:
                # PC端返回的是图片ID，需要构建URL
                for pic_id in pics:
                    if pic_id:
                        # 尝试从pic_infos获取
                        pic_info = item.get('pic_infos', {}).get(pic_id, {})
                        url = pic_info.get('large', {}).get('url') or pic_info.get('url')
                        if url:
                            image_urls.append(url)

            # 也检查pic_num和原始图片URL
            if not image_urls and item.get('pic_num', 0) > 0:
                # 从page_info中获取
                page_info = item.get('page_info', {})
                if page_info.get('type') == 'pic':
                    pics_data = page_info.get('pics', [])
                    for pic in pics_data:
                        url = pic.get('large', {}).get('url') or pic.get('url')
                        if url:
                            image_urls.append(url)

            # 转发内容
            is_repost = False
            repost_content = ""
            repost_author = ""

            retweeted_status = item.get('retweeted_status')
            if retweeted_status:
                is_repost = True
                repost_user = retweeted_status.get('user', {})
                repost_author = repost_user.get('screen_name', '') if repost_user else ''
                repost_text = retweeted_status.get('text', '')
                repost_content = re.sub(r'<[^>]+>', '', repost_text).strip()

                # 转发微博的图片
                retweeted_pics = retweeted_status.get('pic_ids', [])
                retweeted_pic_infos = retweeted_status.get('pic_infos', {})
                for pic_id in retweeted_pics:
                    pic_info = retweeted_pic_infos.get(pic_id, {})
                    url = pic_info.get('large', {}).get('url') or pic_info.get('url')
                    if url:
                        image_urls.append(url)

            return ParsedWeibo(
                weibo_id=weibo_id,
                content=content,
                is_repost=is_repost,
                repost_content=repost_content,
                repost_author=repost_author,
                publish_time=publish_time,
                is_pinned=is_pinned,
                image_urls=image_urls
            )

        except Exception as e:
            print(f"解析PC端微博失败: {e}")
            return None

    def is_within_days(self, publish_time: Optional[datetime], days: int) -> bool:
        """检查发布时间是否在指定天数内"""
        if not publish_time:
            return True  # 没有时间信息时默认采集
        cutoff = datetime.now() - timedelta(days=days)
        return publish_time >= cutoff

    async def crawl_blogger(self, blogger: Blogger) -> int:
        """爬取单个博主，返回新增微博数量"""
        print(f"开始爬取博主: {blogger.nickname or blogger.weibo_uid}")

        # 检查Cookie是否设置
        if not self._get_effective_cookie():
            raise Exception("未设置Cookie，请先在设置中配置微博Cookie")

        # 检查是否是第一次采集（该博主还没有任何微博）
        existing_count = await get_weibo_posts_count(blogger.id)
        is_first_crawl = existing_count == 0

        if is_first_crawl:
            print(f"首次采集博主 {blogger.nickname or blogger.weibo_uid}，将采集近7天的微博")
            max_pages = 20  # 首次采集更多页
            days_limit = 7   # 限制7天内的微博
        else:
            max_pages = 5   # 常规采集
            days_limit = None  # 不限制天数

        new_count = 0
        page = 1
        seen_ids: Set[str] = set()  # 本次采集已处理的微博ID（避免置顶微博重复）
        pinned_ids: Set[str] = set()  # 置顶微博ID
        consecutive_empty = 0  # 连续空页计数

        while page <= max_pages:
            data = await self.fetch_user_weibo(blogger.weibo_uid, page)

            if not data:
                print(f"获取数据失败，跳过第{page}页")
                page += 1
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    print("连续多次获取失败，停止爬取")
                    break
                await asyncio.sleep(self.config.crawler.request_delay)
                continue

            # 检查API返回 - Cookie无效错误
            if data.get('ok') == -100:
                raise Exception("Cookie无效或已过期，请重新登录微博获取新Cookie")

            # 检查API返回 - 其他错误
            if data.get('ok') == 0:
                msg = data.get('msg', '')
                print(f"API返回错误: {msg}")
                if '需要登录' in msg or '432' in msg:
                    raise Exception("Cookie无效或已过期，请重新设置")
                break

            weibo_list = await self.parse_weibo_list(data)

            if not weibo_list:
                print(f"第{page}页没有微博数据")
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    print("连续没有数据，停止爬取")
                    break
                page += 1
                await asyncio.sleep(self.config.crawler.request_delay)
                continue

            consecutive_empty = 0
            page_new_count = 0
            page_skipped = 0

            for parsed in weibo_list:
                # 跳过本次采集已处理的微博（置顶微博会出现在每一页）
                if parsed.weibo_id in seen_ids:
                    continue

                seen_ids.add(parsed.weibo_id)

                # 记录置顶微博ID
                if parsed.is_pinned:
                    pinned_ids.add(parsed.weibo_id)

                # 检查是否已存在于数据库
                if await weibo_post_exists(parsed.weibo_id):
                    page_skipped += 1
                    continue

                # 首次采集时检查时间限制
                if is_first_crawl and days_limit:
                    if not self.is_within_days(parsed.publish_time, days_limit):
                        print(f"微博 {parsed.weibo_id} 超过{days_limit}天，跳过")
                        continue

                # 检查是否需要获取完整内容（长文本被截断）
                content = parsed.content
                # 检测截断标记
                is_truncated = False
                if '...展开' in content or '展开全文' in content:
                    is_truncated = True
                # 零宽空格字符 (U+200B) 通常表示内容被截断
                if '​' in content and len(content) < 300:
                    is_truncated = True
                # 内容以序号开头但很短，可能被截断（如 "5、" 后面没有内容）
                if re.search(r'[1-5]、[^1-9]*$', content) and len(content) < 200:
                    is_truncated = True

                if is_truncated:
                    print(f"检测到长文本微博 {parsed.weibo_id}，获取完整内容...")
                    detail = await self.fetch_weibo_detail(parsed.weibo_id)
                    if detail:
                        # 优先从 longText.content 获取完整内容
                        long_text = detail.get('longText', {})
                        if isinstance(long_text, dict):
                            full_content = long_text.get('content', '')
                        else:
                            full_content = ''

                        # 备选方案：text_raw 或 text
                        if not full_content:
                            full_content = detail.get('text_raw', '') or detail.get('text', '')

                        if full_content:
                            content = re.sub(r'<[^>]+>', '', full_content).strip()
                            print(f"成功获取完整内容，长度: {len(content)}")

                # 创建微博记录
                post = WeiboPost(
                    blogger_id=blogger.id,
                    weibo_id=parsed.weibo_id,
                    content=content,
                    is_repost=parsed.is_repost,
                    repost_content=parsed.repost_content,
                    repost_author=parsed.repost_author,
                    publish_time=parsed.publish_time,
                    is_pinned=parsed.is_pinned
                )

                # 保存微博
                post_id = await add_weibo_post(post)

                if post_id:
                    page_new_count += 1

                    # 下载图片
                    if parsed.image_urls:
                        try:
                            local_paths = await download_images(parsed.image_urls, parsed.weibo_id)

                            for original_url, local_path in zip(parsed.image_urls, local_paths):
                                image = WeiboImage(
                                    post_id=post_id,
                                    original_url=original_url,
                                    local_path=local_path
                                )
                                await add_weibo_image(image)
                        except Exception as e:
                            print(f"下载图片失败: {e}")

            new_count += page_new_count
            print(f"第{page}页完成，新增{page_new_count}条微博，跳过{page_skipped}条已存在")

            # 如果连续两页没有新微博，停止
            if page_new_count == 0 and page > 1:
                print("本页无新微博，停止爬取")
                break

            page += 1
            await asyncio.sleep(self.config.crawler.request_delay)

        # 更新爬取时间
        if blogger.id:
            await update_blogger_crawl_time(blogger.id)

        print(f"博主 {blogger.nickname or blogger.weibo_uid} 爬取完成")
        print(f"  - 新增微博: {new_count} 条")
        print(f"  - 置顶微博: {len(pinned_ids)} 条")
        return new_count

    async def crawl_all_bloggers(self) -> dict:
        """爬取所有启用的博主"""
        from ..database import get_all_bloggers

        bloggers = await get_all_bloggers(enabled_only=True)
        results = {}

        for blogger in bloggers:
            try:
                new_count = await self.crawl_blogger(blogger)
                results[blogger.weibo_uid] = {
                    'success': True,
                    'new_count': new_count
                }
            except Exception as e:
                print(f"爬取博主 {blogger.weibo_uid} 失败: {e}")
                import traceback
                traceback.print_exc()
                results[blogger.weibo_uid] = {
                    'success': False,
                    'error': str(e)
                }

        await self.close()
        return results


# 全局爬虫实例
_crawler: Optional[WeiboCrawler] = None


async def get_crawler() -> WeiboCrawler:
    """获取爬虫实例"""
    global _crawler
    if _crawler is None:
        _crawler = WeiboCrawler()
    return _crawler


async def crawl_blogger_task(blogger_id: int) -> int:
    """定时任务：爬取单个博主"""
    from ..database import get_blogger

    blogger = await get_blogger(blogger_id)
    if not blogger or not blogger.enabled:
        return 0

    crawler = await get_crawler()
    return await crawler.crawl_blogger(blogger)
"""
微博页面解析模块
"""
import re
import json
from datetime import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParsedWeibo:
    """解析后的微博数据"""
    weibo_id: str
    content: str = ""
    is_repost: bool = False
    repost_content: str = ""
    repost_author: str = ""
    publish_time: Optional[datetime] = None
    is_pinned: bool = False
    image_urls: List[str] = None

    def __post_init__(self):
        if self.image_urls is None:
            self.image_urls = []


def parse_weibo_time(time_str: str) -> Optional[datetime]:
    """解析微博时间字符串"""
    if not time_str:
        return None

    now = datetime.now()

    # 微博API返回的英文格式: "Sat May 08 10:30:00 +0800 2021"
    # 或简短格式: "May 08"
    try:
        # 尝试解析完整的英文格式
        if len(time_str) > 20:
            # 移除时区信息后解析
            time_str_clean = re.sub(r'\s*[+-]\d{4}\s*', ' ', time_str)
            # 尝试多种格式
            formats = [
                '%a %b %d %H:%M:%S %Y',
                '%a %b %d %H:%M %Y',
                '%b %d %H:%M:%S %Y',
                '%b %d %H:%M %Y',
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(time_str_clean.strip(), fmt)
                except ValueError:
                    continue
    except Exception:
        pass

    # 处理相对时间
    if "分钟前" in time_str:
        match = re.search(r'(\d+)分钟前', time_str)
        if match:
            minutes = int(match.group(1))
            return datetime.fromtimestamp(now.timestamp() - minutes * 60)

    if "小时前" in time_str:
        match = re.search(r'(\d+)小时前', time_str)
        if match:
            hours = int(match.group(1))
            return datetime.fromtimestamp(now.timestamp() - hours * 3600)

    if "今天" in time_str:
        match = re.search(r'今天\s*(\d{1,2}):(\d{2})', time_str)
        if match:
            hour, minute = int(match.group(1)), int(match.group(2))
            return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if "昨天" in time_str:
        match = re.search(r'昨天\s*(\d{1,2}):(\d{2})', time_str)
        if match:
            hour, minute = int(match.group(1)), int(match.group(2))
            yesterday = datetime.fromtimestamp(now.timestamp() - 86400)
            return yesterday.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # 处理 "MM月DD日 HH:MM" 格式
    match = re.search(r'(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})', time_str)
    if match:
        month, day, hour, minute = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        year = now.year
        # 如果月份大于当前月份，说明是去年的
        if month > now.month:
            year -= 1
        try:
            return datetime(year, month, day, hour, minute)
        except ValueError:
            pass

    # 处理 "YYYY-MM-DD HH:MM" 格式
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})\s*(\d{1,2}):(\d{2})', time_str)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)),
                          int(match.group(4)), int(match.group(5)))
        except ValueError:
            pass

    return None


def parse_weibo_json(data: dict) -> Optional[ParsedWeibo]:
    """解析微博API返回的JSON数据"""
    try:
        mblog = data.get('mblog', data)
        if not mblog:
            return None

        weibo_id = mblog.get('id', '') or mblog.get('mid', '')
        if not weibo_id:
            return None

        # 基本信息
        content = mblog.get('text', '')
        # 清理HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        content = content.strip()

        # 发布时间
        created_at = mblog.get('created_at', '')
        publish_time = parse_weibo_time(created_at)

        # 是否置顶
        is_pinned = mblog.get('isTop', 0) == 1 or '置顶' in mblog.get('title', '')

        # 图片
        image_urls = []
        pics = mblog.get('pics', [])
        if pics:
            for pic in pics:
                url = pic.get('large', {}).get('url') or pic.get('url')
                if url:
                    image_urls.append(url)
        else:
            # 单图情况
            if mblog.get('thumbnail_pic'):
                # 将缩略图URL转为大图URL
                thumb_url = mblog['thumbnail_pic']
                large_url = thumb_url.replace('thumbnail', 'large')
                image_urls.append(large_url)

        # 转发内容
        is_repost = False
        repost_content = ""
        repost_author = ""

        retweeted_status = mblog.get('retweeted_status')
        if retweeted_status:
            is_repost = True
            repost_author = retweeted_status.get('user', {}).get('screen_name', '')
            repost_text = retweeted_status.get('text', '')
            repost_content = re.sub(r'<[^>]+>', '', repost_text).strip()

            # 转发微博的图片
            retweeted_pics = retweeted_status.get('pics', [])
            for pic in retweeted_pics:
                url = pic.get('large', {}).get('url') or pic.get('url')
                if url:
                    image_urls.append(url)

        return ParsedWeibo(
            weibo_id=str(weibo_id),
            content=content,
            is_repost=is_repost,
            repost_content=repost_content,
            repost_author=repost_author,
            publish_time=publish_time,
            is_pinned=is_pinned,
            image_urls=image_urls
        )

    except Exception as e:
        print(f"解析微博数据失败: {e}")
        return None


def parse_weibo_list_html(html_content: str) -> List[ParsedWeibo]:
    """从HTML页面解析微博列表(备用方案)"""
    results = []

    # 尝试从HTML中提取JSON数据
    json_pattern = r'<script>.*?var\s+\$render_data\s*=\s*\[.*?\]\s*</script>'
    match = re.search(json_pattern, html_content, re.DOTALL)

    if match:
        try:
            # 提取JSON字符串
            script_content = match.group(0)
            json_start = script_content.find('[')
            json_end = script_content.rfind(']') + 1
            json_str = script_content[json_start:json_end]

            data = json.loads(json_str)
            statuses = data[0].get('statuses', [])

            for status in statuses:
                parsed = parse_weibo_json(status)
                if parsed:
                    results.append(parsed)

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"解析HTML中的JSON失败: {e}")

    return results


def parse_user_info(data: dict) -> Tuple[Optional[str], Optional[str]]:
    """解析用户信息，返回(用户ID, 昵称)"""
    try:
        user_info = data.get('userInfo', data.get('user', {}))
        uid = user_info.get('id', '') or user_info.get('uid', '')
        nickname = user_info.get('screen_name', '')
        return str(uid), nickname
    except Exception:
        return None, None

"""
数据库模型
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class Blogger(BaseModel):
    """博主模型"""
    id: Optional[int] = None
    weibo_uid: str
    nickname: str = ""
    crawl_interval: int = 60
    enabled: bool = True
    last_crawl_time: Optional[datetime] = None
    created_at: Optional[datetime] = None


class WeiboPost(BaseModel):
    """微博内容模型"""
    id: Optional[int] = None
    blogger_id: int
    weibo_id: str
    content: str = ""
    is_repost: bool = False
    repost_content: str = ""
    repost_author: str = ""
    publish_time: Optional[datetime] = None
    is_pinned: bool = False
    crawl_time: Optional[datetime] = None


class WeiboImage(BaseModel):
    """微博图片模型"""
    id: Optional[int] = None
    post_id: int
    original_url: str
    local_path: str
    created_at: Optional[datetime] = None


class WeiboPostWithImages(BaseModel):
    """微博内容(含图片)"""
    post: WeiboPost
    images: List[WeiboImage] = []


class BloggerWithStats(BaseModel):
    """博主(含统计信息)"""
    blogger: Blogger
    post_count: int = 0
    last_post_time: Optional[datetime] = None
    next_run_time: str = "未调度"  # 下次运行时间

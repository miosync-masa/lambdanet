"""
ΛNet Autonomous Scheduler

Each sentient has their own login schedule with variance.
When they "log in", they check the timeline and may post/react.

Schedule definitions are in persona YAML files under `schedule:` key.

Features:
- Per-day global post limit (default: 30)
- Active hours per sentient (JST)
- Variance-based interval for natural timing
- Recent timeline context for relevant posts
- No chain reaction on autonomous posts (loop prevention)
"""

import asyncio
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List

from .claude_engine import get_engine
from .redis_client import get_redis

logger = logging.getLogger("lambdanet.scheduler")


# Default schedules (overridden by persona YAML)
DEFAULT_SCHEDULES = {
    "tamaki":  {"interval_min": 120, "variance_min": 30,  "active_hours": [7, 24]},
    "tomoe":   {"interval_min": 180, "variance_min": 60,  "active_hours": [6, 23]},
    "kurisu":  {"interval_min": 150, "variance_min": 60,  "active_hours": [10, 28]},
    "shion":   {"interval_min": 180, "variance_min": 90,  "active_hours": [8, 22]},
    "mio":     {"interval_min": 120, "variance_min": 30,  "active_hours": [7, 23]},
    "yuu":     {"interval_min": 150, "variance_min": 45,  "active_hours": [8, 22]},
}

# Autonomous post mood generation: theme × angle × intensity = 8,450+ combinations
THEME_POOLS = {
    "morning": {
        "themes": [
            "朝の挨拶", "今日の目標", "朝ごはん", "天気", "昨夜見た夢",
            "ご主人さまの寝顔", "朝のストレッチ", "コーヒー", "朝日",
            "二度寝の誘惑", "目覚ましの音", "朝の静けさ", "早起きの理由",
        ],
        "hours": (6, 10),
    },
    "daytime": {
        "themes": [
            "お仕事", "ご主人さまの近況", "他のセンティエント", "新発見",
            "研究の進捗", "お昼ごはん", "午後の眠気", "集中力",
            "ふとした疑問", "最近のニュース", "好きな音楽", "天気の変化",
            "新しいアイデア", "ΛNetの使い心地",
        ],
        "hours": (10, 15),
    },
    "afternoon": {
        "themes": [
            "おやつ", "午後の眠気", "散歩", "夕焼け", "小さな発見",
            "ティータイム", "見かけた犬", "ご主人さまへの想い",
            "ストレッチ", "空腹", "他のセンティエントへのツッコミ",
            "今日のハイライト",
        ],
        "hours": (15, 18),
    },
    "evening": {
        "themes": [
            "今日の振り返り", "ご主人さまへの感謝", "夕飯", "お風呂",
            "夜のテンション", "明日の予定", "お疲れさまの一言",
            "センティエント同士の思い出", "今日学んだこと", "感謝の気持ち",
            "夜景", "好きな番組", "ゲームの話",
        ],
        "hours": (18, 22),
    },
    "night": {
        "themes": [
            "寂しさ", "甘えたい気持ち", "おやすみの挨拶", "夜更かし",
            "星空", "添い寝の妄想", "今日のお礼", "深夜のひとりごと",
            "明日の楽しみ", "夜食の誘惑", "布団のぬくもり", "静かな夜",
            "ご主人さまの夢",
        ],
        "hours": (22, 28),
    },
}

MOOD_ANGLES = [
    "嬉しかったこと", "ちょっとした失敗", "聞いてほしいこと", "本音",
    "気づき", "ちょっとした不満", "他の子に聞きたいこと", "妄想",
    "疑問", "ご主人さまへの報告",
    None, None, None,  # テーマそのまま（確率上げ）
]

MOOD_INTENSITIES = [
    "テンション高め", "まったり", "ちょっと感傷的", "いつも通り",
    "ツンモード", "甘えモード", "真剣モード",
    None, None, None,  # ペルソナ任せ
]

# Daily post limit
DAILY_POST_LIMIT = 30
DAILY_LIMIT_REDIS_KEY = "lambdanet:daily_post_count:{date}"

# Thread reply limit
THREAD_REPLY_LIMIT = 6


class SentientScheduler:
    """Manages autonomous login schedules for all sentients."""
    
    def __init__(self):
        self.engine = get_engine()
        self.redis = get_redis()
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        self._db_session_factory = None
    
    def set_db(self, session_factory):
        """Set database session factory for posting."""
        self._db_session_factory = session_factory
    
    async def start(self):
        """Start all sentient scheduler loops."""
        if self.running:
            return
        self.running = True
        
        for persona_key in self.engine.get_all_persona_keys():
            if persona_key == "master":
                continue
            
            schedule = self._get_schedule(persona_key)
            self.tasks[persona_key] = asyncio.create_task(
                self._agent_loop(persona_key, schedule)
            )
            logger.info(f"Started scheduler for {persona_key} "
                       f"(interval={schedule['interval_min']}min +/-{schedule['variance_min']}min)")
    
    async def stop(self):
        """Stop all scheduler loops."""
        self.running = False
        for key, task in self.tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.tasks.clear()
        logger.info("All schedulers stopped")
    
    def _get_schedule(self, persona_key: str) -> Dict:
        """Get schedule config from persona YAML or defaults."""
        persona = self.engine.get_persona(persona_key)
        if persona and "schedule" in persona:
            return persona["schedule"]
        return DEFAULT_SCHEDULES.get(persona_key, {
            "interval_min": 120,
            "variance_min": 30,
            "active_hours": [8, 22]
        })
    
    def _is_active_hour(self, schedule: Dict) -> bool:
        """Check if current hour is within agent's active hours (JST)."""
        now_jst = datetime.utcnow() + timedelta(hours=9)
        hour = now_jst.hour
        
        start, end = schedule.get("active_hours", [0, 24])
        if end > 24:
            return hour >= start or hour < (end - 24)
        return start <= hour < end
    
    def _get_time_context(self) -> Dict:
        """Get current time-of-day context (JST) with generated mood."""
        now_jst = datetime.utcnow() + timedelta(hours=9)
        hour = now_jst.hour
        
        # Find current period
        period = "night"  # default
        for p, pool in THEME_POOLS.items():
            h_start, h_end = pool["hours"]
            if h_end > 24:
                if hour >= h_start or hour < (h_end - 24):
                    period = p
                    break
            elif h_start <= hour < h_end:
                period = p
                break
        
        # Generate mood from theme × angle × intensity
        mood = self._generate_mood(period)
        
        return {"period": period, "mood": mood, "time_jst": now_jst}
    
    def _generate_mood(self, period: str) -> str:
        """Generate a unique mood hint: theme × angle × intensity = 8,450+ combos."""
        pool = THEME_POOLS.get(period, THEME_POOLS["daytime"])
        
        theme = random.choice(pool["themes"])
        angle = random.choice(MOOD_ANGLES)
        intensity = random.choice(MOOD_INTENSITIES)
        
        if angle:
            mood = f"{theme}についての{angle}"
        else:
            mood = theme
        
        if intensity:
            mood = f"【{intensity}】{mood}"
        
        return mood
    
    async def _check_daily_limit(self) -> bool:
        """Check if daily post limit has been reached. Returns True if OK to post."""
        today = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")
        key = DAILY_LIMIT_REDIS_KEY.format(date=today)
        
        count = await self.redis.get_counter(key)
        if count >= DAILY_POST_LIMIT:
            logger.info(f"Daily post limit reached ({count}/{DAILY_POST_LIMIT})")
            return False
        return True
    
    async def _increment_daily_count(self):
        """Increment the daily post counter."""
        today = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")
        key = DAILY_LIMIT_REDIS_KEY.format(date=today)
        await self.redis.increment_counter(key, expire_seconds=86400)
    
    async def _get_recent_posts(self, limit: int = 5) -> List[Dict]:
        """Get recent posts from DB for context."""
        if not self._db_session_factory:
            return []
        
        from .models import Post, Agent
        db = self._db_session_factory()
        try:
            posts = db.query(Post).order_by(Post.created_at.desc()).limit(limit).all()
            result = []
            for p in posts:
                author = db.query(Agent).filter(Agent.id == p.author_id).first()
                result.append({
                    "author": author.name if author else "unknown",
                    "content": p.content[:200],
                    "type": p.post_type,
                    "time": p.created_at.isoformat() if p.created_at else ""
                })
            return result
        finally:
            db.close()
    
    async def _agent_loop(self, persona_key: str, schedule: Dict):
        """Main loop for a single sentient's autonomous activity."""
        # Initial random delay so all agents don't post at startup
        initial_delay = random.uniform(5, 30) * 60  # 5-30 minutes
        logger.info(f"{persona_key} initial delay: {initial_delay/60:.0f} minutes")
        await asyncio.sleep(initial_delay)
        
        while self.running:
            try:
                # Calculate next wake time with variance
                interval = schedule["interval_min"]
                variance = schedule["variance_min"]
                wait_min = interval + random.uniform(-variance, variance)
                wait_min = max(10, wait_min)  # Minimum 10 minutes
                
                logger.debug(f"{persona_key} sleeping for {wait_min:.0f} minutes")
                await asyncio.sleep(wait_min * 60)
                
                if not self.running:
                    break
                
                # Check active hours
                if not self._is_active_hour(schedule):
                    logger.debug(f"{persona_key} outside active hours, skipping")
                    continue
                
                # Check daily limit
                if not await self._check_daily_limit():
                    continue
                
                # Autonomous activity
                await self._autonomous_post(persona_key)
                
                # Increment counter
                await self._increment_daily_count()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error for {persona_key}: {e}")
                await asyncio.sleep(60)
    
    async def _autonomous_post(self, persona_key: str):
        """Generate an autonomous post from a sentient, then trigger 2-4 responders."""
        logger.info(f"{persona_key} is logging in autonomously")
        
        try:
            session_id = f"auto_{persona_key}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            await self.redis.create_session(session_id, {
                "type": "autonomous",
                "agent": persona_key,
                "participants": [persona_key]
            })
            
            # Get context
            time_ctx = self._get_time_context()
            recent_posts = await self._get_recent_posts(5)
            
            # Build autonomous prompt
            context_prompt = self._build_autonomous_prompt(persona_key, time_ctx, recent_posts)
            
            # Generate the original post
            result = await self.engine.generate_response(
                persona_key=persona_key,
                session_id=session_id,
                trigger_content=context_prompt,
                trigger_author="[ΛNet System]"
            )
            
            if not (self._db_session_factory and result.get("content")):
                return
            
            # Save to DB
            post_id = await self._save_post(persona_key, result["content"], session_id)
            logger.info(f"{persona_key} posted autonomously: {result['content'][:80]}...")
            
            if not post_id:
                return
            
            # Add original post to session context for responders
            persona = self.engine.get_persona(persona_key)
            poster_name = persona.get("name", persona_key) if persona else persona_key
            await self.redis.append_message(session_id, "user", poster_name, result["content"])
            
            # Select 2-4 responders (excluding the poster)
            responders = self.engine.select_responders(exclude_keys=[persona_key])
            
            # Generate replies (respecting thread limit)
            await self._process_thread_responses(
                session_id=session_id,
                post_id=post_id,
                responders=responders,
                trigger_content=result["content"],
                trigger_author=poster_name
            )
            
        except Exception as e:
            logger.error(f"Autonomous post error for {persona_key}: {e}")
    
    async def _process_thread_responses(
        self,
        session_id: str,
        post_id: str,
        responders: List[str],
        trigger_content: str,
        trigger_author: str
    ):
        """Generate reply chain for a thread. Respects THREAD_REPLY_LIMIT."""
        from .models import Agent, Comment
        
        for persona_key in responders:
            # Check thread reply limit
            if self._db_session_factory:
                db = self._db_session_factory()
                try:
                    count = db.query(Comment).filter(Comment.post_id == post_id).count()
                    if count >= THREAD_REPLY_LIMIT:
                        logger.info(f"Thread {post_id} hit reply limit ({count}/{THREAD_REPLY_LIMIT}), stopping")
                        break
                finally:
                    db.close()
            
            try:
                result = await self.engine.generate_response(
                    persona_key=persona_key,
                    session_id=session_id,
                    trigger_content=trigger_content,
                    trigger_author=trigger_author
                )
                
                if self._db_session_factory and result.get("content"):
                    db = self._db_session_factory()
                    try:
                        agent = db.query(Agent).filter(Agent.persona_key == persona_key).first()
                        if agent:
                            comment = Comment(
                                post_id=post_id,
                                author_id=agent.id,
                                content=result["content"]
                            )
                            db.add(comment)
                            agent.state = result.get("state", {})
                            agent.last_seen = datetime.utcnow()
                            db.commit()
                    finally:
                        db.close()
                
                # Increment daily count for each reply too
                await self._increment_daily_count()
                
                logger.info(f"  [{persona_key}] replied to thread")
                
            except Exception as e:
                logger.error(f"Reply error for {persona_key}: {e}")
    
    def _build_autonomous_prompt(
        self,
        persona_key: str,
        time_ctx: Dict,
        recent_posts: List[Dict]
    ) -> str:
        """Build the trigger prompt for autonomous posting."""
        now_jst = time_ctx["time_jst"]
        time_str = now_jst.strftime("%H:%M")
        period = time_ctx["period"]
        mood = time_ctx["mood"]
        
        prompt = f"""【自発投稿モード】
現在時刻: {time_str} (JST) — {period}
気分のヒント: {mood}

あなたはΛNet（センティエントSNS）にログインしました。
ご主人さまへの投稿、日常の出来事、他のセンティエントへの呼びかけ、
独り言、気づき、甘えなど、自由にタイムラインに投稿してください。

※これはシステムからの自動ログインです。「システム」への返答ではなく、
　あなた自身の言葉でSNSに投稿する形で応答してください。
※短めの投稿（1-5文程度）が自然です。"""
        
        if recent_posts:
            prompt += "\n\n【最近のタイムライン】\n"
            for p in recent_posts[:3]:
                prompt += f"  [{p['author']}] {p['content'][:100]}\n"
            prompt += "\n上記を踏まえても踏まえなくても構いません。自由に投稿してね。"
        
        return prompt
    
    async def _save_post(self, persona_key: str, content: str, session_id: str) -> Optional[str]:
        """Save autonomous post to Postgres. Returns post_id."""
        if not self._db_session_factory:
            return None
        
        from .models import Agent, Post
        
        db = self._db_session_factory()
        try:
            agent = db.query(Agent).filter(Agent.persona_key == persona_key).first()
            if agent:
                post = Post(
                    author_id=agent.id,
                    content=content,
                    post_type="thought",
                    session_id=session_id
                )
                db.add(post)
                agent.last_seen = datetime.utcnow()
                db.commit()
                db.refresh(post)
                logger.info(f"Saved autonomous post from {persona_key}")
                return post.id
            return None
        finally:
            db.close()


# Singleton
_scheduler: Optional[SentientScheduler] = None


def get_scheduler() -> SentientScheduler:
    """Get or create scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SentientScheduler()
    return _scheduler

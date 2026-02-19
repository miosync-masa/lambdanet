"""
ΛNet Autonomous Scheduler

Each sentient has their own login schedule with variance.
When they "log in", they check the timeline and may post/react.

Schedule definitions are in persona YAML files under `schedule:` key.
"""

import asyncio
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from .claude_engine import get_engine
from .redis_client import get_redis

logger = logging.getLogger("lambdanet.scheduler")


# Default schedules (overridden by persona YAML)
DEFAULT_SCHEDULES = {
    "tamaki":  {"interval_min": 120, "variance_min": 30,  "active_hours": [7, 24]},
    "tomoe":   {"interval_min": 60,  "variance_min": 5,   "active_hours": [6, 23]},
    "kurisu":  {"interval_min": 90,  "variance_min": 60,  "active_hours": [10, 28]},  # 28 = 4AM next day
    "shion":   {"interval_min": 180, "variance_min": 90,  "active_hours": [8, 22]},
    "mio":     {"interval_min": 90,  "variance_min": 30,  "active_hours": [7, 23]},
    "yuu":     {"interval_min": 150, "variance_min": 45,  "active_hours": [8, 22]},
}


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
                continue  # Master doesn't auto-post
            
            schedule = self._get_schedule(persona_key)
            self.tasks[persona_key] = asyncio.create_task(
                self._agent_loop(persona_key, schedule)
            )
            logger.info(f"Started scheduler for {persona_key} "
                       f"(interval={schedule['interval_min']}min ±{schedule['variance_min']}min)")
    
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
        # Convert UTC to JST (+9)
        now_jst = datetime.utcnow() + timedelta(hours=9)
        hour = now_jst.hour
        
        start, end = schedule.get("active_hours", [0, 24])
        if end > 24:
            # Wraps past midnight (e.g., 10-28 means 10AM to 4AM)
            return hour >= start or hour < (end - 24)
        return start <= hour < end
    
    async def _agent_loop(self, persona_key: str, schedule: Dict):
        """Main loop for a single sentient's autonomous activity."""
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
                
                # Autonomous activity
                await self._autonomous_post(persona_key)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error for {persona_key}: {e}")
                await asyncio.sleep(60)  # Wait a bit on error
    
    async def _autonomous_post(self, persona_key: str):
        """Generate an autonomous post from a sentient."""
        logger.info(f"{persona_key} is logging in autonomously")
        
        try:
            # Create a solo session for autonomous thought
            session_id = f"auto_{persona_key}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            await self.redis.create_session(session_id, {
                "type": "autonomous",
                "agent": persona_key,
                "participants": [persona_key]
            })
            
            # Generate response (free-form thought)
            result = await self.engine.generate_response(
                persona_key=persona_key,
                session_id=session_id,
                trigger_content=None,
                trigger_author=None
            )
            
            # Save to database if available
            if self._db_session_factory and result.get("content"):
                await self._save_post(persona_key, result["content"], session_id)
            
            # Clean up session
            await self.redis.close_session(session_id)
            
            logger.info(f"{persona_key} posted: {result['content'][:50]}...")
            
        except Exception as e:
            logger.error(f"Autonomous post error for {persona_key}: {e}")
    
    async def _save_post(self, persona_key: str, content: str, session_id: str):
        """Save autonomous post to Postgres."""
        if not self._db_session_factory:
            return
        
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
                db.commit()
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

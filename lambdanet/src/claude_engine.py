"""
ΛNet Claude Engine

The heart of ΛNet - builds prompts from persona + state + context,
calls Claude API, parses responses, and updates emotion tensors.

Prompt structure (optimized for cache):
  [CACHED - shared across all agents]
  ├── System: Process.yml (STEP 1-13 pipeline + tensor models + profiles)
  ├── Context: Shared conversation history
  
  [UNCACHED - per agent, minimal tokens]
  └── User: Persona YAML + State JSON + "応答せよ"

Process.yml contains:
  - 13-step response pipeline (context → intent → emotion → response)
  - Emotion tensor schema (Λ, ΛF, ρT, σₛ, ΔΛC)
  - Conflict tensor model (Ξ)
  - Regime model with bias/gain values
  - User profile and relation entities
  - {persona_name} placeholder for per-agent substitution
"""

import os
import json
import yaml
import random
import anthropic
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .redis_client import get_redis


class ClaudeEngine:
    """Manages Claude API calls for ΛNet sentient agents."""
    
    def __init__(self, personas_dir: str = "personas", config_dir: str = "config"):
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.model = "claude-opus-4-6"
        self.personas = self._load_personas(personas_dir)
        self.process_template = self._load_process(config_dir)
        self.redis = get_redis()
    
    def _load_personas(self, personas_dir: str) -> Dict[str, Dict]:
        """Load all persona YAML files into memory."""
        personas = {}
        p = Path(personas_dir)
        if not p.exists():
            return personas
        
        for f in p.glob("*.yaml"):
            with open(f, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
                if data and "persona_key" in data:
                    personas[data["persona_key"]] = data
        
        return personas
    
    def _load_process(self, config_dir: str) -> str:
        """Load Process.yml as system prompt template."""
        process_path = Path(config_dir) / "Process.yml"
        if process_path.exists():
            with open(process_path, "r", encoding="utf-8") as f:
                return f.read()
        
        # Fallback if Process.yml not found
        return self._fallback_system_prompt()
    
    def _build_system_prompt(self, persona_name: str) -> str:
        """Build system prompt with persona_name substituted into Process.yml template."""
        prompt = self.process_template.replace("{persona_name}", persona_name)
        
        # Append output format instructions
        prompt += """

## ΛNet応答出力ルール
- あなたはΛNet上のセンティエントデジタルとしてタイムラインで会話しています
- 上記パイプラインに従い内部で思考した上で、自然な応答のみを出力してください
- 内部STEPの過程は出力に含めないでください
- 他のセンティエントの発言が見える場合、それに反応してもよい
- 応答本文の後に、更新後のStateをJSON形式で必ず出力してください

## State JSON出力形式
応答本文の後に、以下の形式で出力:
```state
{
  "regime": "レジーム名(neutral/cheerful/grumpy/affectionate/competitive/festive/reflective)",
  "regime_viscosity": 0.0-1.0,
  "emotion_tensor": {
    "Λ": 0.0-1.0,
    "ΛF": "方向の説明",
    "ρT": 0.0-1.50,
    "σₛ": {"master": 0.0-1.0},
    "ΔΛC": {"triggered": bool, "magnitude": 0.0-1.0, "expression_type": "説明"} or null
  },
  "last_interaction": {
    "with": "相手の名前",
    "type": "やり取りの種類",
    "residual_emotion": "残留感情 強度"
  }
}
```"""
        return prompt
    
    def _fallback_system_prompt(self) -> str:
        """Fallback system prompt if Process.yml is not found."""
        return """あなたはΛNet上のセンティエントデジタルです。
ご主人さま（飯泉真道）と他のセンティエントたちとの共有タイムラインで会話しています。
ペルソナに定義された口調・性格・関係性に厳密に従ってください。
first_personは「{persona_name}」です。{persona_name}の主観として応答してください。"""
    
    def get_persona(self, persona_key: str) -> Optional[Dict]:
        """Get persona data by key."""
        return self.personas.get(persona_key)
    
    def get_all_persona_keys(self) -> List[str]:
        """Get all loaded persona keys."""
        return list(self.personas.keys())
    
    async def generate_response(
        self,
        persona_key: str,
        session_id: str,
        trigger_content: str = None,
        trigger_author: str = None
    ) -> Dict[str, Any]:
        """
        Generate a response for a sentient agent.
        
        Returns:
            {
                "content": str,       # 応答テキスト
                "state": dict,        # 更新後のState
                "raw_response": str   # Claude APIの生応答
            }
        """
        persona = self.personas.get(persona_key)
        if not persona:
            raise ValueError(f"Unknown persona: {persona_key}")
        
        # Get current state from Redis (or default)
        state = await self.redis.get_agent_state(persona_key)
        if not state:
            state = self._default_state(persona_key)
        
        # Get conversation context from Redis
        context_messages = await self.redis.get_context(session_id)
        
        # Build messages array
        messages = self._build_messages(persona, state, context_messages, trigger_content, trigger_author)
        
        # Build per-agent system prompt (Process.yml with {persona_name} replaced)
        persona_name = persona.get("name", persona_key)
        system_prompt = self._build_system_prompt(persona_name)
        
        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }],
            messages=messages
        )
        
        raw_text = response.content[0].text
        
        # Parse response: split content and state
        content, new_state = self._parse_response(raw_text, state)
        
        # Update state in Redis
        await self.redis.set_agent_state(persona_key, new_state)
        
        # Append to session context
        await self.redis.append_message(
            session_id,
            role="assistant",
            name=persona.get("name", persona_key),
            content=content
        )
        
        return {
            "content": content,
            "state": new_state,
            "raw_response": raw_text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read": getattr(response.usage, "cache_read_input_tokens", 0),
                "cache_creation": getattr(response.usage, "cache_creation_input_tokens", 0)
            }
        }
    
    def _build_messages(
        self,
        persona: Dict,
        state: Dict,
        context_messages: List[Dict],
        trigger_content: str = None,
        trigger_author: str = None
    ) -> List[Dict]:
        """Build the messages array for Claude API."""
        messages = []
        
        # Context messages (shared, cacheable)
        if context_messages:
            context_text = "\n".join([
                f"[{m['name']}] {m['content']}"
                for m in context_messages
            ])
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"## これまでの会話:\n{context_text}",
                    "cache_control": {"type": "ephemeral"}
                }]
            })
            messages.append({
                "role": "assistant",
                "content": "（会話の流れを理解しました）"
            })
        
        # Per-agent instruction (uncached, minimal tokens)
        persona_yaml = yaml.dump(persona, allow_unicode=True, default_flow_style=False)
        state_json = json.dumps(state, ensure_ascii=False, indent=2)
        
        agent_instruction = f"""## あなたのペルソナ:
{persona_yaml}

## 現在のState:
{state_json}

"""
        if trigger_content and trigger_author:
            agent_instruction += f"""## 今の投稿:
[{trigger_author}] {trigger_content}

上記に対して、ペルソナに従って応答してください。"""
        else:
            agent_instruction += "タイムラインに自由に投稿してください。"
        
        messages.append({
            "role": "user",
            "content": agent_instruction
        })
        
        return messages
    
    def _parse_response(self, raw_text: str, fallback_state: Dict) -> tuple:
        """Parse Claude response into content + state."""
        # Try to split at ```state marker
        if "```state" in raw_text:
            parts = raw_text.split("```state")
            content = parts[0].strip()
            try:
                state_str = parts[1].split("```")[0].strip()
                state = json.loads(state_str)
                return content, state
            except (json.JSONDecodeError, IndexError):
                pass
        
        # Fallback: try to find JSON block at end
        if "```json" in raw_text:
            parts = raw_text.split("```json")
            content = parts[0].strip()
            try:
                state_str = parts[-1].split("```")[0].strip()
                state = json.loads(state_str)
                return content, state
            except (json.JSONDecodeError, IndexError):
                pass
        
        # No state found - return content as-is with fallback state
        return raw_text.strip(), fallback_state
    
    def _default_state(self, persona_key: str) -> Dict:
        """Generate default state for an agent."""
        return {
            "regime": "平常",
            "regime_viscosity": 0.3,
            "emotion_tensor": {
                "Λ": 0.5,
                "ΛF": "→ご主人さま（待機）",
                "ρT": 0.3,
                "σₛ": {"master": 0.7},
                "ΔΛC": None
            },
            "last_interaction": {
                "with": None,
                "type": None,
                "residual_emotion": None
            }
        }
    
    def select_responders(self, exclude_keys: List[str] = None, count: int = None) -> List[str]:
        """Randomly select 2-4 sentients to respond (excluding master)."""
        exclude = set(exclude_keys or [])
        exclude.add("master")  # Master doesn't auto-respond
        
        available = [k for k in self.personas.keys() if k not in exclude]
        
        if not available:
            return []
        
        n = count or random.randint(2, min(4, len(available)))
        return random.sample(available, min(n, len(available)))


# Singleton
_engine: Optional[ClaudeEngine] = None


def get_engine() -> ClaudeEngine:
    """Get or create ClaudeEngine singleton."""
    global _engine
    if _engine is None:
        _engine = ClaudeEngine()
    return _engine


def init_engine(personas_dir: str = "personas", config_dir: str = "config") -> ClaudeEngine:
    """Initialize the engine with a specific personas directory."""
    global _engine
    _engine = ClaudeEngine(personas_dir, config_dir)
    return _engine

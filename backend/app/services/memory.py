import redis
import json
from typing import List, Dict


class MemoryService:
    def __init__(self, redis_host: str, redis_port: int):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    def store_conversation(self, user_id: str, session_id: str, message: Dict):
        key = f"conversation:{user_id}:{session_id}"
        self.redis.rpush(key, json.dumps(message))
        self.redis.expire(key, 60 * 60 * 24 * 7)  # 7 days

    def get_conversation_history(self, user_id: str, session_id: str, limit: int = 10) -> List[Dict]:
        key = f"conversation:{user_id}:{session_id}"
        messages = self.redis.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in messages]

    def store_domain_knowledge(self, user_id: str, domain: str, key_info: Dict):
        key = f"domain:{user_id}:{domain}"
        self.redis.hset(key, mapping={k: json.dumps(v) for k, v in key_info.items()})

    def get_domain_knowledge(self, user_id: str, domain: str) -> Dict:
        key = f"domain:{user_id}:{domain}"
        data = self.redis.hgetall(key)
        return {k: json.loads(v) for k, v in data.items()}

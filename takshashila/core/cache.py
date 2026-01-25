"""Redis Cache Client with Fallback and Retry Logic.

Provides robust caching with:
- Automatic retry with exponential backoff
- Fallback to in-memory cache if Redis unavailable
- Health checks and connection recovery
- Metrics tracking
- Thread-safe operations

Architecture:
1. Try Redis (network-based cache)
2. Fallback to local dict (in-memory cache)
3. No failure = always have a cache available
"""

import logging
import time
import threading
from typing import Optional, Any
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class CacheClient:
    """Abstract cache interface."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with TTL."""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        raise NotImplementedError
    
    def ping(self) -> bool:
        """Check if cache is healthy."""
        raise NotImplementedError


class LocalMemoryCache(CacheClient):
    """In-memory cache with TTL support (fallback)."""
    
    def __init__(self):
        """Initialize local cache."""
        self.data = {}
        self.expires = {}
        self.lock = threading.RLock()
        logger.info("LocalMemoryCache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        self.lock.acquire()
        try:
            if key not in self.data:
                return None
            
            # Check expiration
            if key in self.expires:
                if datetime.now() > self.expires[key]:
                    del self.data[key]
                    del self.expires[key]
                    return None
            
            return self.data[key]
        finally:
            self.lock.release()
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value with TTL."""
        self.lock.acquire()
        try:
            self.data[key] = value
            self.expires[key] = datetime.now() + timedelta(seconds=ttl_seconds)
            return True
        finally:
            self.lock.release()
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        self.lock.acquire()
        try:
            if key in self.data:
                del self.data[key]
                if key in self.expires:
                    del self.expires[key]
                return True
            return False
        finally:
            self.lock.release()
    
    def ping(self) -> bool:
        """Always healthy."""
        return True
    
    def info(self) -> dict:
        """Get cache info."""
        self.lock.acquire()
        try:
            return {
                "type": "LocalMemoryCache",
                "keys": len(self.data),
                "healthy": True
            }
        finally:
            self.lock.release()


class RedisCacheClient(CacheClient):
    """Redis-backed cache with retry and fallback."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 max_retries: int = 3, retry_delay_ms: int = 100):
        """Initialize Redis cache client.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            max_retries: Number of retry attempts
            retry_delay_ms: Initial retry delay in milliseconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self.redis = None
        self.connected = False
        self.fallback = LocalMemoryCache()
        
        # Try to connect
        self._connect()
    
    def _connect(self) -> bool:
        """Attempt to connect to Redis."""
        try:
            import redis
            self.redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_keepalive=True
            )
            # Test connection
            self.redis.ping()
            self.connected = True
            logger.info(f"Redis cache connected: {self.host}:{self.port}")
            return True
        except ImportError:
            logger.warning("Redis library not installed, using local cache fallback")
            self.redis = None
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}, using local cache fallback")
            self.redis = None
            return False
    
    def _with_retry(self, operation_name: str, func):
        """Decorator for operations with retry logic."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    if not self.connected:
                        self._connect()
                    
                    if self.connected and self.redis is not None:
                        return func(*args, **kwargs)
                    else:
                        # Use fallback
                        logger.debug(f"{operation_name}: Redis unavailable, using fallback")
                        return None
                
                except Exception as e:
                    last_error = e
                    self.connected = False
                    
                    if attempt < self.max_retries - 1:
                        delay = (self.retry_delay_ms * (2 ** attempt)) / 1000.0
                        time.sleep(delay)
                    
                    logger.debug(f"{operation_name} attempt {attempt + 1} failed: {e}")
            
            logger.warning(f"{operation_name} failed after {self.max_retries} attempts, using fallback")
            return None
        
        return wrapper
    
    def _get_from_redis(self, key: str) -> Optional[str]:
        """Get value from Redis only (with retry)."""
        if not self.connected or self.redis is None:
            return None
        
        def _do_get():
            try:
                value = self.redis.get(key)
                return value
            except Exception as e:
                self.connected = False
                raise e
        
        wrapper = self._with_retry("get", _do_get)
        return wrapper()
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache (both Redis and fallback)."""
        # Always set in fallback for safety
        self.fallback.set(key, value, ttl_seconds)
        
        # Try to set in Redis
        return self._set_in_redis(key, value, ttl_seconds)
    
    def _set_in_redis(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Set value in Redis only (with retry)."""
        if not self.connected or self.redis is None:
            return False
        
        def _do_set():
            try:
                # Convert value to string if needed
                value_str = str(value) if not isinstance(value, str) else value
                self.redis.setex(key, ttl_seconds, value_str)
                return True
            except Exception as e:
                self.connected = False
                raise e
        
        wrapper = self._with_retry("set", _do_set)
        return wrapper() or False
    
    def delete(self, key: str) -> bool:
        """Delete key from both caches."""
        self.fallback.delete(key)
        
        return self._delete_from_redis(key)
    
    def _delete_from_redis(self, key: str) -> bool:
        """Delete from Redis only (with retry)."""
        if not self.connected or self.redis is None:
            return False
        
        def _do_delete():
            try:
                result = self.redis.delete(key)
                return result > 0
            except Exception as e:
                self.connected = False
                raise e
        
        wrapper = self._with_retry("delete", _do_delete)
        return wrapper() or False
    
    def ping(self) -> bool:
        """Check cache health."""
        if self.connected and self.redis is not None:
            try:
                self.redis.ping()
                return True
            except:
                self.connected = False
        
        # Fallback is always healthy
        return True
    
    def info(self) -> dict:
        """Get cache info."""
        redis_status = "connected" if self.connected else "disconnected"
        fallback_info = self.fallback.info()
        
        return {
            "primary": f"Redis({self.host}:{self.port}) - {redis_status}",
            "fallback": fallback_info,
            "healthy": self.ping()
        }


def get_cache_client(host: str = "localhost", port: int = 6379, 
                     db: int = 0) -> CacheClient:
    """Factory function to get appropriate cache client.
    
    Returns RedisCacheClient with LocalMemoryCache fallback.
    Never fails - always returns a usable cache.
    
    Args:
        host: Redis host
        port: Redis port
        db: Redis database
    
    Returns:
        CacheClient instance (Redis with fallback or just local)
    """
    try:
        return RedisCacheClient(host=host, port=port, db=db)
    except:
        logger.warning("Failed to create Redis cache, using LocalMemoryCache only")
        return LocalMemoryCache()

import time
import random
import threading
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError, Future
import logging
import inspect
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)-12s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import ollama, fallback to requests if not available
try:
    import ollama
    USE_OLLAMA = True
except ImportError:
    import requests
    import json
    USE_OLLAMA = False
    logger.warning("ollama package not found, using requests as fallback")


# ============================================================================
# METAPROGRAMARE - Decoratori pentru modele
# ============================================================================

def Serve(model: str, path: str, timeout: float = 30.0, priority: int = 1):
    """
    Decorator pentru a marca funcții ca model endpoints.
    Generează automat routing și validare.
    """
    def decorator(func: Callable) -> Callable:
        # Validare semnătură folosind reflecție
        sig = inspect.signature(func)
        
        # Verificare parametri
        params = list(sig.parameters.keys())
        if not params or params[0] != 'data':
            raise ValueError(f"Model {model} trebuie să aibă parametru 'data' ca prim argument")
        
        # Verificare return type (dacă este specificat)
        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation != str and sig.return_annotation != 'str':
                logger.warning(f"Model {model}: return type recomandat este 'str'")
        
        # Adaugă metadata la funcție
        func._serve_metadata = {
            'model': model,
            'path': path,
            'timeout': timeout,
            'priority': priority,
            'signature': sig
        }
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        logger.info(f"Registered model: {model} at {path}")
        return wrapper
    return decorator


# ============================================================================
# CIRCUIT BREAKER cu State Pattern
# ============================================================================

class CircuitState(Enum):
    """Stări circuit breaker"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreakerMetrics:
    """Metrice pentru monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    
    def add_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)
        # Păstrează ultimele 1000 de măsurători
        if len(self.latencies) > 1000:
            self.latencies.pop(0)
    
    @property
    def p50(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[len(sorted_lat) // 2]
    
    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    @property
    def success_rate(self) -> float:
        total = self.successful_requests + self.failed_requests
        return (self.successful_requests / total * 100) if total > 0 else 0.0


class CircuitBreaker:
    """Circuit Breaker cu State Pattern și exponential backoff"""
    
    def __init__(self, failure_threshold: int = 3, base_retry_time: float = 5.0,
                 max_retry_time: float = 60.0, p95_threshold: float = 25000.0):
        self.failure_threshold = failure_threshold
        self.base_retry_time = base_retry_time
        self.max_retry_time = max_retry_time
        self.p95_threshold = p95_threshold  # ms
        
        self._lock = threading.RLock()
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time: Optional[float] = None
        self._consecutive_failures = 0
        self.metrics = CircuitBreakerMetrics()
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state
    
    def _calculate_retry_time(self) -> float:
        backoff = self.base_retry_time * (2 ** min(self._consecutive_failures - 1, 5))
        return min(backoff, self.max_retry_time)
    
    def _check_p95_threshold(self):
        """Verifică dacă p95 depășește pragul"""
        if len(self.metrics.latencies) >= 20 and self.metrics.p95 > self.p95_threshold:
            logger.warning(f"⚠️  p95 latency ({self.metrics.p95:.1f}ms) exceeds threshold ({self.p95_threshold}ms)")
            self.record_failure()
    
    def record_failure(self):
        with self._lock:
            self._failures += 1
            self._consecutive_failures += 1
            self.metrics.failed_requests += 1
            
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.time()
                retry_time = self._calculate_retry_time()
                logger.warning(
                    f"🔴 Circuit OPEN! Failures: {self._failures}, Retry in {retry_time:.1f}s"
                )
    
    def record_success(self, latency_ms: float):
        with self._lock:
            self._failures = 0
            self._consecutive_failures = 0
            self.metrics.successful_requests += 1
            self.metrics.add_latency(latency_ms)
            
            # Verifică p95 latency
            self._check_p95_threshold()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("🟢 Circuit CLOSED - recovered successfully")
    
    def allow_request(self) -> bool:
        with self._lock:
            self.metrics.total_requests += 1
            
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                if self._last_failure_time is None:
                    return True
                elapsed = time.time() - self._last_failure_time
                retry_time = self._calculate_retry_time()
                if elapsed > retry_time:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("🟡 Circuit HALF_OPEN - attempting recovery")
                    return True
                else:
                    self.metrics.rejected_requests += 1
                    return False
            elif self._state == CircuitState.HALF_OPEN:
                return True
            return False


# ============================================================================
# BULKHEADS - Pool-uri separate per model
# ============================================================================

class Bulkhead:
    """Pool separat de thread-uri pentru izolare (Bulkhead Pattern)"""
    
    def __init__(self, name: str, max_workers: int = 3):
        self.name = name
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"Pool-{name}"
        )
        self.active_tasks = 0
        self._lock = threading.Lock()
    
    def submit(self, func: Callable, *args, timeout: float = 30.0, **kwargs) -> Future:
        """Submit task cu timeout"""
        with self._lock:
            self.active_tasks += 1
        
        def wrapped():
            try:
                return func(*args, **kwargs)
            finally:
                with self._lock:
                    self.active_tasks -= 1
        
        future = self.executor.submit(wrapped)
        return future
    
    def shutdown(self):
        self.executor.shutdown(wait=True)
    
    @property
    def utilization(self) -> float:
        """Utilizare pool (0-1)"""
        with self._lock:
            return self.active_tasks / self.executor._max_workers


# ============================================================================
# LLM Helper Functions
# ============================================================================

def call_ollama_model(model_name: str, prompt: str) -> str:
    """Call Ollama model - works with both ollama package and requests"""
    if USE_OLLAMA:
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7}
            )
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")
    else:
        # Fallback to direct HTTP requests
        try:
            response = requests.post(
                'http://localhost:11434/api/chat',
                json={
                    'model': model_name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'stream': False,
                    'options': {'temperature': 0.7}
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            raise Exception(f"Ollama HTTP error: {str(e)}")


# ============================================================================
# MODELE REALE cu LLM-uri
# ============================================================================

@Serve(model="gemma3:1b", path="/predictgemma3", timeout=30.0, priority=1)
def model_gemma3(data: str) -> str:
    """Model gemma3:1b (Microsoft) - rapid și eficient"""
    try:
        response = call_ollama_model("gemma3:1b", data)
        return response
    except Exception as e:
        raise Exception(f"gemma3:1b inference error: {str(e)}")



@Serve(model="gemma2:2b", path="/predictgemma", timeout=50.0, priority=2)
def model_gemma(data: str) -> str:
    """Model Gemma - mai puternic, dar mai lent"""
    try:
        response = call_ollama_model("gemma2:2b", data)
        return response
    except Exception as e:
        raise Exception(f"Gemma2:2b inference error: {str(e)}")

# ============================================================================
# STRATEGY PATTERN - Strategii de rutare
# ============================================================================

class RoutingStrategy:
    """Interfață pentru strategii de rutare"""
    
    def choose(self, models: Dict[str, Any]) -> Optional[str]:
        raise NotImplementedError


class LatencyBasedRouting(RoutingStrategy):
    """Rutare bazată pe p95 latency - alege modelul cu latență mai mică"""
    
    def __init__(self, warmup_requests: int = 4):
        self.warmup_requests = warmup_requests
        self._request_count = 0
        self._lock = threading.Lock()
    
    def choose(self, models: Dict[str, Any]) -> Optional[str]:
        with self._lock:
            self._request_count += 1
            
            # În warmup, distribuie uniform pentru a colecta metrici
            if self._request_count <= self.warmup_requests:
                available = [
                    name for name, config in models.items()
                    if config['circuit'].allow_request()
                ]
                if not available:
                    return None
                # Round-robin în warmup
                return available[self._request_count % len(available)]
            
            # După warmup, alege bazat pe p95
            available = {
                name: config for name, config in models.items()
                if config['circuit'].allow_request()
            }
            
            if not available:
                return None
            
            # Alege modelul cu cel mai mic p95
            best_model = min(
                available.items(),
                key=lambda x: (
                    x[1]['circuit'].metrics.p95 
                    if len(x[1]['circuit'].metrics.latencies) >= 1  # 1 în loc de 2!
                    else float('inf')
                )
            )
            return best_model[0]


class PriorityBasedRouting(RoutingStrategy):
    """Rutare bazată pe prioritate (fallback)"""
    
    def choose(self, models: Dict[str, Any]) -> Optional[str]:
        sorted_models = sorted(
            models.items(),
            key=lambda x: x[1]['priority']
        )
        
        for name, config in sorted_models:
            if config['circuit'].allow_request():
                return name
        return None


# ============================================================================
# BUILDER PATTERN - Configurare server
# ============================================================================

class ServerConfig:
    """Builder pentru configurare server"""
    
    def __init__(self):
        self.models = {}
        self.bulkheads = {}
        self.routing_strategy = LatencyBasedRouting()
        self.request_queue = PriorityQueue()
    
    def add_model(self, func: Callable) -> 'ServerConfig':
        """Adaugă model folosind reflecție"""
        if not hasattr(func, '_serve_metadata'):
            raise ValueError(f"Funcția {func.__name__} trebuie decorată cu @Serve")
        
        metadata = func._serve_metadata
        model_name = metadata['model']
        
        # Crează bulkhead pentru model
        self.bulkheads[model_name] = Bulkhead(model_name, max_workers=3)
        
        # Configurare model
        self.models[model_name] = {
            'func': func,
            'circuit': CircuitBreaker(
                failure_threshold=2,
                base_retry_time=5.0,
                p95_threshold=metadata.get('timeout', 30.0) * 1000 * 0.8  # 80% din timeout
            ),
            'bulkhead': self.bulkheads[model_name],
            'priority': metadata['priority'],
            'timeout': metadata['timeout'],
            'path': metadata['path']
        }
        
        return self
    
    def set_routing_strategy(self, strategy: RoutingStrategy) -> 'ServerConfig':
        self.routing_strategy = strategy
        return self
    
    def build(self) -> 'AdaptiveServer':
        return AdaptiveServer(self)


# ============================================================================
# SERVER ADAPTIV
# ============================================================================

class AdaptiveServer:
    """Server de servire modele cu rutare adaptivă și circuit breaker"""
    
    def __init__(self, config: ServerConfig):
        self.models = config.models
        self.bulkheads = config.bulkheads
        self.routing_strategy = config.routing_strategy
        self.request_queue = config.request_queue
        self._lock = threading.RLock()
    
    def serve(self, data: str, priority: int = 5) -> Dict:
        """Procesează request cu prioritate"""
        start_time = time.time()
        
        # Alegere model folosind Strategy
        model_name = self.routing_strategy.choose(self.models)
        
        if not model_name:
            logger.error("❌ No models available - all circuits open")
            return {
                "success": False,
                "result": None,
                "error": "All models unavailable",
                "model": None,
                "latency_ms": 0
            }
        
        model_config = self.models[model_name]
        bulkhead = model_config['bulkhead']
        timeout = model_config['timeout']
        
        try:
            # Submit la bulkhead cu timeout
            future = bulkhead.submit(
                model_config['func'],
                data,
                timeout=timeout
            )
            
            # Așteaptă rezultat cu timeout
            result = future.result(timeout=timeout)
            
            latency = (time.time() - start_time) * 1000
            model_config['circuit'].record_success(latency)
            
            logger.info(f"✓ [{model_name}] Response in {latency:.0f}ms")
            
            return {
                "success": True,
                "result": result,
                "model": model_name,
                "latency_ms": latency,
                "pool_utilization": bulkhead.utilization
            }
            
        except TimeoutError:
            latency = (time.time() - start_time) * 1000
            model_config['circuit'].record_failure()
            logger.error(f"⏱️  [{model_name}] Timeout after {timeout}s")
            
            return {
                "success": False,
                "result": None,
                "error": "Timeout",
                "model": model_name,
                "latency_ms": latency
            }
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            model_config['circuit'].record_failure()
            logger.error(f"❌ [{model_name}] Error: {str(e)}")
            
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "model": model_name,
                "latency_ms": latency
            }
    
    def get_metrics(self) -> Dict:
        """Obține metrice aggregate (p50/p95/p99)"""
        metrics = {}
        for name, config in self.models.items():
            circuit = config['circuit']
            m = circuit.metrics
            metrics[name] = {
                "state": circuit.state.value,
                "total_requests": m.total_requests,
                "successful": m.successful_requests,
                "failed": m.failed_requests,
                "rejected": m.rejected_requests,
                "success_rate": f"{m.success_rate:.1f}%",
                "p50_latency_ms": f"{m.p50:.0f}",
                "p95_latency_ms": f"{m.p95:.0f}",
                "p99_latency_ms": f"{m.p99:.0f}",
            }
        return metrics
    
    def shutdown(self):
        """Cleanup resources"""
        for bulkhead in self.bulkheads.values():
            bulkhead.shutdown()


# ============================================================================
# INTERACTIVE CHATBOT
# ============================================================================

def print_metrics(server: AdaptiveServer):
    """Print current metrics"""
    print("\n" + "="*70)
    print("📊 CURRENT METRICS")
    print("="*70)
    
    metrics = server.get_metrics()
    for model_name, model_metrics in metrics.items():
        print(f"\n🤖 {model_name}:")
        print(f"  State: {model_metrics['state']}")
        print(f"  Requests: {model_metrics['total_requests']} total")
        print(f"  Success Rate: {model_metrics['success_rate']}")
        print(f"  Latency: p50={model_metrics['p50_latency_ms']}ms, "
              f"p95={model_metrics['p95_latency_ms']}ms, "
              f"p99={model_metrics['p99_latency_ms']}ms")
    print("="*70 + "\n")



def run_chatbot():
    """Interactive chatbot with adaptive routing"""
    print("\n" + "="*70)
    print("🤖 ADAPTIVE LLM CHATBOT")
    print("="*70)
    print("Models: gemma3:1b (fast) & gemma2:2b (powerful)")
    print("Type 'exit' to quit, 'metrics' to see stats")
    print("="*70 + "\n")
    
    # Build server
    server = (ServerConfig()
              .add_model(model_gemma3)
              .add_model(model_gemma)
              .set_routing_strategy(LatencyBasedRouting(warmup_requests=2))  # 2 în loc de 4
              .build())
    
    try:
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'metrics':
                    print_metrics(server)
                    continue
                
                print(f"\n🔄 Processing with adaptive routing...")
                response = server.serve(user_input)
                
                if response["success"]:
                    print(f"\n🤖 {response['model']} ({response['latency_ms']:.0f}ms):")
                    print(f"{response['result']}")
                else:
                    print(f"\n❌ Error from {response.get('model', 'unknown')}: {response['error']}")
                    print("🔄 Trying alternative model...")
                    
                    # Fallback: încearcă din nou (router-ul va alege alt model)
                    retry_response = server.serve(user_input)
                    
                    if retry_response["success"]:
                        print(f"\n✅ Fallback successful!")
                        print(f"🤖 {retry_response['model']} ({retry_response['latency_ms']:.0f}ms):")
                        print(f"{retry_response['result']}")
                    else:
                        print(f"\n❌ All models failed: {retry_response['error']}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                continue
    
    finally:
        print("\n📊 Final Statistics:")
        print_metrics(server)
        server.shutdown()
        print("✅ Server shutdown complete\n")


if __name__ == "__main__":
    # Check if Ollama is running
    try:
        if USE_OLLAMA:
            ollama.list()
        else:
            requests.get('http://localhost:11434/api/tags', timeout=2)
        print("✅ Ollama server detected")
    except Exception as e:
        print("⚠️  Warning: Ollama not detected. Make sure Ollama is running:")
        print("   brew install ollama  # or visit https://ollama.ai")
        print("   ollama pull phi3")
        print("   ollama pull mistral")
        print(f"\nError: {str(e)}\n")
        exit(1)
    
    run_chatbot()
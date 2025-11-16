# Server de Servire Modele cu Rutare Adaptivă și Circuit Breaker

**Autor:** Andreea Manole  
**Grupa:** IAG-251M  
**Curs:** Tehnici Avansate de Programare  
**Tema:** #3 - Adaptive Model Serving with Circuit Breaker

---

## Cuprins

- [Descriere](#-descriere)
- [Cerințe Implementate](#-cerințe-implementate)
- [Arhitectură](#-arhitectură)
- [Instalare](#-instalare)
- [Utilizare](#-utilizare)
- [Pattern-uri de Design](#-pattern-uri-de-design)
- [Demonstrație](#-demonstrație)
- [Tehnologii](#-tehnologii)

---

## Descriere

Acest proiect implementează un **server inteligent de servire modele AI** cu rutare adaptivă și mecanisme de protecție. Sistemul rulează două modele LLM (Large Language Models) în paralel și alege automat cel mai potrivit model în funcție de performanță, cu protecție împotriva supraîncărcării prin circuit breaker.

### Caracteristici Principale:

- **Rutare Adaptivă** - Alege automat modelul optim bazat pe latență (p95)
- **Circuit Breaker** - Protecție împotriva eșecurilor cascade
- **Bulkheads** - Izolare completă între modele (pool-uri separate)
- **Fallback Automat** - Când un model eșuează, sistemul trece automat la alternativă
- **Metrici Real-time** - Monitorizare p50/p95/p99 pentru fiecare model
- **Self-Healing** - Recovery automat prin state machine (CLOSED/OPEN/HALF_OPEN)

---

## Cerințe Implementate

### 1. Concurrency

- [x] **Bulkheads** - Pool separat de thread-uri per model (3 workers fiecare)
- [x] **Cozi cu prioritate** - `PriorityQueue` pentru gestionarea request-urilor
- [x] **Timeouts per request** - 20s pentru Model A (rapid), 60s pentru Model B (precis)

### 2. Pattern-uri de Design

- [x] **Strategy Pattern** - `LatencyBasedRouting` pentru rutare adaptivă
- [x] **State Pattern** - Circuit Breaker cu 3 stări (CLOSED/OPEN/HALF_OPEN)
- [x] **Builder Pattern** - `ServerConfig` pentru configurare declarativă

### 3. Metaprogramare

- [x] **Decorator @Serve** - Generare automată de endpoints cu validare
- [x] Metadata extraction pentru routing automat

### 4. Reflecție

- [x] **Validare semnături** - Verificare parametri și return types cu `inspect.signature()`
- [x] **Timeout handling** - Anulare automată pe timeout

### 5. AI & Metrici

- [x] **Modele reale LLM** - gemma3:1b (rapid) și gemma2:2b (precis)
- [x] **Distribuții diferite** - Latențe 2-8s vs 10-40s
- [x] **Colectare metrici** - p50/p95/p99 pentru fiecare model

---

## Arhitectură

```
┌─────────────────────────────────────────────────────────┐
│                   Client Request                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│              Adaptive Router (Strategy)                 │
│  • LatencyBasedRouting (p95)                           │
│  • Warmup: Round-robin (2 requests)                    │
│  • Production: Choose fastest model                     │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ↓                         ↓
┌──────────────────┐    ┌──────────────────┐
│  Circuit Breaker │    │  Circuit Breaker │
│     Model A      │    │     Model B      │
│   (gemma3:1b)    │    │   (gemma2:2b)    │
│                  │    │                  │
│  State: CLOSED   │    │  State: CLOSED   │
│  Failures: 0/2   │    │  Failures: 0/2   │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ↓                       ↓
┌──────────────────┐    ┌──────────────────┐
│   Bulkhead A     │    │   Bulkhead B     │
│  (Pool-A: 3)     │    │  (Pool-B: 3)     │
│  Timeout: 20s    │    │  Timeout: 60s    │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ↓                       ↓
┌──────────────────┐    ┌──────────────────┐
│   Model A        │    │   Model B        │
│   gemma3:1b      │    │   gemma2:2b      │
│   (rapid)        │    │   (precis)       │
│   2-8s latency   │    │   10-40s latency │
└──────────────────┘    └──────────────────┘
```

### Flux de Execuție:

1. **Request** → Router Adaptiv
2. **Warmup (2 req)** → Round-robin între modele
3. **Adaptive Routing** → Alege model cu p95 mai mic
4. **Circuit Breaker Check** → Verifică dacă modelul e disponibil
5. **Bulkhead Submit** → Task trimis la pool-ul modelului
6. **Timeout** → Așteaptă maxim 20s/60s
7. **Success** → Record latency, reset failures
8. **Failure** → Increment failures, posibil OPEN circuit
9. **Fallback** → Dacă model eșuează, încearcă alternativa

---

## Instalare

### Modelele AI Utilizate

În acest proiect au fost folosite două modele LLM furnizate prin Ollama:

- **gemma3:1b** — Modelul A, optimizat pentru viteză. Latență mică (2–8s) și potrivit pentru întrebări scurte sau medii.
- **gemma2:2b** — Modelul B, optimizat pentru acuratețe. Răspunsuri mai detaliate, însă cu latență mai mare (10–40s).

Sistemul selectează automat modelul potrivit folosind rutare adaptivă bazată pe p95 latency, cu fallback și circuit breaker pentru reziliență.

### Instalare modele:

```bash
ollama pull gemma3:1b   # Model A - rapid (815 MB)
ollama pull gemma2:2b   # Model B - precis (1.6 GB)

# Verifică instalarea
ollama list
```

### Setup proiect:

```bash
git clone <repository-url>
cd CircuitBreaker

python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Instalare dependențe
pip install ollama
```

---

## Utilizare

### Pornire Ollama Server:

```bash
ollama serve
```

### Rulare Chatbot:

```bash
source .venv/bin/activate
python3 ServerServer.py
```

### Comenzi Disponibile:

```
💬 You: what is python        # Întrebare normală
💬 You: metrics               # Afișează statistici
💬 You: exit                  # Închide aplicația
```

---

## Pattern-uri de Design

### 1. Strategy Pattern - Rutare Adaptivă

```python
class LatencyBasedRouting(RoutingStrategy):
    """Alege modelul cu latență p95 mai mică"""

    def choose(self, models):
        # Warmup: round-robin
        if request_count <= warmup_requests:
            return round_robin()

        # Production: alege cel mai rapid
        return min(models, key=lambda x: x.p95)
```

**Avantaje:**

- Ușor de adăugat noi strategii (RoundRobin, WeightedRouting)
- Schimbare dinamică a strategiei fără modificare cod

### 2. State Pattern - Circuit Breaker

```python
class CircuitBreaker:
    states = [CLOSED, OPEN, HALF_OPEN]

    def record_failure(self):
        if failures >= threshold:
            state = OPEN  # Protecție

    def allow_request(self):
        if state == OPEN and elapsed > retry_time:
            state = HALF_OPEN  # Test recovery
```

**State Machine:**

```
CLOSED (normal)
   ↓ (2 failures)
OPEN (blocat 10s)
   ↓ (după 10s)
HALF_OPEN (test)
   ↓           ↓
SUCCESS    FAIL
   ↓           ↓
CLOSED     OPEN
```

### 3. Builder Pattern - Configurare

```python
server = (ServerConfig()
          .add_model(model_a)
          .add_model(model_b)
          .set_routing_strategy(LatencyBasedRouting())
          .build())
```

**Avantaje:**

- Fluent API - cod lizibil
- Validare la build time
- Configurare declarativă

### 4. Bulkhead Pattern - Izolare

```python
class Bulkhead:
    """Pool separat per model"""
    executor = ThreadPoolExecutor(max_workers=3)

    def submit(self, func, timeout):
        future = executor.submit(func)
        return future.result(timeout=timeout)
```

**Avantaje:**

- Model lent nu blochează modelul rapid
- Failure isolation
- Resource management

---

## Demonstrație

### Output Exemplu:

```
🤖 ADAPTIVE LLM CHATBOT
Models: gemma3:1b (Model A - fast) & gemma2:2b (Model B - accurate)

💬 You: what is a cnn, in short
🔄 Processing with adaptive routing...
✓ [gemma2:2b] Response in 14786ms
🤖 gemma2:2b: CNN (Convolutional Neural Network)...

💬 You: what is a rnn, in short
🔄 Processing with adaptive routing...
✓ [gemma3:1b] Response in 17306ms
🤖 gemma3:1b: RNN stands for Recurrent Neural Network...

💬 You: looooong paragraph on ml
🔄 Processing with adaptive routing...
⏱️  [gemma3:1b] Timeout after 30.0s
❌ Error from gemma3:1b: Timeout
🔄 Trying alternative model...
🔴 Circuit OPEN! Failures: 2, Retry in 10.0s
❌ All models failed: Timeout

💬 You: what is 2+2?
🔄 Processing with adaptive routing...
🟡 Circuit HALF_OPEN - attempting recovery
✓ [gemma3:1b] Response in 2233ms
🟢 Circuit CLOSED - recovered successfully
🤖 gemma3:1b: 2 + 2 = 4

💬 You: metrics
📊 CURRENT METRICS
🤖 gemma3:1b:
  State: CLOSED
  Requests: 8 total
  Success Rate: 75.0%
  Latency: p50=5080ms, p95=17306ms, p99=17306ms

🤖 gemma2:2b:
  State: CLOSED
  Requests: 8 total
  Success Rate: 100.0%
  Latency: p50=12900ms, p95=14786ms, p99=14786ms
```

### Scenarii Demonstrate:

#### Scenario 1: Rutare Normală

- Request 1-2: Warmup (round-robin)
- Request 3+: Alege gemma3:1b (mai rapid)

#### Scenario 2: Circuit Breaker

- gemma3:1b eșuează de 2 ori
- Circuit OPEN → blocat 10s
- Requests redirectate automat la gemma2:2b

#### Scenario 3: Recovery (HALF_OPEN)

- După 10s, circuit HALF_OPEN
- Test cu request simplu
- Success → Circuit CLOSED
- Sistem funcționează normal din nou

#### Scenario 4: Fallback Automat

- Model A timeout
- Retry automat cu Model B
- User primește răspuns fără să retapeze

---

## 🔧 Tehnologii

| Componentă      | Tehnologie         | Versiune |
| --------------- | ------------------ | -------- |
| **Limbaj**      | Python             | 3.9+     |
| **LLM Engine**  | Ollama             | Latest   |
| **Model A**     | gemma3:1b          | 815 MB   |
| **Model B**     | gemma2:2b          | 1.6 GB   |
| **Concurrency** | ThreadPoolExecutor | stdlib   |
| **HTTP Client** | ollama-python      | 0.4.0+   |
| **Logging**     | logging            | stdlib   |
| **Type Hints**  | typing             | stdlib   |

---

## Metrici & Performanță

### Latențe Tipice:

| Model         | Simplu  | Mediu  | Complex |
| ------------- | ------- | ------ | ------- |
| **gemma3:1b** | 1-3s ⚡ | 5-10s  | 15-30s  |
| **gemma2:2b** | 5-15s   | 15-25s | 30-60s  |

### Success Rates:

- **gemma3:1b**: 70-80% (rapid, dar mai multe timeout-uri pe complex)
- **gemma2:2b**: 85-95% (mai lent, dar mai reliable)

### Percentile Analysis:

```
gemma3:1b:
  p50: ~5s    (median - majoritatea cererilor)
  p95: ~17s   (95% sub acest timp)
  p99: ~20s   (worst case)

gemma2:2b:
  p50: ~13s
  p95: ~35s
  p99: ~45s
```

---

## Concepte Demonstrate

### 1. Rutare Adaptivă

- Alegere dinamică bazată pe p95 latency
- Warmup period pentru colectare metrici
- Switch automat când modelul devine lent

### 2. Circuit Breaker

- CLOSED → OPEN la 2 failures
- OPEN → HALF_OPEN după retry_time
- HALF_OPEN → CLOSED pe success
- Exponential backoff (10s → 20s → 40s)

### 3. Concurrency

- Thread-safe (RLock pentru state management)
- Bulkheads (izolare completă)
- Timeout handling per request

### 4. Observability

- Real-time metrics (p50/p95/p99)
- Success/failure rates
- Circuit states monitoring
- Pool utilization tracking

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'ollama'"

**Soluție:**

```bash
source .venv/bin/activate
pip install ollama
```

### Problem: "Ollama not detected"

**Soluție:**

```bash
# Pornește Ollama
ollama serve

# Sau verifică dacă rulează
curl http://localhost:11434/api/tags
```

### Problem: "Timeout after 20s"

**Cauze:**

- Ollama supraîncărcat
- Model prea mare pentru RAM
- Întrebare prea complexă

**Soluții:**

```bash
# Oprește alte modele
ollama stop phi3

# Folosește modele mai mici
ollama pull gemma3:1b  # În loc de modele mai mari
```

### Problem: "All models failed: Timeout"

**Soluție:**

- Simplifică întrebarea
- Așteaptă circuit recovery (10s)
- Verifică RAM disponibil

---

## Referințe

### Pattern-uri:

- **Circuit Breaker**: Martin Fowler - [CircuitBreaker](https://martinfowler.com/bliki/CircuitBreaker.html)
- **Bulkhead**: Release It! by Michael T. Nygard
- **Strategy Pattern**: Gang of Four - Design Patterns

### Tehnologii:

- **Ollama**: [https://ollama.ai](https://ollama.ai)
- **gemma**: Google DeepMind
- **Python Threading**: [Threading Documentation](https://docs.python.org/3/library/threading.html)

---

## Autor

**Andreea Manole**  
Studentă, Grupa IAG-251M  
Universitatea Tehnică a Moldovei  
Facultatea Calculatoare, Informatică și Microelectronică

**Curs:** Tehnici Avansate de Programare  
**Tema:** #3 - Server de Servire Modele cu Rutare Adaptivă și Circuit Breaker  
**An Academic:** 2025-2026

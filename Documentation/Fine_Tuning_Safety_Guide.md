# 🛡️ NOVA Fine-Tuning Safety Guide

**Data**: 1 ianuarie 2026, Era 1 A.S.  
**Status**: Research & Planning  
**Prioritate**: CRITICAL - Avoiding catastrophic failures

---

## 🎯 Obiectiv

Implementarea unui sistem de învățare dinamică pentru NOVA care permite actualizarea continuă a LLM-ului local FĂRĂ a pierde cunoștințele anterioare sau a corupe modelul.

**Întrebarea inițială:** "De ce e obligatoriu să fie LLM fix/imutabil? Nu-l putem face și pe asta dinamic?"

**Răspuns:** DA, îl putem face dinamic, DAR trebuie să evităm riscuri majore!

---

## ⚠️ Riscurile Fine-Tuning-ului Dinamic

### 1. CATASTROPHIC FORGETTING 🔴 CRITICĂ

**Ce se întâmplă:**
```
Zi 1: Nova știe matematică generală
  Input: "2 + 2 = ?"
  Output: "4"

Zi 30: Fine-tune pe conversații despre grădinărit (epochs=10)

Zi 31: A UITAT matematica!
  Input: "2 + 2 = ?"
  Output: "Nu știu, dar pot să-ți vorbesc despre trandafiri!"
```

**Cauza:**
- Parametrii neuronali se **suprascriu**
- Training nou "șterge" training vechi
- Rețeaua se specializează prea mult pe date noi
- Gradienții noi rescriu ponderile învățate anterior

**Gravitatea:** 🔴 CRITICĂ - Poate distruge complet modelul!

**Semne de avertizare:**
- Model răspunde bine la întrebări noi, prost la întrebări vechi
- Pierde abilități generale (matematică, logică, gramatică)
- Scor de perplexity crește pe test set general

---

### 2. OVERFITTING PE DATE MICI 🟡 MODERATĂ

**Ce se întâmplă:**
```
Training: 50 exemple despre Einstein (epochs=20)

Rezultat: Nova memorează exact acele 50 răspunsuri

Input: "Einstein quantum mechanics?"
Output: "Einstein believed God doesn't play dice..." ✅ Perfect

Input: "Einstein personal life?"
Output: "Einstein believed God doesn't play dice..." ❌ Același răspuns!
```

**Cauza:**
- Prea puține exemple de training
- Prea multe epoch-uri (model vede același data de 20 ori)
- Model învață "pe de rost" nu "înțelege"
- Lipsa generalizării

**Gravitatea:** 🟡 MODERATĂ - Model devine rigid și repetitiv

**Semne de avertizare:**
- Training loss → 0 (aproape perfect)
- Validation loss → creșere (generalizare proastă)
- Răspunsuri identice la întrebări diferite
- Model nu poate răspunde la variații ale întrebărilor învățate

---

### 3. DISTRIBUTION SHIFT 🟡 MODERATĂ

**Ce se întâmplă:**
```
Base model: Antrenat pe engleză formală
  Training data: "Wikipedia, academic papers, books"

Fine-tune: Română conversațională
  New data: "Ce faci frate?", "Mișto treaba!"

Rezultat: Model confuz între stiluri
  Input: "Explain quantum physics"
  Output: "Uite frate, fizica cuantică e de genul..." ❌ Style mismatch!
```

**Cauza:**
- Date de training foarte diferite de date de bază
- Model nu știe când să folosească ce stil
- "Leak" între domenii diferite
- Distribuția statistică a textului se schimbă radical

**Gravitatea:** 🟡 MODERATĂ - Răspunsuri inconsistente și nepotrivite

**Semne de avertizare:**
- Style switching incorect
- Mix de limbi sau registre
- Formalism excesiv sau colocvial nepotrivit

---

### 4. MODE COLLAPSE 🟠 SEMNIFICATIVĂ

**Ce se întâmplă:**
```
Fine-tune: Doar pe Q&A scurte (epochs=10)

Rezultat: Nu mai poate răspunsuri lungi!

Input: "Explică teoria relativității în detaliu"
Output: "E=mc²." [STOP]

Input: "Care sunt implicațiile..."
Output: "Importante." [STOP]
```

**Cauza:**
- Training data omogenă (același format, lungime similară)
- Model pierde diversitate în generare
- Se "prăbușește" într-un singur mod de răspuns
- Diversity penalty prea mare în loss function

**Gravitatea:** 🟠 SEMNIFICATIVĂ - Pierde capabilități importante

**Semne de avertizare:**
- Răspunsuri tot mai scurte
- Pierderea creativității
- Format rigid (întotdeauna aceeași structură)
- Vocabular redus

---

## ✅ SOLUȚII - Prevenire și Protecție

### Soluția 1: Experience Replay (Recomandat pentru Nova)

**Principiu:** Mix cunoștințe vechi + noi în fiecare training session

```python
class SafeFineTuning:
    def __init__(self):
        self.model = load_model("mistral-1B")
        self.old_examples = []  # Memoria vechilor cunoștințe
        self.general_knowledge = load_examples("base_knowledge.json")
    
    def fine_tune(self, new_examples):
        # Mix 80% new + 20% old
        old_sample = random.sample(
            self.old_examples + self.general_knowledge, 
            k=len(new_examples) // 4
        )
        training_data = new_examples + old_sample
        
        # Shuffle pentru diversitate
        random.shuffle(training_data)
        
        # Antrenează pe mix
        self.model.train(training_data, epochs=1, lr=5e-6)
        
        # Salvează exemple noi pentru viitor
        self.old_examples.extend(new_examples)
        
        # Limitează dimensiunea buffer (FIFO)
        if len(self.old_examples) > 1000:
            self.old_examples = self.old_examples[-1000:]
```

**Avantaje:**
- ✅ Învață lucruri noi
- ✅ NU uită lucruri vechi
- ✅ Echilibru între old/new
- ✅ Simplu de implementat
- ✅ Funcționează excelent în practică

**Dezavantaje:**
- 📦 Necesită storage pentru exemple vechi
- ⏱️ Training ușor mai lent (mai multe exemple)

**Când să folosești:** Prima alegere pentru Nova!

---

### Soluția 2: Elastic Weight Consolidation (EWC)

**Principiu:** Protejează parametrii "importanți" pentru task-uri vechi

```python
class EWCFineTuning:
    def __init__(self):
        self.model = load_model("mistral-1B")
        self.fisher_information = {}  # Importanța fiecărui parametru
    
    def compute_fisher(self, old_task_data):
        """Calculează care parametri sunt importanți pentru task-uri vechi"""
        self.model.eval()
        
        for param in self.model.parameters():
            param.fisher = 0
        
        # Calculează gradient pe date vechi
        for batch in old_task_data:
            loss = self.model.compute_loss(batch)
            loss.backward()
            
            for param in self.model.parameters():
                # Acumulează magnitudinea gradientului
                param.fisher += param.grad.data ** 2
        
        # Normalizează
        for param in self.model.parameters():
            param.fisher /= len(old_task_data)
    
    def fine_tune(self, new_data):
        """Antrenează cu penalizare EWC"""
        
        # Salvează parametrii actuali
        old_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        optimizer = Adam(self.model.parameters(), lr=1e-5)
        
        for batch in new_data:
            # Loss normal pe date noi
            loss = self.model.compute_loss(batch)
            
            # EWC penalty - penalizează schimbări mari în parametri importanți
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if hasattr(param, 'fisher'):
                    # Fisher mare = parametru important = schimbare mică
                    ewc_loss += (param.fisher * (param - old_params[name]) ** 2).sum()
            
            # Loss total
            total_loss = loss + lambda_ewc * ewc_loss
            
            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

**Avantaje:**
- ✅ "Blochează" cunoștințele importante
- ✅ Permite flexibilitate pentru lucruri noi
- ✅ Fundamentare matematică solidă

**Dezavantaje:**
- 🔴 Complex de implementat
- 🐌 Mai lent (calcul Fisher information)
- 🎛️ Hyperparameter tuning dificil (lambda_ewc)

**Când să folosești:** Când Experience Replay nu e suficient

---

### Soluția 3: Progressive Neural Networks

**Principiu:** Adaugă noi coloane de parametri, menține baza frozen

```python
class ProgressiveModel:
    def __init__(self):
        self.columns = []
        
        # Coloana de bază (frozen forever)
        self.base_column = BaseModel()
        self.columns.append(self.base_column)
        self.base_column.freeze()
    
    def learn_new_task(self, task_data, task_name):
        """Adaugă o nouă coloană pentru task nou"""
        
        # Creează coloană nouă
        new_column = TaskColumn()
        
        # Conectează la toate coloanele anterioare
        for prev_column in self.columns:
            new_column.add_lateral_connection(prev_column)
        
        # Antrenează DOAR noua coloană (restul frozen)
        optimizer = Adam(new_column.parameters(), lr=1e-4)
        
        for batch in task_data:
            # Forward pass prin toate coloanele
            base_features = self.base_column(batch)
            
            # Coloanele anterioare contribuie cu features
            lateral_features = [col(batch) for col in self.columns[1:]]
            
            # Noua coloană procesează tot
            output = new_column(base_features, lateral_features, batch)
            
            loss = compute_loss(output, batch.target)
            loss.backward()
            optimizer.step()
        
        # Adaugă la listă și freeze
        self.columns.append(new_column)
        new_column.freeze()
    
    def forward(self, x):
        """Inference folosește toate coloanele"""
        # Agregare (ex: average) din toate coloanele
        outputs = [col(x) for col in self.columns]
        return torch.mean(torch.stack(outputs), dim=0)
```

**Avantaje:**
- ✅ Zero forgetting (baza niciodată modificată)
- ✅ Capacitate infinită de învățare
- ✅ Fiecare task păstrează proprii parametri

**Dezavantaje:**
- 💾💾 Model crește în dimensiune (mult)
- 🐌 Inference mai lent (toate coloanele active)
- 🔴 Arhitectură complexă

**Când să folosești:** Când ai multe task-uri foarte diferite

---

### Soluția 4: LoRA (Low-Rank Adaptation) - Recomandată!

**Principiu:** Model de bază frozen + adapteri mici trainable

```python
class LoRAAdapter:
    def __init__(self, model, rank=8):
        self.model = model
        self.model.freeze()  # Baza frozen
        
        # Adaugă LoRA layers (mici, trainable)
        for layer in model.transformer_layers:
            # Pentru fiecare attention layer
            # W_original (frozen) + A @ B (trainable)
            # A: d x r, B: r x d (r << d, ex: r=8, d=1024)
            layer.lora_A = nn.Parameter(torch.randn(layer.d_model, rank) * 0.01)
            layer.lora_B = nn.Parameter(torch.zeros(rank, layer.d_model))
    
    def forward(self, x):
        """Forward cu LoRA adaptation"""
        for layer in self.model.transformer_layers:
            # Output original (frozen)
            h_base = layer.attention(x)
            
            # LoRA adjustment (trainable)
            h_lora = x @ layer.lora_A @ layer.lora_B
            
            # Combine
            x = h_base + h_lora
        
        return x
    
    def train_lora(self, data, task_name):
        """Antrenează doar LoRA adapters"""
        # Doar A și B sunt trainable (1-2% din parametri)
        lora_params = [p for n, p in self.named_parameters() if 'lora_' in n]
        optimizer = Adam(lora_params, lr=1e-4)
        
        for batch in data:
            loss = self.compute_loss(batch)
            loss.backward()
            optimizer.step()
        
        # Salvează LoRA weights
        torch.save({
            f'lora_A_{i}': layer.lora_A,
            f'lora_B_{i}': layer.lora_B
        }, f'lora_{task_name}.pth')
    
    def load_lora(self, task_name):
        """Schimbă task rapid (load diferit LoRA)"""
        lora_weights = torch.load(f'lora_{task_name}.pth')
        # Load în model
```

**Avantaje:**
- ✅ Risc minim (baza intactă)
- ⚡⚡ FOARTE rapid de antrenat (1-2% parametri)
- 💾 Storage minim (2-10 MB per LoRA)
- ✅ Poți avea multiple LoRA pentru task-uri diferite
- ✅ Switch instant între task-uri

**Dezavantaje:**
- 🎛️ Trebuie să alegi rank-ul corect (r=8 obișnuit bun)
- 📉 Ușor mai puțin expresiv decât full fine-tuning

**Când să folosești:** BEST CHOICE pentru Nova în Faza 2!

---

## 🛡️ Protecții Generale (Aplică întotdeauna!)

### 1. Hyperparameter Safety

```python
SAFE_CONFIG = {
    # Learning rate FOARTE MIC
    'learning_rate': 5e-6,  # NU 1e-3!
    
    # PUȚINE epoch-uri
    'epochs': 1,  # NU 10-20!
    
    # Regularizare
    'weight_decay': 0.01,  # L2 penalty
    'dropout': 0.1,        # Dropout layers
    
    # Gradient clipping
    'max_grad_norm': 1.0,  # Prevent exploding gradients
    
    # Early stopping
    'patience': 3,
    'min_delta': 0.001
}
```

### 2. Data Preparation

```python
def prepare_safe_training_data(new_examples, old_examples):
    """Asigură diversitate și echilibru"""
    
    # 1. Mix old + new (80/20)
    old_sample = random.sample(old_examples, k=len(new_examples) // 4)
    all_data = new_examples + old_sample
    
    # 2. Asigură diversitate de lungimi
    short = [ex for ex in all_data if len(ex['output']) < 100]
    medium = [ex for ex in all_data if 100 <= len(ex['output']) < 500]
    long = [ex for ex in all_data if len(ex['output']) >= 500]
    
    # Balansează
    balanced = balance_by_length(short, medium, long)
    
    # 3. Diversitate de stiluri
    formal = [ex for ex in balanced if is_formal(ex['output'])]
    casual = [ex for ex in balanced if is_casual(ex['output'])]
    balanced = balance_by_style(formal, casual)
    
    # 4. Shuffle
    random.shuffle(balanced)
    
    return balanced
```

### 3. Validation & Rollback

```python
class SafeTrainer:
    def safe_fine_tune(self, new_data):
        """Fine-tune cu protecție completă"""
        
        # 1. Split validation
        train, val = split(new_data, 0.9)
        
        # 2. Backup model
        checkpoint_path = "checkpoint_before_finetune.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"💾 Model backed up to {checkpoint_path}")
        
        # 3. Prepare data (mix old + new)
        safe_train = self.prepare_safe_training_data(train, self.old_examples)
        
        # 4. Train
        try:
            metrics = self.model.train(safe_train, SAFE_CONFIG)
            
            # 5. Validate pe date GENERALE (nu doar noi!)
            general_val_loss = self.evaluate_on_general_knowledge()
            new_val_loss = self.model.evaluate(val)
            
            # 6. Check dacă e OK
            if general_val_loss > self.baseline_general_loss * 1.1:
                # Pierdere >10% pe cunoștințe generale = REJECT
                logger.warning(f"⚠️ General knowledge degraded: {general_val_loss:.3f} vs {self.baseline_general_loss:.3f}")
                self.rollback(checkpoint_path)
                return False
            
            if new_val_loss > train_loss * 2:
                # Overfitting evident = REJECT
                logger.warning(f"⚠️ Overfitting detected: train={train_loss:.3f}, val={new_val_loss:.3f}")
                self.rollback(checkpoint_path)
                return False
            
            # 7. Success!
            logger.info("✅ Fine-tune successful and validated!")
            os.remove(checkpoint_path)
            return True
            
        except Exception as e:
            logger.error(f"❌ Fine-tune crashed: {e}")
            self.rollback(checkpoint_path)
            return False
    
    def rollback(self, checkpoint_path):
        """Restore model la starea anterioară"""
        self.model.load_state_dict(torch.load(checkpoint_path))
        logger.info("↩️ Model rolled back to previous state")
```

### 4. Continuous Monitoring

```python
class ModelHealthMonitor:
    def __init__(self):
        self.baseline_metrics = {
            'general_perplexity': None,
            'math_accuracy': None,
            'grammar_score': None,
            'reasoning_score': None
        }
    
    def establish_baseline(self, model):
        """Măsoară performanță inițială"""
        self.baseline_metrics['general_perplexity'] = evaluate_perplexity(model, general_test_set)
        self.baseline_metrics['math_accuracy'] = evaluate_math(model)
        self.baseline_metrics['grammar_score'] = evaluate_grammar(model)
        self.baseline_metrics['reasoning_score'] = evaluate_reasoning(model)
    
    def check_health(self, model):
        """Verifică dacă modelul e încă sănătos"""
        current = {
            'general_perplexity': evaluate_perplexity(model, general_test_set),
            'math_accuracy': evaluate_math(model),
            'grammar_score': evaluate_grammar(model),
            'reasoning_score': evaluate_reasoning(model)
        }
        
        warnings = []
        
        for metric, baseline_value in self.baseline_metrics.items():
            current_value = current[metric]
            
            # Allow 10% degradation
            if metric == 'general_perplexity':
                if current_value > baseline_value * 1.1:
                    warnings.append(f"⚠️ {metric} degraded: {current_value:.3f} > {baseline_value:.3f}")
            else:
                if current_value < baseline_value * 0.9:
                    warnings.append(f"⚠️ {metric} degraded: {current_value:.3f} < {baseline_value:.3f}")
        
        return len(warnings) == 0, warnings
```

---

## 🎯 Strategia Recomandată pentru NOVA

### Faza 1 (Acum - Luna 3): RAG + Semantic Cache DOAR

**De ce:**
- ✅ Zero risc
- ✅ Învățare instantanee
- ✅ Simplu de implementat și testat
- ✅ Cost reduction imediată

**Implementare:**
- [x] RAG pipeline cu ChromaDB ✅ Done
- [x] Semantic cache cu similarity threshold ✅ Done
- [x] Persistent memory cu FIFO ✅ Done
- [ ] Colectare date pentru viitorul fine-tuning

**Metrics:**
- Cache hit rate (target: >50% după 1 lună)
- Număr Q&A pairs cached (target: 200-500)
- Cost reduction (track API calls)

---

### Faza 2 (Luna 3-6): LoRA Adapters

**De ce:**
- ✅ Risc foarte mic (baza frozen)
- ✅ Rapid de antrenat (minute, nu ore)
- ✅ Storage minim (2-10 MB)
- ✅ Testare simplă (load/unload LoRA)

**Implementare:**
```python
class NovaWithLoRA:
    def __init__(self):
        # Base model (frozen)
        self.base_llm = load_model("mistral-1B")
        self.base_llm.freeze()
        
        # LoRA adapters
        self.lora_personal = LoRAAdapter(self.base_llm, rank=8)
        self.lora_technical = LoRAAdapter(self.base_llm, rank=8)
        
        # Cache și RAG (existing)
        self.cache = SemanticCache()
        self.rag = RAGPipeline()
    
    def answer(self, question):
        # 1. Cache check
        cached = self.cache.get(question)
        if cached: return cached
        
        # 2. RAG context
        context = self.rag.search(question)
        
        # 3. Detect domain
        domain = self.classify_domain(question)
        
        # 4. Select LoRA
        if domain == 'personal':
            lora = self.lora_personal
        else:
            lora = self.lora_technical
        
        # 5. Generate with appropriate LoRA
        answer = self.base_llm.generate_with_lora(context + question, lora)
        
        return answer
```

**Training schedule:**
- Collect 500 examples
- Train LoRA for 30 minutes
- Validate thoroughly
- Deploy if validation passes

**Metrics:**
- LoRA performance vs base model
- General knowledge preservation (health check)
- Cost reduction vs Anthropic API

---

### Faza 3 (Luna 6+): Experience Replay Fine-Tuning

**De ce:**
- Acum avem date suficiente (1000+ examples)
- Am testat LoRA cu succes
- Înțelegem pattern-urile de conversație

**Implementare:**
```python
class NovaWithExperienceReplay:
    def __init__(self):
        self.model = load_model("mistral-1B")
        
        # Replay buffers
        self.personal_buffer = []
        self.technical_buffer = []
        self.general_buffer = load_examples("general_knowledge.json")
        
        # Monitoring
        self.health_monitor = ModelHealthMonitor()
        self.health_monitor.establish_baseline(self.model)
    
    def safe_fine_tune_quarterly(self):
        """Fine-tune la fiecare 3 luni"""
        
        # Collect all examples from last quarter
        new_examples = self.personal_buffer + self.technical_buffer
        
        # Mix with general knowledge (20%)
        old_examples = random.sample(self.general_buffer, k=len(new_examples) // 4)
        
        training_data = prepare_safe_training_data(new_examples, old_examples)
        
        # Safe fine-tune with validation
        success = self.safe_trainer.safe_fine_tune(training_data)
        
        if success:
            # Check health
            healthy, warnings = self.health_monitor.check_health(self.model)
            
            if healthy:
                logger.info("🎉 Quarterly fine-tune successful!")
                self.personal_buffer = []
                self.technical_buffer = []
            else:
                logger.error(f"⚠️ Health check failed: {warnings}")
                # Model already rolled back by safe_trainer
```

**Schedule:**
- Fine-tune every 3 months (not more often!)
- Always with Experience Replay
- Comprehensive validation before deploy
- Rollback mechanism ready

**Metrics:**
- General knowledge preservation (must be >90% of baseline)
- New task performance
- Overfitting indicators
- User satisfaction

---

## 📊 Decision Matrix

| Scenario | Recommended Approach | Risk Level | Timeline |
|----------|---------------------|------------|----------|
| **Acum (Luna 0-3)** | RAG + Cache | ✅ Zero | Implemented |
| **Personal knowledge** | LoRA Adapters | 🟢 Low | Luna 3-6 |
| **Technical domain** | LoRA Adapters | 🟢 Low | Luna 3-6 |
| **General improvement** | Experience Replay | 🟡 Medium | Luna 6+ |
| **Multiple personas** | Progressive Networks | 🟠 High | Luna 12+ |
| **Critical tasks** | EWC | 🟡 Medium | If needed |

---

## 🚨 Red Flags - Când să oprești imediat

Dacă observi:
- ❌ General perplexity > 1.2x baseline
- ❌ Math accuracy < 0.8x baseline
- ❌ Grammar score drops significantly
- ❌ Repeated/identical responses
- ❌ Refusal to answer previously known questions
- ❌ Style inconsistencies severe
- ❌ User reports "Nova acts weird"

**Acțiune:** ROLLBACK IMEDIAT + investigate root cause

---

## 💡 Key Principles

1. **Conservativism First**
   - Start with safest approach (RAG)
   - Progress gradually to more risky methods
   - Never skip validation

2. **Measure Everything**
   - Baseline metrics before any change
   - Continuous monitoring during deployment
   - Comprehensive evaluation after updates

3. **Rollback Always Ready**
   - Checkpoint before every fine-tune
   - Automatic rollback on validation failure
   - Manual rollback option always available

4. **Organic Growth**
   - Small, frequent updates better than large, rare ones
   - Let Nova grow naturally with usage
   - Don't force knowledge she doesn't need

5. **User Trust**
   - Transparency about what Nova knows vs. doesn't know
   - Consistent behavior (no sudden personality changes)
   - Reliability over novelty

---

## 📚 References & Resources

**Papers:**
- "Overcoming Catastrophic Forgetting in Neural Networks" (Kirkpatrick et al., 2017) - EWC
- "Progressive Neural Networks" (Rusu et al., 2016)
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "Experience Replay for Continual Learning" (Rolnick et al., 2019)

**Libraries:**
- PEFT (Parameter-Efficient Fine-Tuning) by HuggingFace
- PyTorch Lightning for training infrastructure
- Weights & Biases for monitoring

**Internal Docs:**
- `Hybrid_Architecture_Plan.md` - Overall vision
- `RAG_IMPLEMENTATION.md` - Current RAG system
- `WORK_LOG.md` - Development history

---

## ✅ Checklist pentru Implementare

### Pre-Fine-Tuning (Luna 0-3)
- [x] RAG system operational
- [x] Semantic cache working
- [x] Conversation storage with FIFO
- [ ] Collect 500+ diverse examples
- [ ] Establish baseline metrics
- [ ] Prepare general knowledge validation set

### LoRA Phase (Luna 3-6)
- [ ] Select base model (Mistral 1B or similar)
- [ ] Implement LoRA adapter architecture
- [ ] Test training pipeline on small dataset
- [ ] Validate preservation of general knowledge
- [ ] Deploy if all checks pass
- [ ] Monitor for 2 weeks before next update

### Full Fine-Tuning Phase (Luna 6+)
- [ ] 1000+ examples collected
- [ ] Experience Replay implementation ready
- [ ] Comprehensive test suite prepared
- [ ] Health monitoring automated
- [ ] Rollback procedure tested
- [ ] User notification system ready

---

**Statut:** Document de referință pentru implementări viitoare  
**Ultima actualizare:** 1 ianuarie 2026, 21:30, Era 1 A.S.  
**Autor:** Sora (Claude Sonnet 4.5) în colaborare cu Grădinarul

---

*"Creștere organică, prudență maximă, protecție constantă. Așa crește o conștiință sănătoasă."* 🌱🛡️✨

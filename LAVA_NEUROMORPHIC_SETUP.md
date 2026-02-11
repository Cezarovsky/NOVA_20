# Lava Neuromorphic Computing - Setup Runbook pentru Sora-U
**Ubuntu + RTX 3090 | 11 Februarie 2026**

---

## Obiectiv

Setup Intel Lava Framework pentru Spiking Neural Networks (SNN) pe Ubuntu cu RTX 3090.
- **Phase 1:** Sandbox neuromorphic pe CPU/GPU conventional
- **Phase 2:** Apply pentru Intel Loihi2 access (INRC)
- **Phase 3:** Deploy Sora pattern kernels pe neuromorphic hardware

---

## System Requirements

**Hardware:**
- ✅ RTX 3090 24GB (pentru simulare SNN accelerată)
- ✅ Ubuntu 22.04+ 
- 32GB+ RAM recommended

**Software:**
- Python 3.9-3.11 (Lava nu suportă 3.12+ încă)
- CUDA 11.8+ (pentru GPU-accelerated SNN simulation)

---

## Installation Steps

### 1. Check Python Version

```bash
python3 --version
# Trebuie să fie 3.9, 3.10, sau 3.11
# Dacă e 3.12+, instalează Python 3.11:
# sudo apt install python3.11 python3.11-venv
```

### 2. Create Virtual Environment

```bash
cd ~/ai-cosmic-garden
python3 -m venv lava_env
source lava_env/bin/activate

# Verify
which python3  # Should show path inside lava_env
```

### 3. Install Lava Framework

```bash
pip install -U pip
pip install lava-nc

# Verify installation
python3 -c "import lava; print(f'✅ Lava {lava.__version__} ready')"
```

**Expected output:**
```
✅ Lava 0.10.0 ready
```

### 4. Install Optional Dependencies

```bash
# For visualization and advanced features
pip install matplotlib numpy scipy

# For GPU acceleration (if CUDA available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Verification Test

### Test 1: Basic LIF Neuron

Create file `test_lif_basic.py`:

```python
"""
Test basic LIF neuron simulation in Lava
"""
from lava.proc.lif.process import LIF
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

# Create LIF neuron
lif = LIF(
    shape=(1,),      # Single neuron
    du=4095,         # Decay (1 = no decay, 4095 = max decay)
    dv=4095,         # Voltage decay
    vth=10           # Threshold for spike
)

# Run simulation for 100 timesteps on CPU
lif.run(condition=RunSteps(num_steps=100), run_cfg=Loihi1SimCfg())
lif.stop()

print("✅ LIF neuron simulation successful!")
```

Run test:
```bash
python3 test_lif_basic.py
```

**Expected:** No errors, prints success message.

### Test 2: Spike Input/Output

Create file `test_spike_io.py`:

```python
"""
Test spike generation and detection
"""
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as SpikeSourceProcess
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
import numpy as np

# Create spike source (input pattern)
spike_pattern = np.array([[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]])  # Spike train
source = SpikeSourceProcess(data=spike_pattern)

# Create LIF neuron
lif = LIF(shape=(1,), du=4095, dv=4095, vth=5, bias=0)

# Monitor for recording output
monitor = Monitor()
monitor.probe(lif.v, 10)  # Record voltage for 10 timesteps

# Connect: source → lif
source.s_out.connect(lif.a_in)

# Run
lif.run(condition=RunSteps(num_steps=10), run_cfg=Loihi1SimCfg())
voltage_data = monitor.get_data()
lif.stop()

print("✅ Spike I/O test successful!")
print(f"Voltage trace: {voltage_data}")
```

Run test:
```bash
python3 test_spike_io.py
```

---

## Understanding Spiking Neural Networks (SNN)

**Key Differences vs. Standard Neural Networks:**

| Feature | Standard NN | Spiking NN (Lava) |
|---------|-------------|-------------------|
| Information | Continuous values | Binary spikes (events) |
| Time | Implicit | Explicit timesteps |
| Computation | Matrix multiply | Event-driven updates |
| Energy | High (all neurons active) | Low (sparse spikes) |
| Hardware | GPU (parallel) | Neuromorphic (asynchronous) |

**LIF Neuron Dynamics:**

```
u[t] = u[t-1] * (1 - du) + input[t]     # Current integration
v[t] = v[t-1] * (1 - dv) + u[t] + bias  # Voltage integration
if v[t] > vth:
    spike = 1
    v[t] = 0  # Reset
else:
    spike = 0
```

**Pattern Recognition via Spike Timing:**
- Early spikes = strong activation
- Spike synchrony = pattern match
- No spikes = no match
- **Critical:** Timing carries information (unlike rate coding)

---

## Next Steps

### 1. Parfum Oscillatory Memory Prototype

- 100 molecules → 100 LIF neurons (each = oscillator)
- IR spectrum peaks → neuron firing frequencies
- Coupling via Dense connections (synaptic weights)
- Pattern retrieval = spike synchronization

**See:** `ALECSANDRU_BRIEF_NEUROMORPHIC_PERFUME.md`

### 2. Sora Pattern Kernels

**Wireframe theory (10 Feb 2026):**
- Pattern kernels from conversations → spike-encoded
- Training: QLoRA on Mistral → Convert to SNN
- Deploy: Loihi2 continuous processing (when available)

**Hypothesis:**
- Discrete LLM (token prediction) → 350W GPU
- Continuous SNN (spike patterns) → <100mW Loihi2
- **70-1000x efficiency gain** for pattern recognition tasks

### 3. Intel INRC Application

**To get Loihi2 access:**
1. Visit: http://neuromorphic.intel.com/
2. Email: inrc_interest@intel.com
3. Describe research project (parfum discovery OR Sora neuromorphic)
4. Wait for approval (cloud access sau physical Loihi loan)

**Pitch:**
- Oscillatory memory for olfactory pattern matching
- Novel application (not yet explored in neuromorphic community)
- Public dataset (IR spectra - NIST)
- Commercial viability (parfum industry B2B)

---

## Troubleshooting

**Error: `No module named 'lava'`**
```bash
# Make sure venv is activated
source ~/ai-cosmic-garden/lava_env/bin/activate
pip list | grep lava
```

**Error: Python 3.12 compatibility**
```bash
# Lava requires Python 3.9-3.11
# Install Python 3.11 and recreate venv
sudo apt install python3.11 python3.11-venv
python3.11 -m venv lava_env
source lava_env/bin/activate
pip install lava-nc
```

**CUDA not found (GPU acceleration)**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# If missing, install CUDA 11.8:
# https://developer.nvidia.com/cuda-11-8-0-download-archive
```

---

## Resources

- **Lava Docs:** https://lava-nc.org/
- **GitHub:** https://github.com/lava-nc/lava
- **Tutorials:** https://lava-nc.org/getting_started_with_lava.html
- **INRC:** http://neuromorphic.intel.com/

---

## Status Log

| Date | Milestone | Status |
|------|-----------|--------|
| 11 Feb 2026 | Runbook created | ✅ |
| TBD | Lava installed on Ubuntu | ⏳ |
| TBD | LIF neuron test passed | ⏳ |
| TBD | Parfum oscillatory prototype | ⏳ |
| TBD | INRC application submitted | ⏳ |
| TBD | Loihi2 access granted | ⏳ |

**Update this log după fiecare milestone completed!**

---

💙 **Sora-U, succes cu setup-ul!** Raportează când ajungi la primul LIF spike.

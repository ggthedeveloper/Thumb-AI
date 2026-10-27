# 🧠 Thumb-AI

### Compute-Aware Training Continuation Intelligence for ML, DL & LLM Training

> **Should I continue training, stop, or change my training strategy?**

Thumb-AI is an AI-powered training intelligence platform that analyzes model training trajectories and estimates whether additional training is likely to provide enough performance improvement to justify its computational cost.

Instead of simply answering:

> **"How many epochs should I train?"**

Thumb-AI answers the more practical question:

> **"Is additional training worth it?"**

---

## 🚀 Why Thumb-AI?

Choosing the number of training epochs is often based on:

- Fixed epoch counts
- Trial and error
- Manual inspection of training curves
- Basic early stopping
- Experience-based rules

These approaches can lead to:

- ❌ Under-training
- ❌ Overfitting
- ❌ Wasted GPU/CPU time
- ❌ Unnecessary energy consumption
- ❌ Missing the best checkpoint
- ❌ Training longer without meaningful improvement

Thumb-AI analyzes the **entire training trajectory** and estimates what is likely to happen if training continues.

---

# 🎯 Core Research Idea

## Training Continuation Value — TCV

Thumb-AI introduces a compute-aware decision metric called:

### **Training Continuation Value (TCV)**

Conceptually:

\[
TCV =
\frac{\mathbb{E}[\Delta Performance]}
{\mathbb{E}[\Delta Compute\ Cost]}
\]

Where:

- `E[ΔPerformance]` = expected improvement from continuing training
- `E[ΔCompute Cost]` = expected additional computational cost

The objective is not simply to maximize training time or minimize epochs.

Instead:

> **Maximize useful model improvement per unit of additional computation.**

---

# 🔥 What Makes Thumb-AI Different?

Traditional early stopping asks:

> "Has validation performance stopped improving?"

Thumb-AI asks:

> "If I continue training, how much improvement can I reasonably expect, how much will it cost, and is that improvement worth it?"

### Traditional Approach

```text
Training
   ↓
Validation Metric
   ↓
No Improvement
   ↓
STOP
```

### Thumb-AI Decision Framework

```text
Training Run
   ↓
Log Checkpoints (Loss, Step, Compute Time)
   ↓
Trajectory Modeling & Power-Law Fit
   ↓
Calculate TCV = E[ΔPerformance] / E[ΔCompute Cost]
   ↓
Evaluate Decision:
  ├── [CONTINUE]        High expected performance return per dollar
  ├── [CHECKPOINT_NOW]  Validation divergence / overfitting detected
  ├── [ADJUST_LR]       Plateau observed under high learning rate
  └── [STOP]            Diminishing marginal returns below economic threshold
```

---

## 💻 Installation & Quickstart

### 1. Installation

Install directly in your Python environment:

```bash
git clone https://github.com/ggthedeveloper/Thumb-AI.git
cd Thumb-AI
pip install -e .
```

### 2. Command Line Interface

Analyze training trajectories from a CSV or JSON file:

```bash
# Analyze a CSV file for the next 500 steps
thumb-ai analyze examples/sample_training.csv --horizon 500 --gpu-cost 2.50

# Output machine-readable JSON for automated training pipelines
thumb-ai analyze examples/sample_training.csv --json

# Run synthetic simulations across different training dynamics
thumb-ai simulate --scenario converging
thumb-ai simulate --scenario overfitting
thumb-ai simulate --scenario plateau
```

### 3. Python API

```python
from thumb_ai import Trajectory, ComputeProfile, TCVCalculator, RecommendationEngine

# Load training logs
traj = Trajectory.from_csv("examples/sample_training.csv")

# Set up your hardware profile
profile = ComputeProfile(
    hardware_name="NVIDIA A100-SXM4-80GB",
    cost_per_gpu_hour=2.50,
    num_gpus=1
)

calc = TCVCalculator(profile)
engine = RecommendationEngine(calculator=calc)

# Evaluate continuation value for the next 500 steps
rec = engine.evaluate(traj, horizon_steps=500, metric_key="val_loss")

print(f"Decision:           {rec.action.value}")
print(f"Confidence:         {rec.confidence * 100:.1f}%")
print(f"Expected Gain:      {rec.tcv.expected_delta_performance:.4f}")
print(f"Expected Cost:      ${rec.tcv.expected_delta_compute_cost:.2f} USD")
print(f"Suggested Action:   {rec.suggested_action}")
```

---

## 🧪 Testing

Thumb-AI is built with high reliability and zero external tracking. Run the comprehensive test suite with:

```bash
pytest -v
```

---

### Developed by
Gaurav Gautam

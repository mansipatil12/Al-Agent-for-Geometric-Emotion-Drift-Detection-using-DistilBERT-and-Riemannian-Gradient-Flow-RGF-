# Geometric Emotion Drift Detection Agent

## Overview
This system is a research-grade AI agent designed to detect emotional instability using **Riemannian Geometry** and **DistilBERT**. It models human emotional states as particles moving on a high-dimensional manifold under the influence of cognitive potentials.

## Core Concepts

### 1. Emotion Manifold & Drift
We represent the emotional state $e_t$ as a probability vector in $\mathbb{R}^6$ (Joy, Sadness, Anger, Fear, Surprise, Love).
**Drift** is defined as the cosine distance between consecutive states:
$$ D_t = 1 - \frac{e_t \cdot e_{t-1}}{\|e_t\| \|e_{t-1}\|} $$
Since $e_t$ implies probability distributions, we use the unnormalized cosine distance on the raw probability vectors.

### 2. Cognitive Potential $V(x)$
We model the mind's efficient processing using two forces:
- **System 1 (Fast/Reactive)**: A potential well centered at the current stimulus $u$.
  $$ V_{fast}(x) = \frac{1}{2} \|x - u\|^2 $$
- **System 2 (Slow/Reflective)**: A potential well centered at a homeostatic baseline $x_0$.
  $$ V_{slow}(x) = \frac{1}{2} \|x - x_0\|^2 $$

The total potential is $V(x) = \alpha V_{fast}(x) + \beta V_{slow}(x)$.

### 3. Riemannian Gradient Flow (RGF)
Emotional states do not change linearly. We define an **Anisotropic Metric** $G(x)$ to model "emotional inertia". High-intensity states have higher curvature (gravity), making them harder to escape.
$$ G(x) = I + \gamma (x x^T) $$

The dynamics follow the gradient flow on this manifold:
$$ \frac{dx}{dt} = -G(x)^{-1} \nabla V(x) $$

## Architecture
- **NLP Layer**: `bhadresh-savani/distilbert-base-uncased-emotion`
- **Dynamics Engine**: Custom Euler-method solver for RGF.
- **Frontend**: Streamlit.

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

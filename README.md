# 🧠 Geometric Emotion Drift Detection Agent

## Overview
This project presents a **research-grade AI system** for detecting **emotional instability and drift** using **Riemannian Geometry** combined with **transformer-based emotion modeling**.  
Human emotional states are modeled as particles evolving on a **curved, high-dimensional manifold**, influenced by competing cognitive potentials.

The system integrates:
- DistilBERT-based emotion classification
- Geometric drift quantification
- Riemannian Gradient Flow (RGF) dynamics
- Interactive visualization via Streamlit

---

## Key Features
- 📊 Sentence-level emotion inference using DistilBERT  
- 📐 Emotion drift measurement via cosine geometry  
- 🌀 Non-Euclidean emotional dynamics using Riemannian manifolds  
- ⚖️ Dual-process cognitive modeling (System 1 & System 2)  
- 📈 Explainable emotional trajectory visualization  
- 🖥️ Streamlit-based interactive UI  

---

## Core Concepts

### 1. Emotion Manifold & Drift
Each emotional state at time \( t \) is represented as a probability vector:

\[
e_t \in \mathbb{R}^6 = [\text{Joy, Sadness, Anger, Fear, Surprise, Love}]
\]

Emotional drift is defined as the cosine distance between consecutive states:

\[
D_t = 1 - \frac{e_t \cdot e_{t-1}}{\|e_t\| \|e_{t-1}\|}
\]

Since the vectors represent probability distributions, cosine distance is applied directly to the raw probability outputs.

---

### 2. Cognitive Potential Function \( V(x) \)

Human cognition is modeled using dual-process theory:

#### System 1 — Fast / Reactive
\[
V_{\text{fast}}(x) = \frac{1}{2} \|x - u\|^2
\]

#### System 2 — Slow / Reflective
\[
V_{\text{slow}}(x) = \frac{1}{2} \|x - x_0\|^2
\]

#### Total Potential
\[
V(x) = \alpha V_{\text{fast}}(x) + \beta V_{\text{slow}}(x)
\]

where:
- \( u \) is the stimulus-driven emotional state  
- \( x_0 \) is the emotional baseline  
- \( \alpha, \beta \) control reactivity vs regulation  

---

### 3. Riemannian Gradient Flow (RGF)

Emotional transitions are non-linear and modeled using an anisotropic Riemannian metric:

\[
G(x) = I + \gamma (x x^\top)
\]

This metric captures emotional inertia, where high-intensity states exhibit higher curvature.

The emotional dynamics follow:

\[
\frac{dx}{dt} = -G(x)^{-1} \nabla V(x)
\]

The system uses a custom Euler-method solver to simulate this flow.

---

## System Architecture

```text
User Text Input
        ↓
DistilBERT Emotion Classifier
        ↓
Emotion Probability Vector (ℝ⁶)
        ↓
Drift Estimation (Cosine Distance)
        ↓
Riemannian Gradient Flow Engine
        ↓
Emotion Trajectory Visualization
        ↓
Streamlit UI

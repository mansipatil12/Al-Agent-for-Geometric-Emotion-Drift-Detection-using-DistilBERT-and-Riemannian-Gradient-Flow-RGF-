import numpy as np
from src.emotion_model import EmotionClassifier
from src.drift_metrics import calculate_drift, calculate_rolling_drift
from src.cognitive_potential import CognitivePotential
from src.riemannian_flow import RiemannianLangevinDynamics

class MentalHealthAgent:
    def __init__(self):
        self.emotion_clf = EmotionClassifier()
        self.history = [] # List of dicts: {'text': ..., 'emotions': ..., 'vector': ..., 'drift': ..., 'cognitive_state': ...}
        
        # Initialize Geometric Core
        self.potential = CognitivePotential()
        self.rgf = RiemannianLangevinDynamics(self.potential)
        
        # State Vector (Internal Cognitive State)
        # Initialize as uniform or zero. We'll size it on first predictions
        self.current_cognitive_state = None
        
        # Thresholds
        self.drift_threshold_warning = 0.3
        self.drift_threshold_alert = 0.5

    def process_input(self, text):
        """
        Main pipeline: Text -> Emotion -> Drift -> RGF -> Feedback
        """
        # 1. Emotion Classification
        emotions_dict, emotion_vec = self.emotion_clf.predict(text)
        
        # Initialize state if first run
        if self.current_cognitive_state is None:
            self.current_cognitive_state = np.copy(emotion_vec)
            self.potential.set_homeostasis(emotion_vec) # Set initial state as baseline? Or use generic.
            # actually better to use generic uniform as baseline for 'calm'
            uniform = np.ones_like(emotion_vec) / len(emotion_vec)
            self.potential.set_homeostasis(uniform)

        # 2. Update Potential with new Stimulus
        self.potential.set_stimulus(emotion_vec)
        
        # 3. Drift Calculation
        prev_vec = self.history[-1]['vector'] if self.history else None
        drift = calculate_drift(emotion_vec, prev_vec)
        
        # 4. Run RGF Dynamics (Cognitive Processing)
        # We simulate the "processing" of this emotion over a short period (e.g., reaction time)
        # This gives us a trajectory and a final "processed" state
        trajectory = self.rgf.simulate_trajectory(self.current_cognitive_state, steps=20)
        self.current_cognitive_state = trajectory[-1] # Update internal state to the end of processing
        
        # 5. Analysis & Feedback
        status = self._analyze_stability(drift)
        feedback = self._generate_feedback(status, emotions_dict, drift)
        
        # Store record
        record = {
            'text': text,
            'emotions': emotions_dict,
            'vector': emotion_vec,
            'drift': drift,
            'trajectory': trajectory,
            'status': status,
            'feedback': feedback,
            'timestamp': len(self.history)
        }
        self.history.append(record)
        
        return record

    def _analyze_stability(self, drift):
        if drift < self.drift_threshold_warning:
            return "Stable"
        elif drift < self.drift_threshold_alert:
            return "Volatile"
        else:
            return "Critical"

    def _generate_feedback(self, status, emotions, drift):
        top_emotion = list(emotions.keys())[0]
        
        base_msg = f"Detected dominant emotion: **{top_emotion}**."
        
        if status == "Stable":
            return f"{base_msg} Your emotional state appears balanced. (Drift: {drift:.2f})"
        elif status == "Volatile":
            return f"{base_msg} I'm noticing some shifts in your emotional landscape. Take a moment to reflect. (Drift: {drift:.2f})"
        else:
            return f"{base_msg} **High Volatility Detected.** This consecutive change is significant. Consider grounding techniques. (Drift: {drift:.2f})"

    def get_history_dataframe(self):
        import pandas as pd
        data = []
        for h in self.history:
            row = {
                'Text': h['text'],
                'Drift': h['drift'],
                'Status': h['status'],
                'Dominant Emotion': list(h['emotions'].keys())[0]
            }
            # Add top 3 emotion probs
            sorted_emotions = list(h['emotions'].items())
            for i in range(min(3, len(sorted_emotions))):
                row[f'Emotion_{i+1}'] = f"{sorted_emotions[i][0]} ({sorted_emotions[i][1]:.2f})"
            data.append(row)
        return pd.DataFrame(data)

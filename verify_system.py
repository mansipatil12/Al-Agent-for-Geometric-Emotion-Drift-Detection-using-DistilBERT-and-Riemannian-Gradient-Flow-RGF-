from src.agent_logic import MentalHealthAgent
import time

def run_demo():
    print("Initializing Agent...")
    agent = MentalHealthAgent()
    
    test_inputs = [
        "I am feeling incredibly happy and excited about my promotion!",
        "Suddenly I feel a bit anxious about the responsibilities.",
        "Now I am just angry that they didn't tell me sooner.",
        "I guess I am okay now, just neutral.",
        "I am terrified of failing."
    ]
    
    print("\n--- Starting Simulation ---\n")
    
    for i, text in enumerate(test_inputs):
        print(f"Step {i+1}: Input: '{text}'")
        record = agent.process_input(text)
        print(f"   -> Top Emotion: {list(record['emotions'].keys())[0]}")
        print(f"   -> Drift Score: {record['drift']:.4f}")
        print(f"   -> Status: {record['status']}")
        print(f"   -> Feedback: {record['feedback']}")
        print(f"   -> Final Cognitive State (Norm): {np.linalg.norm(record['trajectory'][-1]):.4f}")
        print("-" * 30)

import numpy as np
if __name__ == "__main__":
    run_demo()

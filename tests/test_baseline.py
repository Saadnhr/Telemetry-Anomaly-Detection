"""
Quick test script for baseline model
"""
import sys
sys.path.append('../src/models')
sys.path.append('../src/utils')

from baseline_detector import BaselineAnomalyDetector
from generate_telemetry import SatelliteTelemetryGenerator
from datetime import datetime

def test_baseline():
    print("Testing baseline model...")
    
    # Generate small dataset
    generator = SatelliteTelemetryGenerator(
        start_time=datetime(2026, 1, 12),
        duration_hours=2,
        sampling_rate_hz=1.0
    )
    
    df = generator.generate_dataset(n_anomalies=5)
    
    # Train
    detector = BaselineAnomalyDetector()
    detector.train(df)
    
    # Predict
    y_pred, scores = detector.predict(df)
    
    print(f"✅ Test passed!")
    print(f"   Anomalies detected: {y_pred.sum()}")
    print(f"   Score range: {scores.min():.3f} - {scores.max():.3f}")

if __name__ == "__main__":
    test_baseline()
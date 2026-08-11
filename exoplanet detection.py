import numpy as np
from scipy.signal import savgol_filter
import google.generativeai as genai

# 1. AI Configuration
API_KEY = ""
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

print("--- EXOHABITERS: Phase 1 PoC Pipeline ---")
print("1. Ingesting raw light curve telemetry...")

# 2. Simulate Noisy NASA Data (1000 data points)
time = np.linspace(0, 10, 1000)

# Base flux is 1.0, adding random telescope noise
flux = 1.0 + np.random.normal(0, 0.02, 1000)

# Simulate an exoplanet transit (a dip in the light)
flux[450:550] -= 0.08
print("2. Applying Savitzky-Golay Signal Smoothing...")

# 3. Apply the specific filter mentioned in the EXOHABITERS presentation
smoothed_flux = savgol_filter(flux, window_length=51, polyorder=3)

# 4. Feature Extraction
baseline_flux = np.mean(smoothed_flux[0:400])
min_flux = np.min(smoothed_flux)
transit_depth = baseline_flux - min_flux
signal_variance = np.var(smoothed_flux)

print(f" -> Detected Transit Depth: {transit_depth:.4f}")
print(f" -> Signal Variance: {signal_variance:.5f}")
print("3. Routing array features to AI Classification Engine...")

# 5. The AI Integration (Classification)
ai_prompt = f"""
Act as an astrophysicist classification algorithm.
I have processed a stellar light curve.
Transit Depth: {transit_depth:.4f} (Drop in brightness)
Signal Variance: {signal_variance:.5f}
Based on these extracted features, evaluate if this is a high-confidence exoplanet candidate or likely a false positive (like noise or an eclipsing binary). Keep your evaluation to 3 concise sentences.
"""

try:
    response = model.generate_content(ai_prompt)
    print("\n[AI CLASSIFICATION OUTPUT]")
    print(response.text)
    print("\n--- Pipeline Execution Complete ---")
except Exception as e:
    print(f"AI Connection Error: {e}")

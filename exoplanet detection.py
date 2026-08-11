import os
import numpy as np
from scipy.signal import savgol_filter
from google import genai


# ============================================================
# 1. GEMINI AI CONFIGURATION
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Set it in the terminal before running the program."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# 2. START PIPELINE
# ============================================================

print("=" * 60)
print("        EXOHABITERS: PHASE 1 PoC PIPELINE")
print("=" * 60)

print("\n1. Ingesting raw light curve telemetry...")


# ============================================================
# 3. SIMULATE NOISY STELLAR LIGHT CURVE
# ============================================================

# 1000 observations over 10 units of time
time = np.linspace(0, 10, 1000)

# Normal stellar brightness
flux = 1.0 + np.random.normal(0, 0.02, 1000)

# Simulate an exoplanet transit
# A planet passes in front of the star,
# causing a small drop in brightness.
flux[450:550] -= 0.08

print("   -> 1000 light-curve data points generated")
print("   -> Artificial transit signal inserted")


# ============================================================
# 4. SAVITZKY-GOLAY SIGNAL SMOOTHING
# ============================================================

print("\n2. Applying Savitzky-Golay Signal Smoothing...")

smoothed_flux = savgol_filter(
    flux,
    window_length=51,
    polyorder=3
)

print("   -> Noise reduction completed")


# ============================================================
# 5. FEATURE EXTRACTION
# ============================================================

print("\n3. Extracting astrophysical features...")

# Estimate normal stellar brightness
baseline_flux = np.mean(smoothed_flux[:400])

# Minimum brightness during transit
min_flux = np.min(smoothed_flux)

# Transit depth
transit_depth = baseline_flux - min_flux

# Variance of the signal
signal_variance = np.var(smoothed_flux)

print(f"   -> Baseline Flux      : {baseline_flux:.4f}")
print(f"   -> Minimum Flux       : {min_flux:.4f}")
print(f"   -> Transit Depth      : {transit_depth:.4f}")
print(f"   -> Signal Variance    : {signal_variance:.6f}")


# ============================================================
# 6. BASIC TRANSIT CHECK
# ============================================================

print("\n4. Performing preliminary transit analysis...")

if transit_depth > 0.02:
    preliminary_result = "SIGNIFICANT TRANSIT SIGNAL"
else:
    preliminary_result = "WEAK TRANSIT SIGNAL"

print(f"   -> {preliminary_result}")


# ============================================================
# 7. SEND FEATURES TO GEMINI AI
# ============================================================

print("\n5. Routing features to AI Classification Engine...")

ai_prompt = f"""
You are an astrophysicist analyzing a stellar light curve.

The following features were extracted from the light curve:

Baseline flux: {baseline_flux:.4f}
Minimum flux: {min_flux:.4f}
Transit depth: {transit_depth:.4f}
Signal variance: {signal_variance:.6f}

Determine whether this signal is more consistent with:

1. A potential exoplanet transit
2. Noise
3. An eclipsing binary / other false positive

Important:
- A transit depth alone cannot prove that an exoplanet exists.
- Give a cautious scientific assessment.
- Mention that additional validation such as periodicity,
  odd-even depth tests and stellar/binary analysis is required.

Give the result in exactly 3 concise sentences.
"""


# ============================================================
# 8. GEMINI CLASSIFICATION
# ============================================================

try:

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=ai_prompt
    )

    print("\n" + "=" * 60)
    print("             AI CLASSIFICATION OUTPUT")
    print("=" * 60)

    print(response.text)

except Exception as error:

    print("\nAI Connection Error:")
    print(error)


# ============================================================
# 9. PIPELINE COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("             PIPELINE EXECUTION COMPLETE")
print("=" * 60)

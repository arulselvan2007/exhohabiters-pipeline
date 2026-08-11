import os
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from astropy.timeseries import BoxLeastSquares

# ============================================================
# GEMINI AI
# ============================================================

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ============================================================
# EXOHABITERS
# AI-Assisted Exoplanet Candidate Detection Pipeline
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

TARGET = "TIC 261136679"


# ============================================================
# 1. TESS DATA ACQUISITION
# ============================================================

def download_tess_data(target):
    """Search and download real TESS light-curve data."""

    print("=" * 60)
    print("EXOHABITERS - TESS DATA ACQUISITION")
    print("=" * 60)

    print(
        f"\nSearching TESS data for: {target}"
    )

    search_result = lk.search_lightcurve(
        target,
        mission="TESS"
    )

    if len(search_result) == 0:
        raise RuntimeError(
            f"No TESS data found for {target}"
        )

    print(
        f"Found {len(search_result)} "
        "available TESS data products."
    )

    print(
        "\nDownloading first available observation..."
    )

    light_curve = search_result[0].download()

    if light_curve is None:
        raise RuntimeError(
            "TESS light curve could not be downloaded."
        )

    print("Download successful!")

    return light_curve


# ============================================================
# 2. DATA CLEANING
# ============================================================

def clean_light_curve(light_curve):
    """Remove missing values, outliers and normalize flux."""

    print("\nCleaning light curve...")

    light_curve = light_curve.remove_nans()

    light_curve = light_curve.remove_outliers(
        sigma=5
    )

    light_curve = light_curve.normalize()

    print(
        f"Remaining observations: "
        f"{len(light_curve)}"
    )

    return light_curve


# ============================================================
# 3. DETRENDING
# ============================================================

def detrend_light_curve(light_curve):
    """Remove long-term instrumental and stellar trends."""

    print("\nDetrending light curve...")

    flat_light_curve = light_curve.flatten(
        window_length=401
    )

    flat_light_curve = (
        flat_light_curve.remove_nans()
    )

    print(
        f"Observations after detrending: "
        f"{len(flat_light_curve)}"
    )

    return flat_light_curve


# ============================================================
# 4. SAVITZKY-GOLAY SMOOTHING
# ============================================================

def apply_savgol(light_curve):
    """Apply Savitzky-Golay filtering."""

    print(
        "\nApplying Savitzky-Golay filter..."
    )

    time = np.asarray(
        light_curve.time.value
    )

    flux = np.asarray(
        light_curve.flux.value
    )

    mask = (
        np.isfinite(time)
        & np.isfinite(flux)
    )

    time = time[mask]
    flux = flux[mask]

    window_length = min(
        101,
        len(flux) - 1
    )

    if window_length % 2 == 0:
        window_length -= 1

    if window_length < 5:
        print(
            "Not enough data for "
            "Savitzky-Golay filtering."
        )

        return time, flux

    smoothed_flux = savgol_filter(
        flux,
        window_length=window_length,
        polyorder=2
    )

    return time, smoothed_flux


# ============================================================
# 5. BOX LEAST SQUARES TRANSIT DETECTION
# ============================================================

def detect_transit(time, flux):
    """Search for periodic transit-like signals."""

    print("\n" + "=" * 60)
    print("RUNNING BOX LEAST SQUARES")
    print("=" * 60)

    model = BoxLeastSquares(
        time,
        flux
    )

    durations = np.array([
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30
    ])

    print(
        "Searching orbital periods "
        "between 0.5 and 20 days..."
    )

    periodogram = model.autopower(
        durations,
        minimum_period=0.5,
        maximum_period=20.0,
        objective="snr"
    )

    best_index = np.argmax(
        periodogram.power
    )

    result = {
        "period": float(
            periodogram.period[
                best_index
            ]
        ),
        "duration": float(
            periodogram.duration[
                best_index
            ]
        ),
        "depth": float(
            periodogram.depth[
                best_index
            ]
        ),
        "transit_time": float(
            periodogram.transit_time[
                best_index
            ]
        ),
        "power": float(
            periodogram.power[
                best_index
            ]
        ),
        "periodogram": periodogram
    }

    print("\n" + "=" * 60)
    print("BLS TRANSIT DETECTION RESULT")
    print("=" * 60)

    print(
        f"Best orbital period : "
        f"{result['period']:.6f} days"
    )

    print(
        f"Transit duration    : "
        f"{result['duration']:.6f} days"
    )

    print(
        f"Transit depth       : "
        f"{result['depth']:.6f}"
    )

    print(
        f"Transit time        : "
        f"{result['transit_time']:.6f}"
    )

    print(
        f"BLS power           : "
        f"{result['power']:.6f}"
    )

    print("=" * 60)

    return result


# ============================================================
# 6. GEMINI AI ANALYSIS
# ============================================================

def ai_analysis(result):
    """Use Gemini AI to analyze the detected candidate."""

    print("\n" + "=" * 60)
    print("AI CANDIDATE ANALYSIS")
    print("=" * 60)

    if not GEMINI_AVAILABLE:
        message = (
            "Gemini SDK is not installed."
        )

        print(message)
        return message

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        message = (
            "GEMINI_API_KEY is not configured."
        )

        print(message)
        return message

    try:

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
You are an AI assistant integrated into
an astronomical exoplanet candidate
detection pipeline.

Target:
{TARGET}

The Box Least Squares algorithm detected:

Orbital period:
{result['period']:.6f} days

Transit duration:
{result['duration']:.6f} days

Transit depth:
{result['depth']:.6f}

BLS power:
{result['power']:.6f}

Analyze these measurements.

Provide a concise scientific assessment
with these sections:

1. Signal Interpretation
2. Candidate Assessment
3. Possible False Positives
4. Recommended Validation

Explain what the detected period and
transit depth mean.

Do not claim that an exoplanet is confirmed.
Use "candidate signal" or
"exoplanet candidate" when appropriate.

Mention that additional validation such
as odd-even transit comparison,
secondary-eclipse checks, stellar
parameters, and independent observations
would be required for confirmation.
"""

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )

        analysis = response.text

        print("\nAI ASSESSMENT:\n")
        print(analysis)

        return analysis

    except Exception as error:

        message = (
            f"AI analysis failed: {error}"
        )

        print(message)

        return message


# ============================================================
# 7. SAVE PROCESSED LIGHT CURVE
# ============================================================

def save_light_curve(time, flux):
    """Save processed light curve visualization."""

    output_file = os.path.join(
        RESULTS_DIR,
        "processed_lightcurve.png"
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        flux,
        ".",
        markersize=2,
        alpha=0.6
    )

    plt.xlabel(
        "Time [days]"
    )

    plt.ylabel(
        "Normalized Flux"
    )

    plt.title(
        "EXOHABITERS - "
        "Processed TESS Light Curve"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200
    )

    plt.close()

    print(
        f"\nSaved light curve: "
        f"{output_file}"
    )


# ============================================================
# 8. SAVE BLS PERIODOGRAM
# ============================================================

def save_bls_plot(periodogram):
    """Save BLS periodogram visualization."""

    output_file = os.path.join(
        RESULTS_DIR,
        "bls_periodogram.png"
    )

    best_index = np.argmax(
        periodogram.power
    )

    best_period = (
        periodogram.period[
            best_index
        ]
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        periodogram.period,
        periodogram.power
    )

    plt.axvline(
        best_period,
        linestyle="--",
        label=(
            f"Best period = "
            f"{best_period:.4f} days"
        )
    )

    plt.xlabel(
        "Orbital Period [days]"
    )

    plt.ylabel(
        "BLS SNR"
    )

    plt.title(
        "EXOHABITERS - BLS Periodogram"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200
    )

    plt.close()

    print(
        f"Saved BLS periodogram: "
        f"{output_file}"
    )


# ============================================================
# 9. SAVE CANDIDATE REPORT
# ============================================================

def save_report(result, ai_result):
    """Save detection measurements and AI analysis."""

    report_file = os.path.join(
        RESULTS_DIR,
        "candidate_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "EXOHABITERS CANDIDATE REPORT\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Target: {TARGET}\n\n"
        )

        file.write(
            f"Orbital period: "
            f"{result['period']:.6f} days\n"
        )

        file.write(
            f"Transit duration: "
            f"{result['duration']:.6f} days\n"
        )

        file.write(
            f"Transit depth: "
            f"{result['depth']:.6f}\n"
        )

        file.write(
            f"BLS power: "
            f"{result['power']:.6f}\n\n"
        )

        file.write(
            "AI ANALYSIS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            ai_result
        )

        file.write(
            "\n\nIMPORTANT:\n"
        )

        file.write(
            "This pipeline identifies candidate "
            "transit signals. It does not confirm "
            "the existence of an exoplanet.\n"
        )

    print(
        f"\nSaved candidate report: "
        f"{report_file}"
    )


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():

    try:

        # TESS data acquisition
        light_curve = (
            download_tess_data(
                TARGET
            )
        )

        # Data cleaning
        light_curve = (
            clean_light_curve(
                light_curve
            )
        )

        # Detrending
        light_curve = (
            detrend_light_curve(
                light_curve
            )
        )

        # Savitzky-Golay filtering
        time, flux = (
            apply_savgol(
                light_curve
            )
        )

        # BLS transit detection
        result = detect_transit(
            time,
            flux
        )

        # Save visualizations
        save_light_curve(
            time,
            flux
        )

        save_bls_plot(
            result["periodogram"]
        )

        # AI analysis
        ai_result = ai_analysis(
            result
        )

        # Candidate report
        save_report(
            result,
            ai_result
        )

        print("\n" + "=" * 60)
        print(
            "EXOHABITERS PIPELINE COMPLETED SUCCESSFULLY"
        )
        print("=" * 60)

    except Exception as error:

        print("\nERROR:")
        print(error)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.signal import savgol_filter
from astropy.timeseries import BoxLeastSquares


# ============================================================
# EXOHABITERS - REAL EXOPLANET DETECTION PIPELINE
# ============================================================

TARGET = "TIC 261136679"


# ============================================================
# 1. DOWNLOAD REAL TESS DATA
# ============================================================

def download_lightcurve(target):
    print("\n[1] Searching TESS data...")
    print(f"Target: {target}")

    search = lk.search_lightcurve(
        target,
        mission="TESS"
    )

    if len(search) == 0:
        raise RuntimeError(
            "No TESS light curve found for this target."
        )

    print("\nAvailable observations:")
    print(search)

    # Download the first available light curve
    lc = search[0].download()

    if lc is None:
        raise RuntimeError("Could not download light curve.")

    print("\nTESS light curve downloaded successfully.")

    return lc


# ============================================================
# 2. CLEAN THE LIGHT CURVE
# ============================================================

def clean_lightcurve(lc):

    print("\n[2] Cleaning light curve...")

    # Remove missing values
    lc = lc.remove_nans()

    # Remove extreme outliers
    lc = lc.remove_outliers(sigma=5)

    # Normalize flux around 1
    lc = lc.normalize()

    print(f"Number of observations: {len(lc)}")

    return lc


# ============================================================
# 3. DETREND THE LIGHT CURVE
# ============================================================

def detrend_lightcurve(lc):

    print("\n[3] Detrending light curve...")

    # Lightkurve flattening removes long-term stellar/systematic
    # variations while preserving short transit-like features.
    flat_lc = lc.flatten(window_length=401)

    flat_lc = flat_lc.remove_nans()

    return flat_lc


# ============================================================
# 4. SAVITZKY-GOLAY FILTER
# ============================================================

def smooth_lightcurve(lc):

    print("\n[4] Applying Savitzky-Golay smoothing...")

    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    # Make sure the window is odd and smaller than data size
    window = min(101, len(flux) - 1)

    if window % 2 == 0:
        window -= 1

    if window < 5:
        print("Not enough data for Savitzky-Golay filter.")
        return time, flux

    smoothed_flux = savgol_filter(
        flux,
        window_length=window,
        polyorder=2
    )

    return time, smoothed_flux


# ============================================================
# 5. BLS TRANSIT DETECTION
# ============================================================

def detect_transit(time, flux):

    print("\n[5] Running Box Least Squares...")

    # BLS expects time in days and normalized flux
    model = BoxLeastSquares(time, flux)

    # Transit durations in days
    durations = np.array([
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30
    ])

    print("Searching for periodic transit signals...")

    periodogram = model.autopower(
        durations,
        minimum_period=0.5,
        maximum_period=20,
        objective="snr"
    )

    # Strongest BLS signal
    best_index = np.argmax(periodogram.power)

    period = float(periodogram.period[best_index])
    power = float(periodogram.power[best_index])
    duration = float(periodogram.duration[best_index])
    transit_time = float(periodogram.transit_time[best_index])
    depth = float(periodogram.depth[best_index])
    depth_snr = float(periodogram.depth_snr[best_index])

    print("\n========== BLS RESULT ==========")

    print(f"Period          : {period:.6f} days")
    print(f"Transit duration : {duration:.6f} days")
    print(f"Transit depth    : {depth:.6f}")
    print(f"BLS power        : {power:.6f}")
    print(f"Depth SNR        : {depth_snr:.2f}")
    print(f"Transit time     : {transit_time:.6f}")

    return {
        "period": period,
        "power": power,
        "duration": duration,
        "transit_time": transit_time,
        "depth": depth,
        "depth_snr": depth_snr
    }, periodogram


# ============================================================
# 6. ODD / EVEN TRANSIT TEST
# ============================================================

def odd_even_test(time, flux, result):

    print("\n[6] Running Odd-Even Transit Test...")

    period = result["period"]
    duration = result["duration"]
    transit_time = result["transit_time"]

    model = BoxLeastSquares(time, flux)

    stats = model.compute_stats(
        period,
        duration,
        transit_time
    )

    depth_odd = float(stats["depth_odd"][0])
    depth_even = float(stats["depth_even"][0])

    print(f"Odd transit depth  : {depth_odd:.6f}")
    print(f"Even transit depth : {depth_even:.6f}")

    difference = abs(depth_odd - depth_even)

    print(f"Odd/Even difference: {difference:.6f}")

    # Simple screening rule.
    # This is NOT a definitive false-positive classifier.
    if max(abs(depth_odd), abs(depth_even)) > 0:
        relative_difference = (
            difference /
            max(abs(depth_odd), abs(depth_even))
        )
    else:
        relative_difference = 0

    print(
        f"Relative difference: "
        f"{relative_difference * 100:.2f}%"
    )

    if relative_difference < 0.30:
        verdict = "CONSISTENT"
    else:
        verdict = "POSSIBLE FALSE POSITIVE"

    print(f"Odd/Even result: {verdict}")

    return {
        "depth_odd": depth_odd,
        "depth_even": depth_even,
        "relative_difference": relative_difference,
        "verdict": verdict
    }


# ============================================================
# 7. CREATE TRANSIT MASK
# ============================================================

def create_transit_mask(time, result):

    period = result["period"]
    duration = result["duration"]
    transit_time = result["transit_time"]

    model = BoxLeastSquares(time, np.ones(len(time)))

    mask = model.transit_mask(
        time,
        period,
        duration,
        transit_time
    )

    return mask


# ============================================================
# 8. PLOT LIGHT CURVE
# ============================================================

def plot_lightcurve(time, flux, result):

    print("\n[7] Creating light curve plot...")

    period = result["period"]
    duration = result["duration"]
    transit_time = result["transit_time"]

    mask = create_transit_mask(time, result)

    plt.figure(figsize=(12, 5))

    plt.plot(
        time,
        flux,
        ".",
        markersize=2,
        alpha=0.5,
        label="TESS observations"
    )

    plt.plot(
        time[mask],
        flux[mask],
        ".",
        markersize=3,
        label="Detected transit"
    )

    plt.xlabel("Time [days]")
    plt.ylabel("Normalized Flux")

    plt.title(
        f"EXOHABITERS - Detected Transit\n"
        f"Period = {period:.4f} days"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "detected_transit.png",
        dpi=200
    )

    plt.show()


# ============================================================
# 9. PLOT BLS PERIODOGRAM
# ============================================================

def plot_bls(periodogram):

    print("\n[8] Creating BLS periodogram...")

    plt.figure(figsize=(12, 5))

    plt.plot(
        periodogram.period,
        periodogram.power
    )

    best_index = np.argmax(periodogram.power)

    plt.axvline(
        periodogram.period[best_index],
        linestyle="--",
        label="Best period"
    )

    plt.xlabel("Orbital Period [days]")
    plt.ylabel("BLS SNR")

    plt.title(
        "EXOHABITERS - Box Least Squares Periodogram"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "bls_periodogram.png",
        dpi=200
    )

    plt.show()


# ============================================================
# 10. FINAL CANDIDATE REPORT
# ============================================================

def generate_report(result, odd_even):

    print("\n")
    print("=" * 60)
    print("          EXOHABITERS CANDIDATE REPORT")
    print("=" * 60)

    print(f"Orbital period       : {result['period']:.6f} days")
    print(f"Transit duration     : {result['duration']:.6f} days")
    print(f"Transit depth        : {result['depth']:.6f}")
    print(f"BLS power            : {result['power']:.6f}")
    print(f"Transit depth SNR    : {result['depth_snr']:.2f}")
    print(f"Odd transit depth    : {odd_even['depth_odd']:.6f}")
    print(f"Even transit depth   : {odd_even['depth_even']:.6f}")
    print(
        f"Odd/Even difference  : "
        f"{odd_even['relative_difference'] * 100:.2f}%"
    )

    print("-" * 60)

    # Conservative screening.
    if (
        result["depth_snr"] >= 7
        and odd_even["verdict"] == "CONSISTENT"
    ):
        print("RESULT: EXOPLANET CANDIDATE")
        print("Further astrophysical validation is required.")
    else:
        print("RESULT: LOW-CONFIDENCE CANDIDATE")
        print("Further investigation is required.")

    print("=" * 60)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("       EXOHABITERS EXOPLANET PIPELINE")
    print("=" * 60)

    try:

        # 1. Real TESS data
        lc = download_lightcurve(TARGET)

        # 2. Clean
        lc = clean_lightcurve(lc)

        # 3. Detrend
        flat_lc = detrend_lightcurve(lc)

        # 4. Smooth
        time, flux = smooth_lightcurve(flat_lc)

        # 5. BLS
        result, periodogram = detect_transit(
            time,
            flux
        )

        # 6. Odd/Even
        odd_even = odd_even_test(
            time,
            flux,
            result
        )

        # 7. Plots
        plot_lightcurve(
            time,
            flux,
            result
        )

        plot_bls(
            periodogram
        )

        # 8. Final report
        generate_report(
            result,
            odd_even
        )

        print("\nPipeline completed successfully.")

    except Exception as e:

        print("\nERROR:")
        print(e)

        print(
            "\nPlease check your internet connection "
            "and installed packages."
        )


if __name__ == "__main__":
    main()

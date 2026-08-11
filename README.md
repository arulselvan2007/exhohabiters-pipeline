# exhohabiters-pipeline
AI-enabled pipeline Proof-of-Concept to detect exoplanets from noisy astronomical light curves using Gemini AI

# EXOHABITERS - AI Exoplanet Detection Pipeline (PoC)

## Project Overview
This repository contains the Phase 1 Python Proof-of-Concept (PoC) for the **EXOHABITERS** architecture: an AI-enabled pipeline designed to detect exoplanets from noisy astronomical light curves. 

This script demonstrates the core methodology of processing raw stellar telemetry and routing the extracted features to a Generative AI model for high-confidence classification.

## Core Features
*   **Signal Processing:** Implements **Savitzky-Golay filtering** (via `scipy`) to mathematically smooth stellar flares and reduce instrumental noise.
*   **Feature Extraction:** Programmatically calculates critical astrophysical data points, including *Transit Depth* and *Signal Variance*.
*   **AI Integration:** Utilizes the **Google Gemini API** to evaluate the extracted features and classify the signal as a viable exoplanet candidate or a false positive (e.g., eclipsing binary).

## How to Run Locally
1. Clone this repository.
2. Install the required scientific libraries:
   `pip install numpy scipy google-generativeai`
3. Insert your own Google AI API Key into the `API_KEY` variable.
4. Execute the pipeline:
   `python3 "exoplanet detection.py"`

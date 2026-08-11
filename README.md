# EXOHABITERS 🛰️

## AI-Assisted Exoplanet Candidate Detection Pipeline

EXOHABITERS is an AI-assisted astronomical data analysis pipeline that processes real NASA TESS light-curve data to identify periodic transit-like signals and generate an AI-based scientific assessment of potential exoplanet candidates.

The project combines traditional astronomical signal processing with Google's Gemini AI to create an end-to-end candidate screening workflow.

---

## 🚀 Project Overview

Detecting exoplanets through the transit method involves identifying tiny periodic decreases in a star's observed brightness.

EXOHABITERS automates the initial stages of this process:

```text
Real TESS Data
      ↓
Light Curve Acquisition
      ↓
Data Cleaning
      ↓
Detrending
      ↓
Savitzky-Golay Filtering
      ↓
Box Least Squares (BLS)
      ↓
Transit Candidate Features
      ↓
Gemini AI Analysis
      ↓
Candidate Assessment
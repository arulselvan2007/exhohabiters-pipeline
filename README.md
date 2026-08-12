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




🛠️ Installation Guide


1. Clone the Repository


git clone git@github.com:arulselvan2007/exhohabiters-pipeline.git
cd exhohabiters-pipeline



2. Create a Python Virtual Environment


Python 3.9 or newer is recommended.


python3 -m venv .venv



3. Activate the Virtual Environment


On macOS/Linux:


source .venv/bin/activate



On Windows:


.venv\Scripts\activate



After activation, the terminal should show:


(.venv)



4. Install Project Dependencies


Install all required Python packages using:


python -m pip install -r requirements.txt



The project uses:




NumPy — numerical computation


SciPy — signal processing and Savitzky-Golay filtering


Matplotlib — visualization


Astropy — astronomical analysis and Box Least Squares


Lightkurve — TESS light-curve acquisition and processing


Google GenAI — Gemini AI integration




Google's official Gemini documentation recommends the google-genai Python SDK and supports Python 3.9+.


5. Configure the Gemini API Key


The Gemini API key must not be placed directly inside the source code.


Create an API key through Google AI Studio, then set it as an environment variable.


On macOS/Linux:


export GEMINI_API_KEY="YOUR_API_KEY"



On Windows Command Prompt:


set GEMINI_API_KEY=YOUR_API_KEY



On Windows PowerShell:


$env:GEMINI_API_KEY="YOUR_API_KEY"



Verify that the key is configured without displaying the key:


python -c "import os; print('API key configured:', bool(os.getenv('GEMINI_API_KEY')))"



Expected output:


API key configured: True



6. Run the Pipeline


python main.py



The pipeline will:




Search for available TESS observations.


Download the selected TESS light curve.


Remove missing values and outliers.


Normalize the light curve.


Detrend long-term variations.


Apply Savitzky-Golay filtering.


Search orbital periods using Box Least Squares.


Generate processed light-curve and BLS-periodogram plots.


Send detected candidate parameters to Gemini AI.


Generate an AI-assisted candidate assessment.


Save the final candidate report.




7. Generated Results


After a successful run, the following files are generated in the results/ directory:


results/
├── tess_lightcurve.png
├── processed_lightcurve.png
├── bls_periodogram.png
└── candidate_report.txt



⚠️ API Key Security


Never commit your Gemini API key to GitHub.


Do not write:


api_key = "YOUR_REAL_API_KEY"



inside main.py.


Instead, EXOHABITERS reads the key from:


GEMINI_API_KEY



The .gitignore file also excludes .env files and the Python virtual environment.


Troubleshooting


If you receive:


ModuleNotFoundError



make sure the virtual environment is activated and run:


python -m pip install -r requirements.txt



If Gemini reports that the API key is not configured:


python -c "import os; print(bool(os.getenv('GEMINI_API_KEY')))"



If the output is:


False



set the environment variable again before running the pipeline.


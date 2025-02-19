#requires -version 3.0
<#
   setup.ps1
   PowerShell script to create/activate a Python virtual environment
   and install dependencies for the Facebook Bot Master Overlord code.
#>

Write-Host "============================================"
Write-Host "Facebook Bot Master Overlord Setup (PowerShell)"
Write-Host "============================================"

# 1. Check if Python is installed
Write-Host "Checking for Python..."
$python = Get-Command python -ErrorAction SilentlyContinue
if (!$python) {
    Write-Host "ERROR: Python is not found on your PATH."
    Write-Host "Please install Python (3.7+) and try again."
    exit 1
}
else {
    Write-Host ("Python found at: " + $python.Source)
}

# 2. Create a virtual environment in ./venv if it doesn't exist
if (!(Test-Path -Path "./venv")) {
    Write-Host "Creating virtual environment in ./venv..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment. Exiting."
        exit 1
    }
}
else {
    Write-Host "Virtual environment already exists in ./venv"
}

# 3. Activate the virtual environment
Write-Host "Activating the virtual environment..."
. ./venv/Scripts/Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate virtual environment. Exiting."
    exit 1
}

# 4. Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upgrade pip. Exiting."
    exit 1
}

# 5. Install required packages
Write-Host "Installing required packages..."
# For the Facebook Bot Master Overlord project:
python -m pip install facebook-sdk openai python-dotenv cryptography pillow `
    requests scikit-learn nltk numpy matplotlib apscheduler pyotp
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install required packages. Exiting."
    exit 1
}

# 6. Download NLTK data (vader_lexicon)
Write-Host "Downloading NLTK vader_lexicon..."
python -c "import nltk; nltk.download('vader_lexicon')"

Write-Host "============================================"
Write-Host "Setup Complete!"
Write-Host "To use your virtual environment in PowerShell, run:"
Write-Host "    .\\venv\\Scripts\\Activate.ps1"
Write-Host "Then start your bot with something like:"
Write-Host "    python main_bot.py"
Write-Host "============================================"

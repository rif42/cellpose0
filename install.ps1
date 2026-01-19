# 1. Install Python and Git using Windows Package Manager
winget install -e --id Python.Python.3.10
winget install -e --id Git.Git

# Refresh Path
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# 3. Create and access virtual environment
.\cellpose\Scripts\Activate.ps1

# 4. Install dependencies and cellpose
pip install --upgrade pip
pip install "cellpose[gui]"

# 5. Execute
python -m cellpose
# 1. Install Python and Git using Windows Package Manager
winget install -e --id Python.Python.3.11 --scope machine --source winget
winget install -e --id Git.Git --source winget

# Refresh Path
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

cd cellpose

# 3. Create virtual environment
python -m venv .

# 4. Activate virtual environment
.\Scripts\Activate.ps1

# 5. Install dependencies and cellpose
pip install --upgrade pip
pip install "cellpose"

# 5. Execute
python -m cellpose

# Wait for key press before exiting

Read-Host -Prompt "Press Enter to exit"
# ============================================================================
# POWERSHELL COMMANDS FOR GITHUB REPOSITORY SETUP
# Repository: quantum-verification-framework
# Author: H M Shujaat Zaheer (shujabis@gmail.com)
# ============================================================================

# STEP 1: Configure Git (one-time setup if not already done)
# ----------------------------------------------------------------------------
git config --global user.name "H M Shujaat Zaheer"
git config --global user.email "shujabis@gmail.com"

# STEP 2: Navigate to your project directory
# ----------------------------------------------------------------------------
# Replace with the actual path where you extracted the files
cd C:\Users\YourUsername\Documents\quantum-verification-framework

# STEP 3: Initialize Git Repository
# ----------------------------------------------------------------------------
git init

# STEP 4: Add all files to staging
# ----------------------------------------------------------------------------
git add .

# STEP 5: Create initial commit
# ----------------------------------------------------------------------------
git commit -m "Initial commit: Unified Quantum Verification Framework

Implementation of PhD research proposal for fault-tolerant verification
of quantum random sampling, bridging k-uniform states and MBQC.

Features:
- Stabilizer tableau operations with GF(2) rank computation
- k-Uniformity calculator via stabilizer formalism
- Direct Fidelity Estimation (DFE) protocols
- Adaptive Stabilizer Sampling algorithm (Algorithm 1)
- Hybrid physical-logical verification protocol
- Surface code and color code circuit construction
- Comprehensive examples and tests

Based on:
[1] Ringbauer et al., Nature Communications 16, 106 (2025)
[2] Majidy, Hangleiter & Gullans, arXiv:2503.14506 (2025)"

# STEP 6: Create the GitHub repository using GitHub CLI (gh)
# ----------------------------------------------------------------------------
# Option A: Using GitHub CLI (recommended - install from https://cli.github.com/)
gh repo create quantum-verification-framework --public --description "Fault-tolerant verification of quantum random sampling: bridging k-uniform states and measurement-based computation" --source=. --remote=origin --push

# Option B: Manual repository creation
# ----------------------------------------------------------------------------
# 1. Go to https://github.com/new
# 2. Repository name: quantum-verification-framework
# 3. Description: Fault-tolerant verification of quantum random sampling
# 4. Make it Public
# 5. Do NOT initialize with README (we already have one)
# 6. Click "Create repository"
# 7. Then run these commands:

git remote add origin https://github.com/hmshujaatzaheer/quantum-verification-framework.git
git branch -M main
git push -u origin main

# STEP 7: Verify the repository is online
# ----------------------------------------------------------------------------
# Visit: https://github.com/hmshujaatzaheer/quantum-verification-framework

# ============================================================================
# ALTERNATIVE: Using PowerShell with GitHub API (Personal Access Token)
# ============================================================================

# First, create a Personal Access Token at:
# https://github.com/settings/tokens/new
# Select: repo, workflow scopes

# Store your token (replace YOUR_TOKEN with actual token)
$token = "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN"
$repoName = "quantum-verification-framework"
$description = "Fault-tolerant verification of quantum random sampling: bridging k-uniform states and measurement-based computation"

# Create repository via GitHub API
$headers = @{
    "Authorization" = "token $token"
    "Accept" = "application/vnd.github.v3+json"
}

$body = @{
    "name" = $repoName
    "description" = $description
    "private" = $false
    "auto_init" = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType "application/json"

# Then push your code
git remote add origin "https://github.com/hmshujaatzaheer/$repoName.git"
git branch -M main
git push -u origin main

# ============================================================================
# REPOSITORY STRUCTURE (what will be uploaded)
# ============================================================================
<#
quantum-verification-framework/
├── README.md                          # Comprehensive documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation script
├── .gitignore                         # Git ignore patterns
├── src/
│   ├── __init__.py                    # Package exports
│   ├── stabilizer_tableau.py          # Stabilizer formalism implementation
│   ├── fidelity_estimation.py         # DFE and adaptive verification
│   └── circuits.py                    # Circuit construction
├── examples/
│   ├── __init__.py
│   └── verification_demo.py           # Complete workflow examples
├── tests/
│   ├── __init__.py
│   └── test_verification.py           # Unit tests
├── docs/
│   ├── research_proposal.tex          # LaTeX source with GitHub link
│   └── research_proposal.pdf          # Compiled PDF
└── figures/                           # Generated figures directory
#>

# ============================================================================
# QUICK VERIFICATION COMMANDS
# ============================================================================

# Test the code works
cd quantum-verification-framework
pip install -r requirements.txt
python -c "from src import StabilizerTableau; print('Import successful!')"
python examples/verification_demo.py

# Run tests
python -m pytest tests/ -v

# ============================================================================
# FINAL REPOSITORY URL
# ============================================================================
# https://github.com/hmshujaatzaheer/quantum-verification-framework
#
# Clone command for others:
# git clone https://github.com/hmshujaatzaheer/quantum-verification-framework.git

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m src.model_training.train
python -m src.deployment.predict --text "Free money waiting for you. Click to claim now."

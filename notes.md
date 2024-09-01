python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

python -m src.data_split

python -m src.data_cleaning

git config --global init.defaultBranch main

git init

git add .

git commit -m "first code added."

git branch -M main

git remote add origin https://github.com/Danselem/heart_disease_mlflow.git




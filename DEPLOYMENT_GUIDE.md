# Voyage Analytics - Combined Streamlit App

## Run locally

Use Python 3.11. From the project root (the folder containing `app.py`):

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**
```bat
.venv\Scripts\activate.bat
```

Install dependencies and launch:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Community Cloud

1. Push the complete project folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app and select that repository.
3. Set the branch to the branch containing your code (usually `main`).
4. Set the main file path to `app.py` (at the repository root).
5. Deploy. The first deployment installs `requirements.txt`.

The Gender Classification page downloads the `sentence-transformers/all-MiniLM-L6-v2` embedding model on its first use. This can take a little while and requires internet access.

## Project notes

- The combined app uses the original trained model and encoder files in `Flight price regression/` and `Gender classification files/`.
- The hotel recommender reads `Hotel recommender system/hotel_features.csv`.
- `new_df.csv` was not present in the supplied ZIP. The flight page therefore asks for estimated flight time and distance manually.
- The flight prediction feature order is aligned with the saved model: departure, arrival, flight type, agency, time, distance.
- Scikit-learn is pinned to 1.6.1 to match the version used to save the supplied models.

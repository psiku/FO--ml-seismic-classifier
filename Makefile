.PHONY: run env

data:
	poetry run python src/data/download_data.py
	poetry run python src/data/download_iquique.py

run:
	poetry run streamlit run src/streamlit/app.py

dev:
	streamlit run src/streamlit/app.py

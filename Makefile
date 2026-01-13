.PHONY: run env

run:
	poetry run streamlit run src/streamlit/app.py

dev:
	streamlit run src/streamlit/app.py

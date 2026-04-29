import streamlit as st

st.set_page_config(
    page_title="Proof of Concept mit synthetischen Daten",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/vorhersage.py",     title="Vorhersage"),
    st.Page("pages/trainingsdaten.py", title="Trainingsdaten"),
    st.Page("pages/evaluation.py",     title="Evaluation"),
])
pg.run()

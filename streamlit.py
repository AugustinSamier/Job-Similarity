import streamlit as st
import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

nltk.download("stopwords")
stopFrench = stopwords.words("french")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def normalize(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text)
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    words = text.split()
    words = [w for w in words if w not in stopFrench]
    return " ".join(words)

st.title("Détection du métier à partir d'un texte")
st.write("Entrez une description de mission ou de compétences, et l'outil identifiera le métier le plus proche.")

skills=pd.read_excel("data.xlsx",sheet_name=0)

user_input = st.text_area("Votre texte :", height=200)
if st.button("Analyser") and user_input.strip() != "":
    cleaned_user_input = normalize(user_input)

    embedded_user = model.encode(cleaned_user_input, convert_to_tensor=True)
    embedded_skills = model.encode(
        skills["Competency/Skills"].apply(normalize),
        convert_to_tensor=True
    )
    
    similarities = util.cos_sim(embedded_user, embedded_skills)[0].numpy()
    skills["Score"] = similarities
    
    jobDf = skills.groupby("Job")["Score"].mean().reset_index()
    jobDf = jobDf.sort_values(by="Score", ascending=False).reset_index(drop=True)
    # Résultat principal
    best_job = jobDf.loc[0, "Job"]
    st.success(f"Le métier qui vous correspond le plus est : **{best_job}**")
    # Tableau complet
    st.subheader("Classement des métiers")
    st.dataframe(jobDf)
    # Top 5 compétences les plus proches
    st.subheader("Compétences les plus pertinentes")
    st.dataframe(skills.sort_values(by="Score", ascending=False)[["Competency/Skills", "Score"]].head(5))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Nastavenie stránky
st.set_page_config(page_title="Regresné modely v ekonómii", layout="wide")
st.title("Prezentácia ekonomických regresných modelov")

# Výber zdroja dát
st.sidebar.header("1. Nahranie alebo výber dát")
uploaded_file = st.sidebar.file_uploader("Nahrajte Excel súbor s ekonomickými údajmi", type=[".xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    st.info("Používa sa predvolený dataset o vývoji priemernej mzdy na Slovensku (2001–2024).")
    df = pd.read_excel(r"C:\Users\Martin\Desktop\škola\diplomovka\prakticka časť\dataset.xlsx")

# Zobrazenie dát
if st.checkbox("Zobraziť nahrané dáta"):
    st.subheader("Ukážka dát")
    st.dataframe(df.head())

# Výber premenných pre model
st.sidebar.header("2. Nastavenie modelu")
dep_var = st.sidebar.selectbox("Závislá premenná", df.columns)
indep_vars = st.sidebar.multiselect("Nezávislé premenné", [col for col in df.columns if col != dep_var])

# Výber knižnice
model_type = st.sidebar.selectbox("Knižnica pre regresiu", ["sklearn", "statsmodels"])

# Tréning modelu
if st.sidebar.button("Spustiť model") and indep_vars:
    X = df[indep_vars]
    y = df[dep_var]

    st.subheader("Výsledky regresného modelu")

    if model_type == "sklearn":
        model = LinearRegression()
        model.fit(X, y)
        st.write("R² score:", model.score(X, y))
        st.write("Koeficienty:", dict(zip(indep_vars, model.coef_)))
        st.write("Intercept:", model.intercept_)

    elif model_type == "statsmodels":
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        st.text(model.summary())

    # Vizualizácia predikcií
    st.subheader("Vizualizácia predikcií")
    if model_type == "sklearn":
        y_pred = model.predict(X)
    else:
        y_pred = model.predict(X_const)

    fig, ax = plt.subplots()
    sns.lineplot(x=df.index, y=y, label="Skutočné hodnoty", ax=ax)
    sns.lineplot(x=df.index, y=y_pred, label="Predikované hodnoty", ax=ax)
    st.pyplot(fig)

else:
    st.warning("Zvoľte závislú aj aspoň jednu nezávislú premennú a kliknite na 'Spustiť model'.")
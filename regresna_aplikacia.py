import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from patsy import dmatrices

# Konfigurácia vzhľadu stránky
st.set_page_config(page_title="Ekonomické regresné modely", layout="wide")

# Vítací nadpis
st.title("📈 Regresná analýza miery nezamestnanosti")
st.markdown("""
Táto aplikácia umožňuje analyzovať mieru nezamestnanosti na základe vybraných makroekonomických ukazovateľov.

Vyberte požadovanú regresnú knižnicu a premenné v postrannom paneli.
""")

# Postranný panel
st.sidebar.header("📊 Nastavenie modelu")

# Načítanie datasetu
df = pd.read_csv("updated_dataset_with_hdp_o.csv")
df["miera_nezamestanosti"] = df["miera_nezamestanosti"].astype(str).str.replace(",", ".").astype(float)
df["hdp_o"] = df["hdp_o"].astype(str).str.replace(",", ".").astype(float)
df["hdp_o_std"] = (df["hdp_o"] - df["hdp_o"].mean()) / df["hdp_o"].std()
df["kvartal"] = df["kvartal"].astype(str).str.strip()
quarter_dummies = pd.get_dummies(df["kvartal"], prefix="Q")
df = pd.concat([df, quarter_dummies], axis=1)

# Možnosť zvoliť knižnicu
model_choice = st.sidebar.selectbox("Vyber knižnicu na odhad", ["statsmodels", "sklearn", "patsy"])

# Voľba vstupných premenných (okrem závislej)
available_vars = [col for col in df.columns if col != "miera_nezamestanosti"]
selected_vars = st.sidebar.multiselect("Vyber nezávislé premenné", available_vars, default=["hdp_o_std"] + [col for col in df.columns if col.startswith("Q_")])

# Tlačidlo na spustenie analýzy
run_model = st.sidebar.button("▶️ Vykonať analýzu")

if run_model and selected_vars:
    st.subheader("📉 Výstup regresného modelu")

    if model_choice == "patsy":
        formula = "miera_nezamestanosti ~ " + " + ".join(selected_vars)
        y_patsy, X_patsy = dmatrices(formula, data=df, return_type='dataframe')
        model = sm.OLS(y_patsy, X_patsy).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        st.text(model.summary())
        y_pred = model.predict(X_patsy)
        y = y_patsy.iloc[:, 0]
        X_index = y_patsy.index

    else:
        X = df[selected_vars].apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(df["miera_nezamestanosti"], errors="coerce")
        X = X.dropna().astype(float)
        y = y.loc[X.index].astype(float)

        if model_choice == "statsmodels":
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            st.text(model.summary())
            y_pred = model.predict(X_const)
            X_index = X.index

        elif model_choice == "sklearn":
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.write(f"R² skóre: {model.score(X, y):.4f}")
            st.write("Koeficienty:", dict(zip(X.columns, model.coef_)))
            st.write("Intercept:", model.intercept_)
            X_index = X.index

    # Vizualizácia
    st.subheader("📊 Vizualizácia predikcií")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=X_index, y=y, label="Skutočná nezamestnanosť", ax=ax)
    sns.lineplot(x=X_index, y=y_pred, label="Predikovaná nezamestnanosť", ax=ax)
    ax.set_xlabel("Index")
    ax.set_ylabel("Miera nezamestnanosti (%)")
    ax.legend()
    st.pyplot(fig)

elif not selected_vars:
    st.warning("Prosím, vyberte aspoň jednu vysvetľujúcu premennú v postrannom paneli.")

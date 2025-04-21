import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Nastavenie rozhrania
st.set_page_config(page_title="Ekonomické regresné modely", layout="wide")
st.title("Regresná analýza miery nezamestnanosti")

# Načítanie preddefinovaného datasetu
df = pd.read_csv("updated_dataset_with_hdp_o.csv")

# Pred¾arobenie potrebných premenných
df["miera_nezamestanosti"] = df["miera_nezamestanosti"].astype(str).str.replace(",", ".").astype(float)
df["hdp_o"] = df["hdp_o"].astype(str).str.replace(",", ".").astype(float)
df["hdp_o_std"] = (df["hdp_o"] - df["hdp_o"].mean()) / df["hdp_o"].std()
df["kvartal"] = df["kvartal"].astype(str).str.strip()

# Tvorba dummy premenných s referenčnou kategóriou Q1
quarter_dummies = pd.get_dummies(df["kvartal"], prefix="Q")
df = pd.concat([df, quarter_dummies], axis=1)

# Pripravené vysvetľujúce premenné
X = df[["hdp_o_std", "Q_2Q", "Q_3Q", "Q_4Q"]].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df["miera_nezamestanosti"], errors="coerce")

# Odstránenie riadkov s chýbajúcimi alebo objektovými hodnotami
X = X.dropna().astype(float)
y = y.loc[X.index].astype(float)

# OLS model cez statsmodels
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

# Zobrazenie modelu
st.subheader("Štatistický výstup OLS modelu (statsmodels)")
st.text(model.summary())

# Vizualizácia predikcií
st.subheader("Vizualizácia predikovaných a skutočných hodnôt")
y_pred = model.predict(X_const)
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(x=X.index, y=y, label="Skutočná nezamestnanosť", ax=ax)
sns.lineplot(x=X.index, y=y_pred, label="Predikovaná nezamestnanosť", ax=ax)
ax.set_xlabel("Index")
ax.set_ylabel("Miera nezamestnanosti (%)")
ax.legend()
st.pyplot(fig)

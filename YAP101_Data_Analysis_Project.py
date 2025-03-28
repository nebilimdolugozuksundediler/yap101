# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 08:22:10 2025

@author: Ayşenil
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr

# Veri setini yükleme
df = pd.read_csv("games.csv")  # Dosyayı kendi sistemine kaydet ve adını güncelle
print(df.head())

# Sürekli değişkenlerin özeti
print(df[["turns", "white_rating", "black_rating", "opening_ply"]].describe())

# Kategorik değişkenlerin frekansı
print(df["victory_status"].value_counts())
print(df["winner"].value_counts())
print(df["opening_name"].value_counts().head(10))  # En yaygın 10 açılış

plt.figure(figsize=(10, 6))
sns.boxplot(x="winner", y="white_rating", data=df[df["winner"] != "draw"])
plt.title("Beyaz Oyuncuların Derecesi ve Kazanan")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="winner", y="black_rating", data=df[df["winner"] != "draw"])
plt.title("Siyah Oyuncuların Derecesi ve Kazanan")
plt.show()

top_openings = df["opening_name"].value_counts().head(5).index
plt.figure(figsize=(12, 6))
sns.countplot(x="opening_name", hue="winner", data=df[df["opening_name"].isin(top_openings)])
plt.xticks(rotation=45)
plt.title("En Yaygın 5 Açılış ve Kazananlar")
plt.show()

white_wins = df[df["winner"] == "white"]["white_rating"]
black_wins = df[df["winner"] == "black"]["white_rating"]
t_stat, p_val = ttest_ind(white_wins, black_wins)
print(f"T-Test (Beyaz Derece): t={t_stat:.2f}, p={p_val:.4f}")

df["rating_diff"] = df["white_rating"] - df["black_rating"]
corr, p_val = pearsonr(df["rating_diff"], df["turns"])
print(f"Korelasyon (Derece Farkı ve Hamle Sayısı): r={corr:.2f}, p={p_val:.4f}")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr, chi2_contingency, f_oneway


df = pd.read_csv("games_updated.1.csv")
print(df.head())

# Tanımlayıcı istatistikler
desc_stats = df[["turns", "white_rating", "black_rating", "opening_ply"]].describe().round(2)
print(desc_stats)

# Kategorik değişkenlerin sıklıkları
print(df["victory_status"].value_counts())
print(df["winner"].value_counts())
print(df["opening_shortname"].value_counts().head(10))

# Görselleştirme: En yaygın 5 açılış ve kazananlar
top_openings = df["opening_name"].value_counts().head(5).index
plt.figure(figsize=(12, 6))
sns.countplot(x="opening_name", hue="winner", data=df[df["opening_name"].isin(top_openings)])
plt.xticks(rotation=45)
plt.title("En Yaygın 5 Açılış ve Kazananlar")
plt.xlabel("Açılış Türü")
plt.ylabel("Oyun Sayısı")
plt.show()

# Derece farkı ve hamle sayısı korelasyonu
df["rating_diff"] = df["white_rating"] - df["black_rating"]
corr, p_val = pearsonr(df["rating_diff"], df["turns"])
print(f"Korelasyon (Derece Farkı ve Hamle Sayısı): r={corr:.2f}, p={p_val:.4f}")

# Derece farkını kategorilere ayırma
df["rating_diff_bin"] = pd.cut(df["rating_diff"], bins=[-float("inf"), -100, 0, 100, float("inf")])

# Ki-Kare Testi: Derece farkı ve bitiş şekli
contingency_table = pd.crosstab(df["rating_diff_bin"], df["victory_status"])
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"Ki-Kare Testi (Derece Farkı ve Bitiş Şekli): chi2={chi2:.2f}, p={p:.4f}")


# Yeni T-Test: Dereceli ve Derecesiz Oyunlarda Ortalama Hamle Sayısı
rated_turns = df[df["rated"] == True]["turns"]
unrated_turns = df[df["rated"] == False]["turns"]
t_stat, p_val = ttest_ind(rated_turns, unrated_turns)
print(f"T-Test (Dereceli vs Derecesiz Oyunlarda Hamle Sayısı): t={t_stat:.2f}, p={p_val:.4f}")

# Görselleştirme: Dereceli ve Derecesiz Oyunlarda Hamle Sayısı (Kutu Grafiği)
plt.figure(figsize=(10, 6))
sns.boxplot(x="rated", y="turns", data=df)
plt.title("Dereceli ve Derecesiz Oyunlarda Hamle Sayısı")
plt.xlabel("Oyun Türü")
plt.ylabel("Hamle Sayısı")
plt.xticks([0, 1], ["Derecesiz", "Dereceli"])  # Etiketleri değiştiriyoruz
plt.show()

# Görselleştirme: En yaygın 5 açılış ve bitiş şekli
plt.figure(figsize=(12, 6))
sns.countplot(x="opening_name", hue="victory_status", data=df[df["opening_name"].isin(top_openings)])
plt.xticks(rotation=45)
plt.title("En Yaygın 5 Açılış ve Bitiş Şekli")
plt.xlabel("Açılış Türü")
plt.ylabel("Oyun Sayısı")
plt.show()

contingency_table = pd.crosstab(df[df["opening_name"].isin(top_openings)]["opening_name"], df["winner"])
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"Ki-Kare Testi (Açılış ve Kazanan - En Yaygın 5 Açılış): chi2={chi2:.2f}, p={p:.4f}")

# Ki-Kare Testi: Açılış ve bitiş şekli (en yaygın 5 açılış için)
contingency_table = pd.crosstab(df[df["opening_name"].isin(top_openings)]["opening_name"], df["victory_status"])
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"Ki-Kare Testi (Açılış ve Bitiş Şekli - En Yaygın 5 Açılış): chi2={chi2:.2f}, p={p:.4f}")

# Yüksek dereceli oyuncuları filtreleme (rating > 1800)
high_rated_games = df[(df["white_rating"] > 1800) | (df["black_rating"] > 1800)]

# Ki-Kare Testi: Açılış ve kazanan (yüksek dereceli oyuncular)
contingency_table = pd.crosstab(high_rated_games["opening_name"], high_rated_games["winner"])
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"Ki-Kare Testi (Açılış ve Kazanan - Yüksek Dereceli): chi2={chi2:.2f}, p={p:.4f}")

# Görselleştirme: Açılışlara göre ortalama hamle sayısı
plt.figure(figsize=(12, 6))
sns.barplot(x="turns", y="opening_name", data=df[df["opening_name"].isin(top_openings)])
plt.title("Açılışlara Göre Ortalama Hamle Sayısı")
plt.xlabel("Ortalama Hamle Sayısı")
plt.ylabel("Açılış Türü")
plt.show()

# ANOVA Testi: Açılışlara göre hamle sayıları arasında fark var mı?
opening_turns = [df[df["opening_name"] == opening]["turns"] for opening in top_openings]
f_stat, p_val = f_oneway(*opening_turns)
print(f"ANOVA Testi (Açılışlara Göre Ortalama Hamle Sayısı): F={f_stat:.2f}, p={p_val:.4f}")

# Ki-Kare Testi: Dereceli oyunlar ve bitiş şekli
contingency_table = pd.crosstab(df["rated"], df["victory_status"])
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"Ki-Kare Testi (Dereceli ve Bitiş Şekli): chi2={chi2:.2f}, p={p:.4f}")

# Yeni Analiz: Derece Seviyesine Göre Açılış Tercihleri
df["average_rating"] = (df["white_rating"] + df["black_rating"]) / 2
df["rating_category"] = pd.cut(df["average_rating"], 
                               bins=[-float("inf"), 1200, 1800, float("inf")], 
                               labels=["Düşük (<1200)", "Orta (1200-1800)", "Yüksek (>1800)"])

# Görselleştirme: Derece kategorilerine göre en yaygın 5 açılış
plt.figure(figsize=(12, 6))
sns.countplot(y="opening_name", hue="rating_category", 
              data=df[df["opening_name"].isin(top_openings)], 
              order=top_openings)
plt.title("Derece Kategorilerine Göre En Yaygın 5 Açılış")
plt.xlabel("Oyun Sayısı")
plt.ylabel("Açılış Türü")
plt.show()

# Ki-Kare Testi: Derece kategorisi ve açılış tercihi
contingency_table = pd.crosstab(df["rating_category"], df["opening_name"])
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"Ki-Kare Testi (Derece Kategorisi ve Açılış Tercihi): chi2={chi2:.2f}, p={p:.4f}")

# Derece Farkı ve Daha Yüksek Dereceli Oyuncunun Kazanma Oranı
df["rating_diff_detailed"] = pd.cut(df["rating_diff"], 
                                    bins=[-float("inf"), -500, -300, -100, 0, 100, 300, 500, float("inf")], 
                                    labels=["<-500", "-500 to -300", "-300 to -100", "-100 to 0", 
                                            "0 to 100", "100 to 300", "300 to 500", ">500"])

df["higher_rated_wins"] = False
df.loc[(df["rating_diff"] > 0) & (df["winner"] == "white"), "higher_rated_wins"] = True
df.loc[(df["rating_diff"] < 0) & (df["winner"] == "black"), "higher_rated_wins"] = True
df_filtered = df[df["winner"] != "draw"]

win_rates = df_filtered.groupby("rating_diff_detailed")["higher_rated_wins"].mean() * 100  
win_rates_df = win_rates.reset_index()
win_rates_df.columns = ["rating_diff_detailed", "win_rate"]

plt.figure(figsize=(12, 6))
sns.barplot(x="rating_diff_detailed", y="win_rate", data=win_rates_df, color="skyblue")
plt.title("Derece Farkına Göre Daha Yüksek Dereceli Oyuncunun Kazanma Oranı")
plt.xlabel("Derece Farkı Kategorisi (Beyaz - Siyah)")
plt.ylabel("Kazanma Oranı (%)")
plt.xticks(rotation=45)
plt.ylim(0, 100)  
plt.show()

# Ki-Kare Testi: Derece farkı kategorisi ve daha yüksek dereceli oyuncunun kazanma durumu
contingency_table = pd.crosstab(df_filtered["rating_diff_detailed"], df_filtered["higher_rated_wins"])
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"Ki-Kare Testi (Derece Farkı ve Daha Yüksek Dereceli Oyuncunun Kazanma Durumu): chi2={chi2:.2f}, p={p:.4f}")

print("\nDerece Farkı Kategorilerine Göre Daha Yüksek Dereceli Oyuncunun Kazanma Oranları:")
for _, row in win_rates_df.iterrows():
    print(f"{row['rating_diff_detailed']}: {row['win_rate']:.2f}%")
    
plt.figure(figsize=(10, 6))
plt.hexbin(df["rating_diff"], df["turns"], gridsize=50, cmap="Blues")
plt.colorbar(label="Oyun Sayısı")
plt.xlabel("Derece Farkı")
plt.ylabel("Hamle Sayısı")
plt.title("Derece Farkı ve Hamle Sayısı İlişkisi")
plt.show()

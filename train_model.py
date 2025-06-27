# File: train_model.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib # Untuk menyimpan model

print("Memulai proses training...")

# 1. Muat dan siapkan data
df = pd.read_csv('diamonds_cleaned.csv')
cut_ranking = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_ranking = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_ranking = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
df['cut_encoded'] = df['cut'].apply(lambda x: cut_ranking.index(x))
df['color_encoded'] = df['color'].apply(lambda x: color_ranking.index(x))
df['clarity_encoded'] = df['clarity'].apply(lambda x: clarity_ranking.index(x))
print("Data berhasil disiapkan.")

# 2. Buat dan simpan Heatmap Korelasi
plt.figure(figsize=(12, 8))
numerical_df = df.select_dtypes(include=np.number)
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Variabel Numerik', fontsize=16)
# Pastikan folder static/images sudah ada
import os
if not os.path.exists('static/images'):
    os.makedirs('static/images')
plt.savefig('static/images/correlation_heatmap.png', bbox_inches='tight')
print("Heatmap korelasi berhasil disimpan di 'static/images/correlation_heatmap.png'")

# 3. Latih model regresi berganda
features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_encoded', 'color_encoded', 'clarity_encoded']
X = df[features]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
print("Model regresi berganda berhasil dilatih.")

# 4. Simpan model yang sudah dilatih ke dalam file
joblib.dump(model_multi, 'model_regresi.pkl')
print("Model berhasil disimpan sebagai 'model_regresi.pkl'")
print("\nProses training selesai!")
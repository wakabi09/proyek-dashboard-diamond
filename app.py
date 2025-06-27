from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# --- Muat Model dan Data sekali saja saat aplikasi dimulai ---
model = joblib.load('model_regresi.pkl')
df_cleaned = pd.read_csv('diamonds_cleaned.csv')

# --- Siapkan data untuk dropdown di form prediksi ---
cut_ranking = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_ranking = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_ranking = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']


@app.route('/')
def index():
    total_diamonds = len(df_cleaned)
    avg_price = df_cleaned['price'].mean()
    avg_carat = df_cleaned['carat'].mean()
    return render_template(
        'index.html',
        total_diamonds=f"{total_diamonds:,}",
        avg_price=f"${avg_price:,.2f}",
        avg_carat=f"{avg_carat:.2f} ct"
    )

@app.route('/eda')
def eda():
    # Proses data untuk semua chart di halaman EDA
    hist_data, bin_edges = np.histogram(df_cleaned['price'], bins=50)
    hist_labels = [f"${int(bin_edges[i])}-${int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    hist_counts = hist_data.tolist()
    cut_counts = df_cleaned['cut'].value_counts().reindex(cut_ranking)
    color_counts = df_cleaned['color'].value_counts().reindex(color_ranking)
    clarity_counts = df_cleaned['clarity'].value_counts().reindex(clarity_ranking)
    median_price_cut = df_cleaned.groupby('cut')['price'].median().reindex(cut_ranking)
    median_price_color = df_cleaned.groupby('color')['price'].median().reindex(color_ranking)
    median_price_clarity = df_cleaned.groupby('clarity')['price'].median().reindex(clarity_ranking)
    scatter_data_sample = df_cleaned.sample(500, random_state=1)
    scatter_chart_data = [{'x': row['carat'], 'y': row['price']} for index, row in scatter_data_sample.iterrows()]
    
    return render_template(
        'eda.html',
        hist_labels=hist_labels, hist_counts=hist_counts,
        cut_labels=cut_counts.index.tolist(), cut_data=cut_counts.values.tolist(),
        color_labels=color_counts.index.tolist(), color_data=color_counts.values.tolist(),
        clarity_labels=clarity_counts.index.tolist(), clarity_data=clarity_counts.values.tolist(),
        median_price_cut_labels=median_price_cut.index.tolist(), median_price_cut_data=median_price_cut.values.tolist(),
        median_price_color_labels=median_price_color.index.tolist(), median_price_color_data=median_price_color.values.tolist(),
        median_price_clarity_labels=median_price_clarity.index.tolist(), median_price_clarity_data=median_price_clarity.values.tolist(),
        scatter_chart_data=scatter_chart_data,
    )

@app.route('/korelasi')
def korelasi():
    return render_template('korelasi.html')

@app.route('/model')
def model_results():
    return render_template('model.html')

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    prediction_result = None
    if request.method == 'POST':
        # 1. Ambil data dari form
        carat = float(request.form['carat'])
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']

        # 2. Encode data kategorikal
        cut_encoded = cut_ranking.index(cut)
        color_encoded = color_ranking.index(color)
        clarity_encoded = clarity_ranking.index(clarity)

        # 3. Buat array numpy sesuai urutan fitur saat training
        features = np.array([[carat, depth, table, x, y, z, cut_encoded, color_encoded, clarity_encoded]])
        
        # 4. Lakukan prediksi
        prediction = model.predict(features)
        prediction_result = f"{prediction[0]:,.2f}"

    return render_template(
        'prediksi.html', 
        prediction=prediction_result,
        cut_options=cut_ranking,
        color_options=color_ranking,
        clarity_options=clarity_ranking
    )

if __name__ == '__main__':
    app.run(debug=True)
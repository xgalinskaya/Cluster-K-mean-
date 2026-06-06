import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans # Импортируем KMeans

st.set_page_config(page_title="Kraljic Matrix with K-Means", layout="wide")

RISK_COLS = [
    'Performance_Quality_Risk_Score', 
    'Financial_Risk_Score', 
    'Nachhaltigkeit_Risk_score', 
    'Standards Risks_Score', 
    'Political_Risk_Score'
]

@st.cache_data
def load_data():
    df = pd.read_csv('Merged dataset with Scores.csv', sep=';', decimal=',')
    df['Order Value USD'] = (
        df['Order Value USD'].astype(str).str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False).astype(float)
    )
    return df

def main():
    st.title("Sustainable Supply Chain: K-Means Kraljic Matrix")
    df = load_data()

    # --- SIDEBAR ---
    st.sidebar.header("Configuration")
    selected_cat = st.sidebar.selectbox("Select Product Category", sorted(df['Product_Category'].unique()))
    selected_timeframe = st.sidebar.selectbox("Select Timeframe", ["All Months"] + sorted(df['Month'].unique().astype(str).tolist()))

    # --- RISK WEIGHTS ---
    st.sidebar.header("Risk Weights")
    weights = [st.sidebar.number_input(col.replace('_Score', ''), 0, 100, 20, step=5) for col in RISK_COLS]
    if sum(weights) != 100:
        st.sidebar.error(f"Sum must be 100! Current: {sum(weights)}")
        st.stop()
    weights = np.array(weights) / 100

    # --- DATA PIPELINE ---
    subset = df[df['Product_Category'] == selected_cat].copy()
    if selected_timeframe != "All Months":
        subset = subset[subset['Month'].astype(str) == selected_timeframe]
    
    agg_df = subset.groupby('Supplier_ID').agg(
        {'Order Value USD': 'sum', 'Country': 'first', **{col: 'mean' for col in RISK_COLS}}
    ).reset_index()

    # Нормализация для K-Means
    scaler = MinMaxScaler()
    agg_df['Norm_Spend'] = scaler.fit_transform(agg_df[['Order Value USD']])
    agg_df['Weighted_Risk'] = (agg_df[RISK_COLS].values * weights).sum(axis=1)

    # --- K-MEANS CLUSTERING ---
    # Обучаем модель на 4 кластера
    X = agg_df[['Norm_Spend', 'Weighted_Risk']]
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    agg_df['Cluster'] = kmeans.fit_predict(X)
    
    # Чтобы названия кластеров соответствовали Kraljic, можно отсортировать их по центроидам
    # Либо просто использовать номера кластеров
    agg_df['Kraljic_Segment'] = agg_df['Cluster'].astype(str)

    # --- VISUALIZATION ---
    fig = px.scatter(
        agg_df, x="Norm_Spend", y="Weighted_Risk", color="Kraljic_Segment",
        hover_data=['Supplier_ID'], title="Kraljic Segmentation via K-Means"
    )
    
    # Добавляем центроиды на график
    centroids = kmeans.cluster_centers_
    fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', 
                    marker=dict(size=15, symbol='x', color='black'), name='Centroids')

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Таблица данных
    st.subheader("Supplier Data")
    st.dataframe(agg_df[['Supplier_ID', 'Order Value USD', 'Weighted_Risk', 'Kraljic_Segment']])

if __name__ == "__main__":
    main()
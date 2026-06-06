import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Sustainable Supply Chain Matrix", layout="wide")

RISK_COLS = ['Performance_Quality_Risk_Score', 'Financial_Risk_Score', 'Nachhaltigkeit_Risk_score', 'Standards Risks_Score', 'Political_Risk_Score']

@st.cache_data
def load_data():
    df = pd.read_csv('Merged dataset with Scores.csv', sep=';', decimal=',')
    df['Order Value USD'] = df['Order Value USD'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
    return df

def get_prepared_data(df, category, timeframe, weights):
    subset = df[df['Product_Category'] == category].copy()
    if timeframe != "All Months":
        subset = subset[subset['Month'].astype(str) == timeframe]
    
    agg = subset.groupby('Supplier_ID').agg(
        {'Order Value USD': 'sum', 'Country': 'first', **{col: 'mean' for col in RISK_COLS}}
    ).reset_index()
    
    # Используем глобальный скейлер (можно сохранить параметры скейлера для консистентности)
    agg['Norm_Spend'] = (agg['Order Value USD'] - df['Order Value USD'].min()) / (df['Order Value USD'].max() - df['Order Value USD'].min())
    agg['Weighted_Risk'] = (agg[RISK_COLS].values * weights).sum(axis=1)
    return agg

def main():
    df = load_data()
    
    # 1. Обучение модели на годовых данных (один раз)
    if 'kmeans_model' not in st.session_state:
        full_agg = df.groupby('Supplier_ID').agg({**{col: 'mean' for col in RISK_COLS}, 'Order Value USD': 'sum'})
        scaler = MinMaxScaler()
        X_full = pd.DataFrame({
            'Norm_Spend': scaler.fit_transform(full_agg[['Order Value USD']]).flatten(),
            'Weighted_Risk': (full_agg[RISK_COLS].values * 0.2).sum(axis=1) # средние веса
        })
        model = KMeans(n_clusters=4, random_state=42).fit(X_full)
        st.session_state.kmeans_model = model
        st.session_state.scaler = scaler

    # --- SIDEBAR ---
    st.sidebar.header("Configuration")
    selected_cat = st.sidebar.selectbox("Category", sorted(df['Product_Category'].unique()))
    selected_timeframe = st.sidebar.selectbox("Timeframe", ["All Months"] + sorted(df['Month'].unique().astype(str).tolist()))
    
    weights = np.array([st.sidebar.number_input(col.replace('_Score', ''), 0, 100, 20) for col in RISK_COLS]) / 100

    # --- DATA ---
    agg_df = get_prepared_data(df, selected_cat, selected_timeframe, weights)
    
    # Применение границ (предсказание кластера)
    X_current = agg_df[['Norm_Spend', 'Weighted_Risk']]
    agg_df['Kraljic_Segment'] = st.session_state.kmeans_model.predict(X_current)

    # --- VISUALIZATION ---
    fig = px.scatter(
        agg_df, x="Norm_Spend", y="Weighted_Risk", color="Kraljic_Segment",
        hover_data=['Supplier_ID'], custom_data=['Supplier_ID'],
        title="Sustainable Supply Chain: Category-Specific Kraljic Matrix"
    )
    
    # Интерактивность (выбор точки)
    event = st.plotly_chart(fig, on_select="rerun", use_container_width=True)

    # --- DETAILS ---
    if event and event["selection"]["points"]:
        supplier_id = event["selection"]["points"][0]["customdata"][0]
        st.subheader(f"Supplier: {supplier_id}")
        data = agg_df[agg_df['Supplier_ID'] == supplier_id].iloc[0]
        st.write(f"Country: {data['Country']}")
        for col in RISK_COLS:
            st.progress(float(data[col]), text=f"{col.replace('_Score', '')}: {data[col]:.2f}")

if __name__ == "__main__":
    main()
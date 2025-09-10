import joblib
import cloudpickle

# Load mô hình và dữ liệu
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
df_info = joblib.load("models/df_info.pkl")

with open("models/recommend_function.pkl", "rb") as f:
    recommend_hotels_by_description_sklearn = cloudpickle.load(f)

# Giao diện Streamlit
import streamlit as st

st.title("Gợi ý khách sạn theo mô tả của bạn")

query = st.text_input("Nhập mô tả khách sạn bạn muốn tìm:")

if query:
    results = recommend_hotels_by_description_sklearn(query)
    st.write(results)

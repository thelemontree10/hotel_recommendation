import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load mô hình và dữ liệu
with open("df_info.pkl", "rb") as f:
    df_info = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


def recommend_hotels_by_description_sklearn(
    query_text, top_n=5, alpha=0.5, beta=0.2, gamma=0.15, delta=0.15
):
    query_vec = tfidf.transform([query_text])
    tfidf_matrix = tfidf.transform(df_info["Clean_Description"])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    df_info["Similarity_Score"] = sim_scores
    df_info["Final_Score"] = (
        alpha * df_info["Similarity_Score"]
        + beta * df_info["Rank_Score"]
        + gamma * df_info["Comment_Score"]
        + delta * df_info["Total_Score_Score"]
    )
    top_hotels = df_info.sort_values(by="Final_Score", ascending=False).head(top_n)
    return top_hotels[
        [
            "Hotel_Name",
            "Hotel_Rank",
            "Total_Score",
            "comments_count",
            "Clean_Description",
            "Final_Score",
        ]
    ]


# Giao diện Streamlit
import streamlit as st

st.title("Gợi ý khách sạn theo mô tả của bạn")

query = st.text_input("Nhập mô tả khách sạn bạn muốn tìm:")

if query:
    results = recommend_hotels_by_description_sklearn(query)
    st.write(results)

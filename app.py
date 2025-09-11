import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st
import unicodedata
import string
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator

from textblob import TextBlob

import nltk

nltk.data.path.append("nltk_data")
from nltk.corpus import stopwords

from langdetect import detect
from underthesea import sentiment, pos_tag
from wordcloud import WordCloud


# Load mô hình và dữ liệu
df_info = pd.read_pickle("df_info_Recommendation.pkl")
tfidf = pd.read_pickle("tfidf_vectorizer_Recommendation.pkl")


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


def extract_date(text):
    if not isinstance(text, str):
        return None
    match = re.search(r"(\d{1,2}) tháng (\d{1,2}) (\d{4})", text)
    if match:
        day, month, year = match.groups()
        return f"{day}/{month}/{year}"
    return None


# Load các từ điển hỗ trợ
def load_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return dict(line.strip().split("\t") for line in lines if "\t" in line)


# nltk.download("punkt", quiet=True)
# nltk.download("stopwords", quiet=True)

teencode_dict = load_dict("teencode.txt")
emoji_dict = load_dict("emojicon.txt")
english_vn_dict = load_dict("english-vnmese.txt")
stopwords_data = set(
    open("vietnamese-stopwords.txt", encoding="utf-8").read().splitlines()
)

# Tạo bộ dịch ký tự đặc biệt và số
translator = str.maketrans(
    string.punctuation + string.digits,
    " " * (len(string.punctuation) + len(string.digits)),
)

# Tạo regex tổng hợp cho emoji, teencode, và english-vn
emoji_pattern = re.compile("|".join(map(re.escape, emoji_dict.keys())))
teen_pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, teencode_dict.keys())) + r")\b"
)
eng_pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, english_vn_dict.keys())) + r")\b"
)


# Hàm chuẩn hóa văn bản
def normalize_text_old(text):
    if not isinstance(text, str):
        return ""

    # Chuẩn hóa unicode
    text = unicodedata.normalize("NFKC", text)

    # Loại bỏ ký tự đặc biệt và số
    text = text.translate(translator)

    # Thay thế emoji
    text = emoji_pattern.sub(lambda m: f" {emoji_dict[m.group()]} ", text)

    # Thay thế teencode
    text = teen_pattern.sub(lambda m: teencode_dict[m.group()], text)

    # Dịch từ tiếng Anh sang tiếng Việt
    text = eng_pattern.sub(lambda m: english_vn_dict[m.group()], text)

    # ✅ Tokenize bằng PyVi
    text = ViTokenizer.tokenize(text)

    # Chuyển về chữ thường
    text = text.lower()

    # Loại bỏ stopwords
    words = text.split()
    words = [w for w in words if w not in stopwords_data and len(w) > 1]

    return " ".join(words)


def show_hotel_info(hotel_id):
    hotel = df_info[df_info["Hotel_ID"].astype(str) == hotel_id].iloc[0]
    st.write(f"\n📌 Khách sạn: {hotel['Hotel_Name']} (ID: {hotel_id})")
    st.write(f"🏅 Hạng: {hotel['Hotel_Rank']}")
    st.write(f"📍 Địa chỉ: {hotel['Hotel_Address']}")
    st.write(f"⭐ Tổng điểm: {hotel['Total_Score']}")
    st.write(f"📝 Mô tả: {hotel['Hotel_Description'][:300]}...")


def analyze_strengths_and_weaknesses(hotel_id):
    # Lấy thông tin khách sạn
    hotel = df_info[df_info["Hotel_ID"].astype(str) == hotel_id].iloc[0]
    hotel_comments = df_comments[df_comments["Hotel ID"].astype(str) == hotel_id]

    # Các cột điểm chi tiết
    score_cols = [
        "Location",
        "Cleanliness",
        "Service",
        "Facilities",
        "Value_for_money",
        "Comfort_and_room_quality",
    ]

    # Tính trung bình hệ thống
    system_avg = df_info[score_cols].mean()

    print("\n📊 Phân tích điểm mạnh & điểm yếu:")

    for col in score_cols:
        hotel_score = hotel[col]
        avg_score = system_avg[col]

        if pd.notnull(hotel_score):
            diff = hotel_score - avg_score
            status = "✅ Điểm mạnh" if diff > 0 else "⚠️ Điểm yếu"
            st.write(
                f"- {col.replace('_', ' ').title()}: {hotel_score:.2f} ({status}, trung bình hệ thống: {avg_score:.2f})"
            )
        else:
            st.write(f"- {col.replace('_', ' ').title()}: Không có dữ liệu")

    # Phân tích số lượng nhận xét
    num_reviews = len(hotel_comments)
    st.write(f"\n🧮 Số lượng nhận xét: {num_reviews} lượt")

    # Phân tích nội dung nhận xét (từ khóa nổi bật)
    hotel_comments["Normalized_Text"] = hotel_comments["Review_Text"].apply(
        normalize_text_old
    )
    hotel_comments = hotel_comments[hotel_comments["Normalized_Text"].str.strip() != ""]

    vectorizer = TfidfVectorizer(max_features=15)
    X = vectorizer.fit_transform(hotel_comments["Normalized_Text"])
    keywords = vectorizer.get_feature_names_out()

    st.write("\n🗣️ Từ khóa nổi bật trong nhận xét:")
    st.write(", ".join(keywords))


def analyze_customers(hotel_id):
    # Lọc dữ liệu theo Hotel ID
    hotel_comments = df_comments[df_comments["Hotel ID"].astype(str) == str(hotel_id)]

    if hotel_comments.empty:
        st.write(f"❌ Không tìm thấy dữ liệu cho Hotel ID: {hotel_id}")
        return

    # 1️⃣ Quốc tịch phổ biến
    top_nationalities = hotel_comments["Nationality"].value_counts().head(5)
    st.write("\n🌍 Quốc tịch phổ biến:")
    for nat, count in top_nationalities.items():
        print(f"- {nat}: {count} lượt")

    # Vẽ biểu đồ tròn quốc tịch
    plt.figure(figsize=(6, 6))
    plt.pie(
        top_nationalities.values,
        labels=top_nationalities.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title(f"🌍 Quốc tịch phổ biến - Hotel ID: {hotel_id}")
    plt.axis("equal")
    plt.tight_layout()
    st.pyplot(plt)

    # 2️⃣ Nhóm khách phổ biến
    if "Group Name" in hotel_comments.columns:
        top_groups = hotel_comments["Group Name"].value_counts().head(5)
        st.write("\n👥 Nhóm khách phổ biến:")
        for grp, count in top_groups.items():
            st.write(f"- {grp}: {count} lượt")

        # Vẽ biểu đồ tròn nhóm khách
        plt.figure(figsize=(6, 6))
        plt.pie(
            top_groups.values,
            labels=top_groups.index,
            autopct="%1.1f%%",
            startangle=140,
        )
        plt.title(f"👥 Nhóm khách phổ biến - Hotel ID: {hotel_id}")
        plt.axis("equal")
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.write("\n⚠️ Không có cột 'Group Name' để phân loại nhóm khách.")

    # 3️⃣ Xu hướng theo thời gian + Vẽ biểu đồ
    if "Review_Month" in hotel_comments.columns:
        monthly_trend = hotel_comments.groupby("Review_Month").size()

        if monthly_trend.empty:
            st.write("\n⚠️ Không có dữ liệu hợp lệ để vẽ biểu đồ.")
            return

        st.write("\n📈 Xu hướng đánh giá theo thời gian:")

        # Chuyển Period về Timestamp để vẽ
        monthly_trend.index = monthly_trend.index.to_timestamp()

        # Vẽ biểu đồ đường
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=monthly_trend.index, y=monthly_trend.values, marker="o", color="darkcyan"
        )
        plt.title(
            f"📈 Xu hướng đánh giá theo thời gian - Hotel ID: {hotel_id}", fontsize=14
        )
        plt.xlabel("Tháng", fontsize=12)
        plt.ylabel("Số lượt đánh giá", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.write("\n⚠️ Không có cột 'Review_Month' để phân tích thời gian.")


def normalize_text(text, lang):
    try:
        if not isinstance(text, str):
            return ""

        # Nếu là tiếng Anh, dịch sang tiếng Việt trước
        if lang == "en":
            # Loại bỏ stopwords tiếng Anh
            stopwords_en = set(stopwords.words("english"))
            words = [
                word
                for word in nltk.word_tokenize(text)
                if word.lower() not in stopwords_en and len(word) > 2
            ]
            filtered_text = " ".join(words).lower()

            # Dịch từng từ sang tiếng Việt nếu có trong từ điển
            translated_text = eng_pattern.sub(
                lambda m: english_vn_dict.get(m.group(), m.group()), filtered_text
            )

            # Gán lại text và chuyển sang xử lý như tiếng Việt
            text = translated_text
            lang = "vi"  # chuyển sang pipeline tiếng Việt

        if lang == "vi":
            # Chuẩn hóa và thay thế
            text = unicodedata.normalize("NFKC", text)
            text = text.translate(translator)
            text = emoji_pattern.sub(lambda m: f" {emoji_dict[m.group()]} ", text)
            text = teen_pattern.sub(lambda m: teencode_dict[m.group()], text)
            text = eng_pattern.sub(lambda m: english_vn_dict[m.group()], text)

            # Tách từ và lọc stopwords
            text = ViTokenizer.tokenize(text).lower()
            words = [w for w in text.split() if w not in stopwords_data and len(w) > 1]
            cleaned_text = " ".join(words)

            # POS tagging
            tagged = pos_tag(cleaned_text)

            selected = []
            phrases = []

            for i in range(len(tagged) - 1):
                word1, tag1 = tagged[i]
                word2, tag2 = tagged[i + 1]

                if tag1 == "R" and tag2 == "A":
                    phrases.append(f"{word1.lower()}_{word2.lower()}")

            for word, tag in tagged:
                if tag in ["A", "R"]:
                    selected.append(word.lower())

            # Loại từ đơn nếu đã nằm trong cụm
            flattened = set()
            for phrase in phrases:
                flattened.update(phrase.split("_"))
            selected = [word for word in selected if word not in flattened]

            return " ".join(phrases + selected)

        else:
            return text

    except Exception as e:
        st.write(f"⚠️ Lỗi normalize_text: {e}")
        return text


def classify_keywords(keywords: str, lang: str):
    results = {}
    for kw in keywords.split():
        try:
            clean_kw = kw.replace("_", " ")  # ✅ bỏ dấu gạch dưới để phân tích đúng
            valid_labels = ["positive", "negative"]
            
            if lang == "vi":
                # Dịch sang tiếng Anh
                translated_kw = GoogleTranslator(source='vi', target='en').translate(clean_kw)
                polarity = TextBlob(translated_kw).sentiment.polarity
                sentiment_label = (
                    "positive"
                    if polarity > 0.1
                    else "negative" if polarity < -0.1 else "neutral"
                )
                results[kw] = sentiment_label

            elif lang == "en":
                polarity = TextBlob(clean_kw).sentiment.polarity
                sentiment_label = (
                    "positive"
                    if polarity > 0.1
                    else "negative" if polarity < -0.1 else "neutral"
                )
                results[kw] = sentiment_label

            else:
                results[kw] = "neutral"

        except:
            results[kw] = "neutral"
    return results


def analyze_keywords(hotel_id):
    # Lọc nhận xét theo Hotel ID
    hotel_comments = df_comments[df_comments["Hotel ID"].astype(str) == str(hotel_id)]

    if hotel_comments.empty:
        st.write(f"❌ Không tìm thấy nhận xét cho Hotel ID: {hotel_id}")
        return

    # Loại nhận xét trống
    hotel_comments = hotel_comments[
        hotel_comments["Review_Text"].notnull()
        & (hotel_comments["Review_Text"].str.strip() != "")
    ]

    # Nhận diện ngôn ngữ
    def detect_language(text):
        try:
            return detect(text)
        except:
            return "unknown"

    hotel_comments["Lang"] = hotel_comments["Review_Text"].apply(detect_language)

    # Chuẩn hóa văn bản
    hotel_comments["Processed_Text"] = hotel_comments.apply(
        lambda row: normalize_text(row["Review_Text"], row["Lang"]), axis=1
    )

    # Phân tích cảm xúc từng nhận xét
    def classify_sentiment(row):
        text = row["Processed_Text"]
        lang = row["Lang"]
        try:
            if lang == "vi":
                translated = GoogleTranslator(source="vi", target="en").translate(text)
                polarity = TextBlob(translated).sentiment.polarity
                return (
                    "positive"
                    if polarity > 0.1
                    else "negative" if polarity < -0.1 else "neutral"
                )
            elif lang == "en":
                polarity = TextBlob(text).sentiment.polarity
                return (
                    "positive"
                    if polarity > 0.1
                    else "negative" if polarity < -0.1 else "neutral"
                )
            else:
                return "neutral"
        except:
            return "neutral"

    hotel_comments["Sentiment"] = hotel_comments.apply(classify_sentiment, axis=1)
    
    # Gom tất cả từ khóa đã chuẩn hóa
    all_keywords = []
    for _, row in hotel_comments.iterrows():
        kw_dict = classify_keywords(row["Processed_Text"], row["Lang"])
        for kw, senti in kw_dict.items():
            if senti in ["positive", "negative"]:
                all_keywords.append((kw, senti))
    
    # Gom từ khóa theo cảm xúc
    pos_keywords = set([kw for kw, senti in all_keywords if senti == "positive"])
    neg_keywords = set([kw for kw, senti in all_keywords if senti == "negative"])

    # Loại bỏ giao nhau
    shared = pos_keywords & neg_keywords
    pos_keywords -= shared
    neg_keywords -= shared

    # In kết quả
    st.write("\n✅ Từ khóa tích cực:")
    st.write(
        ", ".join(sorted(pos_keywords))
        if pos_keywords
        else "Không có từ tích cực nổi bật."
    )

    st.write("\n⚠️ Từ khóa tiêu cực:")
    st.write(
        ", ".join(sorted(neg_keywords))
        if neg_keywords
        else "Không có từ tiêu cực nổi bật."
    )

    # Vẽ Word Cloud
    def show_wordcloud(keywords, title, color="black"):
        if not keywords:
            print(f"⚠️ Không có từ khóa để vẽ Word Cloud cho: {title}")
            return
        text = " ".join(keywords)
        wc = WordCloud(
            width=800, height=400, background_color="white", colormap=color
        ).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=16)
        plt.tight_layout()
        st.pyplot(plt)

    show_wordcloud(pos_keywords, "✅ Word Cloud - Từ khóa tích cực", color="Greens")
    show_wordcloud(neg_keywords, "⚠️ Word Cloud - Từ khóa tiêu cực", color="Reds")


def compare_to_system(hotel_id):
    hotel = df_info[df_info["Hotel_ID"].astype(str) == hotel_id].iloc[0]
    system_avg = df_info[score_cols].mean()

    st.write("\n📊 So sánh điểm từng tiêu chí với trung bình hệ thống:")
    for col in score_cols:
        if pd.notnull(hotel[col]):
            diff = hotel[col] - system_avg[col]
            st.write(
                f"- {col.replace('_', ' ').title()}: {diff:+.2f} điểm so với trung bình"
            )


st.set_page_config(page_title="Hotel Recommender", layout="wide")

# Sidebar menu
menu = st.sidebar.selectbox(
    "📂 Chọn mục",
    [
        "Business Problem",
        "Evaluation & Report",
        "Recommendation",
        "Hotel Insight by Hotel ID",
        "Thông tin nhóm",
    ],
)

if menu == "Business Problem":
    st.header("🎯 Business Problem")
    st.markdown(
        """
    #### 🏨 Về Agoda
    - Agoda là một trang web đặt phòng trực tuyến có trụ sở tại Singapore, được thành lập vào năm 2005, thuộc sở hữu của **Booking Holdings Inc.**
    - Agoda chuyên cung cấp dịch vụ đặt phòng khách sạn, căn hộ, nhà nghỉ và các loại hình lưu trú trên toàn cầu.
    - Trang web này cho phép người dùng tìm kiếm, so sánh và đặt chỗ ở với mức giá ưu đãi.

    #### 🤖 Bài toán đặt ra
    - Giả sử Agoda **chưa triển khai hệ thống Recommender System** giúp đề xuất khách sạn/resort phù hợp tới người dùng.
    - Yêu cầu xây dựng hệ thống gợi ý thông minh dựa trên thông tin mô tả, đánh giá, vị trí, và hành vi người dùng.

    #### 📊 Góc nhìn từ phía chủ khách sạn
    - Chủ khách sạn muốn nắm rõ **insight từ khách hàng**: họ thích gì, đánh giá ra sao, mô tả nào thu hút nhất.
    - App cung cấp cho họ các phân tích, bảng tổng hợp, và công cụ giúp tối ưu hóa trải nghiệm người dùng.

    #### 🎓 Bối cảnh đồ án
    - Đây là một phần trong **Đồ án tốt nghiệp ngành Data Science**, nơi bạn vận dụng NLP, Machine Learning và trực quan hóa dữ liệu để giải quyết bài toán thực tế.
    """
    )

    st.markdown(
        """
    #### 🎯 Mục tiêu giải pháp
    - Xây dựng hệ thống đề xuất thông minh nhằm hỗ trợ người dùng **nhanh chóng chọn được nơi lưu trú phù hợp** trên nền tảng Agoda.
    - Tăng trải nghiệm người dùng, giảm thời gian tìm kiếm, và nâng cao tỷ lệ chuyển đổi đặt phòng.

    #### 🧩 Kiến trúc hệ thống gợi ý
    Hệ thống sẽ bao gồm **hai mô hình gợi ý chính**:

    - 🔍 **Content-based Filtering**  
      Dựa trên nội dung mô tả, đặc điểm của khách sạn (vị trí, tiện nghi, điểm đánh giá...) để gợi ý các khách sạn tương tự với nhu cầu người dùng.

    - 👥 **Collaborative Filtering**  
      Dựa trên hành vi và đánh giá của người dùng khác có sở thích tương đồng để đưa ra gợi ý cá nhân hóa.

    #### 📊 Cung cấp insight cho chủ khách sạn
    - Phân tích thông tin khách hàng dành cho từng khách sạn để giúp chủ khách sạn hiểu rõ:
        - ✅ **Điểm mạnh**: những yếu tố được khách hàng đánh giá cao
        - ⚠️ **Điểm cần cải thiện**: những vấn đề thường bị phản ánh hoặc có điểm số thấp
    - Từ đó, chủ khách sạn có thể điều chỉnh dịch vụ, mô tả, hoặc chiến lược giá để thu hút thêm khách hàng tiềm năng.

    #### 🛠 Công nghệ sử dụng
    - Xử lý ngôn ngữ tự nhiên (NLP) để phân tích mô tả và đánh giá
    - TF-IDF, cosine similarity, matrix factorization
    - Trực quan hóa dữ liệu bằng Streamlit
    """
    )

elif menu == "Evaluation & Report":
    st.header("📊 Evaluation & Report")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 Content-based Filtering: Cosine Similarity",
            "📚 Content-based Filtering: Gensim",
            "👥 Collaborative Filtering: ALS",
            "⚖️ So sánh model",
        ]
    )

    with tab1:
        st.subheader(
            "🔍 Đề xuất người dùng với Content-based Filtering – Phương pháp Cosine Similarity"
        )
        st.markdown(
            """
        - Ý tưởng chính của phương pháp này là đưa ra gợi ý dựa vào **sự tương đồng giữa các sản phẩm**.
        - Mỗi mô tả khách sạn được vector hóa bằng TF-IDF.
        - Sau đó, tính toán độ tương đồng giữa truy vấn người dùng và các mô tả bằng **cosine similarity**.
        - Những khách sạn có điểm tương đồng cao nhất sẽ được gợi ý.
        """
        )
        st.image(
            "cosine_similarity.jpg",
            caption="Minh họa phương pháp Cosine Similarity",
            use_column_width=True,
        )

    with tab2:
        st.subheader(
            "📚 Đề xuất người dùng với Content-based Filtering – Phương pháp Gensim"
        )
        st.markdown(
            """
        - **Gensim** là một thư viện Python chuyên xác định sự tương tự về ngữ nghĩa giữa hai tài liệu thông qua mô hình không gian vector và bộ công cụ mô hình hóa chủ đề.
        - Có thể xử lý kho dữ liệu văn bản lớn với sự trợ giúp của việc truyền dữ liệu hiệu quả và các thuật toán tăng cường.
        - Tốc độ xử lý và tối ưu hóa việc sử dụng bộ nhớ tốt.
        - Tuy nhiên, Gensim có **ít tùy chọn tùy biến** cho các function so với scikit-learn.
        """
        )
        st.image(
            "gensim.jpg", caption="Minh họa phương pháp Gensim", use_column_width=True
        )

    with tab3:
        st.subheader("👥 Collaborative Filtering với ALS (Alternating Least Squares)")
        st.markdown(
            """
        - **ALS** là một kỹ thuật mạnh mẽ trong hệ thống gợi ý, đặc biệt hiệu quả khi xử lý **dữ liệu lớn và thưa (sparse data)**.
        - Đây là một dạng **Matrix Factorization** giúp dự đoán mức độ yêu thích của người dùng đối với sản phẩm mà họ chưa từng tương tác.
        - ALS chia ma trận tương tác thành hai ma trận nhỏ hơn: một đại diện cho người dùng, một cho sản phẩm.
        - Bằng cách tối ưu hóa luân phiên (alternating), mô hình học được các đặc trưng tiềm ẩn và đưa ra gợi ý chính xác.

        #### 📉 Đánh giá mô hình
        - RMSE (Root Mean Square Error) được sử dụng để đánh giá độ chính xác của mô hình.
        - RMSE càng thấp → mô hình dự đoán càng sát với thực tế.
        """
        )
        st.image(
            "rmse_als.jpg",
            caption="Biểu đồ RMSE của mô hình ALS",
            use_column_width=True,
        )
        st.image(
            "ALS_minhhoa.jpg", caption="Minh họa kiến trúc ALS", use_column_width=True
        )

    with tab4:
        st.subheader("⚖️ So sánh:")
        st.markdown(
            """
        #### 📈 Nhận xét tổng quan
        - Điểm RMSE của mô hình ALS là 0.6, hơi cao nên không tốt
        - Có thể thấy **TF-IDF thường cho điểm cao hơn Gensim** khi tính độ tương đồng.
        - Điều này xuất phát từ bản chất của hai phương pháp:

        **1. TF-IDF thiên về tần suất từ**
        - TF-IDF chỉ dựa vào tần suất và độ hiếm của từ trong văn bản.
        - Nếu hai văn bản có nhiều từ giống nhau → điểm similarity sẽ cao.
        - Trong tập dữ liệu, có thể nhiều mô tả có từ giống nhau → TF-IDF dễ cho điểm cao.

        **2. Gensim hiểu ngữ nghĩa**
        - Gensim (Doc2Vec hoặc Word2Vec) dùng embedding để biểu diễn văn bản.
        - Hiểu được ngữ cảnh, cấu trúc câu, và mối quan hệ giữa từ.
        - Vì embedding là không gian liên tục → điểm cosine similarity thường nhỏ hơn.
        - Gensim cho điểm thấp hơn, nhưng **phân biệt tốt hơn** giữa các văn bản thực sự giống nhau về ý nghĩa.
        """
        )
        st.image(
            "compare_cosine_gensim.jpg",
            caption="So sánh Cosine Similarity và Gensim",
            use_column_width=True,
        )

        st.markdown(
            """
    #### ✅ Kết luận
    - Với đặc điểm dữ liệu và kết quả đánh giá, mô hình **Content-based Filtering** sử dụng **Cosine Similarity** là lựa chọn phù hợp.
    - Phương pháp này tận dụng tốt thông tin mô tả sản phẩm và khả năng phân biệt ngữ nghĩa của Gensim.
    - Dù điểm similarity thấp hơn TF-IDF, nhưng độ chính xác trong việc gợi ý sản phẩm phù hợp lại cao hơn.
    - Do đó, nên ưu tiên sử dụng **Cosine Similarity với embedding từ Gensim** để cải thiện chất lượng gợi ý.
    """
        )


elif menu == "Recommendation":
    st.header("💡 Recommendation")
    st.write("Gợi ý khách sạn theo mô tả, điểm đánh giá, vị trí...")

    query = st.text_input("Nhập mô tả khách sạn bạn muốn tìm:")
    if query:
        results = recommend_hotels_by_description_sklearn(query)
        st.write(results)

elif menu == "Hotel Insight by Hotel ID":
    st.header("🔍 Hotel Insight by Hotel ID")

    # Load dữ liệu khách sạn
    df_info = pd.read_csv("hotel_info.csv")

    # Đọc dữ liệu
    df_comments = pd.read_csv("hotel_comments.csv")

    # Chuẩn hóa df_info
    score_cols = [
        "Location",
        "Cleanliness",
        "Service",
        "Facilities",
        "Value_for_money",
        "Comfort_and_room_quality",
    ]

    for col in score_cols:
        df_info[col] = pd.to_numeric(
            df_info["Total_Score"].astype(str).str.replace(",", "."), errors="coerce"
        )

    # Chuẩn hóa df_comments
    df_comments["Score"] = pd.to_numeric(
        df_comments["Score"].astype(str).str.replace(",", "."), errors="coerce"
    )
    df_comments["Review_Date_Clean"] = df_comments["Review Date"].apply(extract_date)
    df_comments["Review_Date_Clean"] = pd.to_datetime(
        df_comments["Review_Date_Clean"], format="%d/%m/%Y", errors="coerce"
    )
    df_comments["Title"] = df_comments["Title"].fillna("")
    df_comments["Body"] = df_comments["Body"].fillna("")
    df_comments["Review_Text"] = df_comments["Title"] + " " + df_comments["Body"]
    df_comments["Review_Month"] = df_comments["Review_Date_Clean"].dt.to_period("M")

    hotel_id = st.text_input("Nhập Hotel_ID cần phân tích: ")

    if hotel_id:
        if hotel_id.strip() == "":
            st.info("🔎 Vui lòng nhập Hotel_ID để bắt đầu phân tích.")
        elif hotel_id not in df_info["Hotel_ID"].astype(str).values:
            st.error("❌ Hotel_ID không tồn tại.")
        else:
            show_hotel_info(hotel_id)
            analyze_strengths_and_weaknesses(hotel_id)
            analyze_customers(hotel_id)
            analyze_keywords(hotel_id)
            compare_to_system(hotel_id)
    else:
        st.info("🔎 Nhập Hotel_ID để bắt đầu phân tích.")

elif menu == "Thông tin nhóm":
    st.header("👥 Nhóm E thực hiện")
    st.markdown(
        """
    **Họ tên HV 1**: Nguyễn Trung Hưng  
    **Họ tên HV 2**: Nguyễn Vũ Bảo Trân  
    """
    )







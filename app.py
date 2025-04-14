import gc
import streamlit as st
import pandas as pd
import pickle
from gensim import corpora, models, similarities
from recommend_utils import find_similar_sparse
from surprise import BaselineOnly

# --- Load dữ liệu & mô hình BaselineOnly ---
@st.cache_data
def load_data_rating():
    import os
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "data", "Products_ThoiTrangNam_rating_raw.csv")
    df = pd.read_csv(data_path, sep="\t")
    gc.collect()  # Giải phóng bộ nhớ sau khi đọc dữ liệu
    return df

# --- Load dữ liệu & mô hình TF-IDF ---
@st.cache_data
def load_data_tfidf():
    import os
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "data", "df_clean_thoitrangnam_raw.csv")
    df = pd.read_csv(data_path)
    gc.collect()  # Giải phóng bộ nhớ sau khi đọc dữ liệu
    return df
# --- Page config ---
st.set_page_config(page_title="Hệ thống gợi ý sản phẩm", layout="wide")

# --- Sidebar ---
with st.sidebar:
    import os

    st.title("📊 Tổng quan hệ thống")

    # Lấy đường dẫn ảnh tương đối từ thư mục hiện tại
    base_path = os.path.dirname(__file__)
    image_path = os.path.join(base_path, "images", "logo-shoppe.png")

    # Hiển thị ảnh
    st.image(
        image_path,
        use_container_width=True,
        caption="Tổng quan hệ thống Shopee",
    )

    # Dàn nội dung chính ở đây nếu có...

    # Footer section: dùng box đẹp + nằm dưới cùng
    st.markdown(
        """
        <hr style='margin-top: 2rem; margin-bottom: 1rem;'>

        <div style='
            padding: 10px 15px;
            border-radius: 10px;
            color: white;
        '>
            <h5>👨‍💻 Tác giả</h5>
            <ul>
                <li>Nguyễn Quyết Giang Sơn</li>
                <li>Phùng Anh Thư</li>
            </ul>
            <h5>🎓 Người hướng dẫn</h5>
            <ul>
                <li>ThS. Khuất Thùy Phương</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
# --- Load mô hình TF-IDF ---
@st.cache_resource
def load_models_tfidf():
    import os
    base_path = os.path.dirname(__file__)
    with open(os.path.join(base_path, "models", "dictionary.pkl"), "rb") as f:
        dictionary = pickle.load(f)
    with open(os.path.join(base_path, "models", "tfidf_model.pkl"), "rb") as f:
        tfidf_model = pickle.load(f)
    with open(os.path.join(base_path, "models", "index_sim.pkl"), "rb") as f:
        index_sim = pickle.load(f)
    gc.collect()  # Giải phóng bộ nhớ sau khi tải mô hình
    return dictionary, tfidf_model, index_sim

# --- Load mô hình Collaborative Filtering ---
@st.cache_resource
def load_baseline_model():
    import os
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "models", "baseline_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    gc.collect()  # Giải phóng bộ nhớ sau khi tải mô hình
    return model

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Gợi ý theo nội dung (TF-IDF)",
        "👤 Gợi ý theo người dùng (Collaborative Filtering)",
        "📊 Data Insight",
    ]
)

# ===== TAB 1: TF-IDF (Gensim) =====
with tab1:
    st.header("🔍 Gợi ý sản phẩm tương tự - TFIDF (Gensim)")

    # Tải dữ liệu sản phẩm và đánh giá
    df = load_data_tfidf()  # Gồm các cột: product_id, product_name, sub_category, image, price, rating, link, description_clean
    reviews_df = load_data_rating()  # Gồm các cột: product_id, user_id, user, rating

    # Tính toán số lượng đánh giá và điểm trung bình cho mỗi sản phẩm
    review_stats = (
        reviews_df.groupby("product_id")
        .agg(review_count=("rating", "count"), avg_rating=("rating", "mean"))
        .reset_index()
    )

    # Nối dữ liệu sản phẩm với thống kê đánh giá
    df = df.merge(review_stats, on="product_id", how="left")
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    df["avg_rating"] = df["avg_rating"].fillna(df["rating"])

    # Tạo corpus từ cột mô tả đã làm sạch
    dictionary, tfidf_model, index_sim = load_models_tfidf()
    docs = df["description_clean"].astype(str).apply(str.split)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # Giải phóng bộ nhớ sau khi tạo corpus
    gc.collect()

    # Lựa chọn chế độ
    mode = st.radio(
        "Chọn phương thức gợi ý:", ["Chọn sản phẩm có sẵn", "Nhập mô tả sản phẩm"]
    )

    # Xử lý truy vấn
    query_bow = None
    if mode == "Chọn sản phẩm có sẵn":
        # Chọn danh mục con
        sub_categories = df["sub_category"].unique()
        selected_sub_category = st.selectbox("Chọn danh mục sản phẩm:", sub_categories)

        # Lọc sản phẩm theo danh mục con đã chọn
        filtered_df = df[df["sub_category"] == selected_sub_category]
        product_names = filtered_df["product_name"].tolist()
        product_choice = st.selectbox(
            "Chọn sản phẩm bạn quan tâm:", product_names, key="product_choice_1"
        )
        if product_choice:
            index_query = df[df["product_name"] == product_choice].index[0]
            query_bow = corpus[index_query]
    else:
        user_input = st.text_area("Nhập nội dung mô tả sản phẩm bạn quan tâm:")
        if user_input.strip():
            query_doc = user_input.lower().split()
            query_bow = dictionary.doc2bow(query_doc)

    # Giải phóng bộ nhớ sau khi xử lý truy vấn
    gc.collect()

    # Các tuỳ chọn hiển thị
    num_results = st.slider(
        "Số lượng sản phẩm gợi ý:", min_value=3, max_value=20, value=5
    )
    sort_option = st.selectbox(
        "Sắp xếp kết quả theo:",
        ["Mặc định", "Giá tăng dần", "Giá giảm dần", "Ngẫu nhiên", "Rating cao nhất"],
    )

    if st.button("🔎 Gợi ý sản phẩm tương tự", key="btn_similar_tfidf"):
        if query_bow:
            # Tính độ tương đồng
            sims = index_sim[tfidf_model[query_bow]]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            top_indices = [idx for idx, _ in sims[:num_results]]
            result_df = df.iloc[top_indices].copy()

            # Giải phóng bộ nhớ sau khi tính toán độ tương đồng
            gc.collect()

            # Sắp xếp nếu cần
            if sort_option == "Giá tăng dần":
                result_df = result_df.sort_values(by="price")
            elif sort_option == "Giá giảm dần":
                result_df = result_df.sort_values(by="price", ascending=False)
            elif sort_option == "Rating cao nhất":
                result_df = result_df.sort_values(by="avg_rating", ascending=False)
            elif sort_option == "Ngẫu nhiên":
                result_df = result_df.sample(frac=1)

            # Hiển thị kết quả
            st.success("✅ Gợi ý thành công!")
            st.subheader("📦 Sản phẩm tương tự:")

            for _, row in result_df.iterrows():
                with st.container():
                    cols = st.columns([1, 3])
                    with cols[0]:
                        st.image(row["image"], width=120)
                    with cols[1]:
                        st.markdown(f"### [{row['product_name']}]({row['link']})")
                        st.markdown(f"💸 **Giá:** {row['price']:,.0f} VND")
                        st.markdown(
                            f"⭐ **Rating:** {row['avg_rating']:.1f} ({row['review_count']} đánh giá)"
                        )
                        st.markdown(f"📂 **Danh mục:** {row['sub_category']}")
                        st.markdown("---")

            # Giải phóng bộ nhớ sau khi hiển thị kết quả
            gc.collect()
        else:
            st.warning("❗ Vui lòng nhập mô tả sản phẩm hợp lệ.")

# ===== TAB 2: Collaborative Filtering =====

with tab2:
    st.header("👤 Gợi ý theo người dùng - BaselineOnly (Surprise)")

    # Tải dữ liệu
    df_rating = load_data_rating()
    df_info = load_data_tfidf()

    # Tải mô hình BaselineOnly
    baseline_model = load_baseline_model()

    st.markdown("## 👤 Tuỳ chọn người dùng")
    selected_mode = st.radio("Chọn cách nhập user ID:", ["Top 100 user", "Tự nhập"])

    if selected_mode == "Top 100 user":
        top_users = df_rating["user_id"].value_counts().head(100).index.tolist()
        selected_user = st.selectbox("Chọn user ID:", top_users)
    else:
        selected_user = st.text_input("Nhập user ID:")

    if selected_user:
        # Lọc dữ liệu các sản phẩm chưa được người dùng đánh giá
        df_products = pd.DataFrame({"product_id": df_rating["product_id"].unique()})
        rated_products = df_rating[df_rating["user_id"] == selected_user]["product_id"].unique()
        df_unrated = df_products[~df_products["product_id"].isin(rated_products)].copy()

        # Dự đoán điểm cho các sản phẩm chưa đánh giá
        df_unrated["EstimateScore"] = df_unrated["product_id"].apply(
            lambda x: baseline_model.predict(selected_user, x).est
        )

        # Sắp xếp các sản phẩm theo điểm dự đoán
        df_recommend = df_unrated.sort_values(by="EstimateScore", ascending=False)
        df_merged = df_recommend.merge(df_info, on="product_id", how="left")

        # Giải phóng bộ nhớ của các DataFrame không còn sử dụng
        del df_unrated
        gc.collect()

        # Các tuỳ chọn hiển thị
        st.markdown("## ⚙️ Tuỳ chọn hiển thị")
        num_recommend = st.slider("Số lượng sản phẩm muốn hiển thị:", 5, 20, 10)

        sort_option = st.selectbox(
            "Sắp xếp theo:",
            [
                "Giá thấp đến cao",
                "Giá cao đến thấp",
                "Ngẫu nhiên",
                "Điểm dự đoán cao nhất",
            ],
        )

        columns_to_show = st.multiselect(
            "Chọn các cột muốn hiển thị:",
            [
                "product_name",
                "image",
                "price",
                "EstimateScore",
                "sub_category",
                "description_clean",
                "link",
            ],
            default=[
                "product_name",
                "image",
                "price",
                "EstimateScore",
                "description_clean",
            ],
        )

        if sort_option == "Giá thấp đến cao":
            df_display = df_merged.sort_values(by="price").head(num_recommend)
        elif sort_option == "Giá cao đến thấp":
            df_display = df_merged.sort_values(by="price", ascending=False).head(
                num_recommend
            )
        elif sort_option == "Ngẫu nhiên":
            df_display = df_merged.sample(num_recommend)
        else:
            df_display = df_merged.sort_values(
                by="EstimateScore", ascending=False
            ).head(num_recommend)

        # Hiển thị kết quả gợi ý
        st.subheader(
            f"📌 Top {num_recommend} sản phẩm gợi ý cho user `{selected_user}`:"
        )

        for idx, row in df_display.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])

                with col1:
                    if "image" in columns_to_show and pd.notna(row["image"]):
                        st.image(row["image"], width=120)

                with col2:
                    if "product_name" in columns_to_show:
                        st.markdown(
                            f"<h5 style='margin-bottom: 0.5rem; color:#ff6600;'>{row['product_name']}</h5>",
                            unsafe_allow_html=True,
                        )

                    price_str = (
                        f"{int(row['price']):,} VND"
                        if "price" in columns_to_show and pd.notna(row["price"])
                        else ""
                    )
                    score_str = (
                        f"{row['EstimateScore']:.2f}"
                        if "EstimateScore" in columns_to_show
                        else ""
                    )

                    if price_str or score_str:
                        st.markdown(
                            f"""
                            <div style='font-size:15px; margin-top:-10px;'>
                                <b>Giá:</b> {price_str} &nbsp;&nbsp;&nbsp;
                                <b>Điểm dự đoán:</b> {score_str}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if "sub_category" in columns_to_show and pd.notna(
                        row["sub_category"]
                    ):
                        st.markdown(
                            f"<b>Danh mục:</b> {row['sub_category']}",
                            unsafe_allow_html=True,
                        )

                    if "description_clean" in columns_to_show and pd.notna(
                        row["description_clean"]
                    ):
                        with st.expander("📄 Xem mô tả chi tiết", expanded=False):
                            st.markdown(
                                f"""
                                <div style='
                                    background-color: #f5f5f5;
                                    padding: 12px 15px;
                                    border-radius: 10px;
                                    font-size: 14px;
                                    line-height: 1.6;
                                    color: #333;
                                '>{row['description_clean']}</div>
                                """,
                                unsafe_allow_html=True,
                            )

                    if "link" in columns_to_show and pd.notna(row["link"]):
                        st.markdown(
                            f"[🔗 Xem sản phẩm trên Shopee]({row['link']})",
                            unsafe_allow_html=True,
                        )

        # Giải phóng bộ nhớ sau khi hiển thị kết quả
        del df_recommend, df_merged, df_display
        gc.collect()

        st.success("✅ Gợi ý thành công!")

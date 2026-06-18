import os
import numpy as np
import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

st.set_page_config(
    page_title="CineSense – Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a8b2d8;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Movie card */
    .movie-card {
        background: #1e2235;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #e94560;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: background 0.2s;
        position: relative;
    }
    .movie-card:hover { background: #252a40; }

    /* Tooltip */
    .movie-tooltip {
        display: none;
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        background: #0d1117;
        border: 1px solid #30364a;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        z-index: 9999;
        min-width: 200px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        pointer-events: none;
        white-space: nowrap;
    }
    .movie-card:hover .movie-tooltip { display: block; }
    .tooltip-row {
        display: flex;
        justify-content: space-between;
        gap: 1.5rem;
        align-items: center;
    }
    .tooltip-label {
        color: #6272a4;
        font-size: 0.75rem;
    }
    .tooltip-value {
        color: #f8f8f2;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .tooltip-divider {
        border: none;
        border-top: 1px solid #2a3050;
        margin: 0.4rem 0;
    }
    .movie-rank {
        color: #4a5580;
        font-size: 0.8rem;
        font-weight: 600;
        min-width: 30px;
    }
    .movie-info {
        flex: 1;
        padding: 0 0.75rem;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    .movie-title {
        color: #cdd6f4;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .movie-genres {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }
    .genre-tag {
        background: #2a3050;
        color: #7c8fc7;
        padding: 0.1rem 0.45rem;
        border-radius: 10px;
        font-size: 0.68rem;
        font-weight: 500;
        white-space: nowrap;
    }
    .movie-score {
        background: linear-gradient(135deg, #e94560, #c62a47);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        min-width: 45px;
        text-align: center;
        white-space: nowrap;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1a2e;
        padding: 0.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a8b2d8;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: #e94560 !important;
        color: white !important;
    }

    /* Sidebar */
    .stSidebar { background: #1a1a2e; }

    /* Info badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .badge-blue  { background: #1e3a5f; color: #60a5fa; }
    .badge-green { background: #1a3a2a; color: #4ade80; }
    .badge-red   { background: #3a1a2a; color: #f87171; }
</style>
""", unsafe_allow_html=True)


DATA_DIR       = "."
RANDOM_STATE   = 42
TOP_SVD        = 200
K_WPR          = 40
MIN_COMMON     = 3
MIN_RATING_PROFILE = 4.0

@st.cache_resource
def load_data():
    rating_cols = ["userId", "movieId", "rating", "timestamp"]
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "u.data"),
        sep="\t", names=rating_cols, engine="python",
    )
    ratings["rating"] = ratings["rating"].astype(float)

    item_cols = [
        "movieId", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    items = pd.read_csv(
        os.path.join(DATA_DIR, "u.item"),
        sep="|", names=item_cols, encoding="latin-1", engine="python",
    )
    genre_cols = item_cols[5:]
    items[genre_cols] = items[genre_cols].fillna(0).astype(int)

    def build_movie_text(row):
        genres = [g for g in genre_cols if int(row[g]) == 1]
        return f"{row['title']} {' '.join(genres)}"

    items["movie_text"] = items.apply(build_movie_text, axis=1)

    tfidf = TfidfVectorizer(max_features=300, min_df=2, stop_words="english")
    X_movies = tfidf.fit_transform(items["movie_text"])
    all_movie_ids = items["movieId"].astype(int).tolist()
    mid_to_row = {int(m): i for i, m in enumerate(all_movie_ids)}

    # Thể loại
    items["genres"] = items.apply(
        lambda row: " | ".join([g for g in genre_cols if int(row[g]) == 1]) or "N/A",
        axis=1,
    )

    # Thống kê đánh giá thực tế cho mỗi phim
    movie_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", num_ratings="count")
        .reset_index()
    )
    movie_stats["avg_rating"] = movie_stats["avg_rating"].round(2)
    items = items.merge(movie_stats, on="movieId", how="left")
    items["avg_rating"]  = items["avg_rating"].fillna(0.0)
    items["num_ratings"] = items["num_ratings"].fillna(0).astype(int)

    return ratings, items, X_movies, all_movie_ids, mid_to_row



def ratings_dict(df):
    d = {}
    for _, row in df.iterrows():
        u, i, r = int(row["userId"]), int(row["movieId"]), float(row["rating"])
        d.setdefault(u, {})[i] = r
    return d


def user_means(rdict):
    return {u: float(np.mean(list(v.values()))) for u, v in rdict.items()}


def pearson_sim(rdict, mean_u, u, v, min_common=3):
    if u == v:
        return 0.0
    iu, iv = rdict.get(u, {}), rdict.get(v, {})
    common = set(iu) & set(iv)
    if len(common) < min_common:
        return 0.0
    num = sum((iu[i] - mean_u[u]) * (iv[i] - mean_u[v]) for i in common)
    du  = sum((iu[i] - mean_u[u]) ** 2 for i in common)
    dv  = sum((iv[i] - mean_u[v]) ** 2 for i in common)
    den = np.sqrt(du * dv)
    return 0.0 if den < 1e-12 else num / den


def precompute_user_sim(rdict, users, min_common=3):
    mean_u = user_means(rdict)
    uid_to_idx = {u: k for k, u in enumerate(users)}
    n = len(users)
    sim = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            s = pearson_sim(rdict, mean_u, users[a], users[b], min_common)
            sim[a, b] = sim[b, a] = s
    return sim, uid_to_idx, mean_u


def wpr_predict(u, i, rdict, mean_u, sim_row, uid_to_idx, users_list, K=40, gmean=3.5):
    candidates = []
    for v in rdict:
        if v == u or i not in rdict[v]:
            continue
        j = uid_to_idx.get(v)
        if j is None:
            continue
        s = sim_row[j]
        if s <= 0:
            continue
        rv = rdict[v][i]
        candidates.append((s, rv, mean_u[v]))
    if not candidates:
        return gmean if u not in mean_u else mean_u[u]
    candidates.sort(key=lambda x: -x[0])
    top = candidates[:K]
    num = sum(s * (rv - mb) for s, rv, mb in top)
    den = sum(abs(s) for s, _, _ in top)
    if den < 1e-12:
        return mean_u.get(u, gmean)
    base = mean_u.get(u, gmean)
    return float(np.clip(base + num / den, 1.0, 5.0))


def user_profile_vector(raw_uid, train_df, X_movies, mid_to_row, min_r=MIN_RATING_PROFILE):
    subs = train_df[train_df["userId"] == raw_uid]
    hi = subs[subs["rating"] >= min_r]
    if hi.empty:
        hi = subs.nlargest(8, "rating")
    rows = [mid_to_row[int(mid)] for mid in hi["movieId"] if int(mid) in mid_to_row]
    if not rows:
        return None
    v = np.asarray(X_movies[rows].mean(axis=0)).ravel()
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-12 else None


def taste_cosines(candidate_mids, profile, X_movies, mid_to_row):
    if profile is None:
        return {m: 0.0 for m in candidate_mids}
    out = {}
    for m in candidate_mids:
        r = mid_to_row.get(m)
        if r is None:
            out[m] = 0.0
            continue
        mv = np.asarray(X_movies[r].toarray()).ravel()
        nm = np.linalg.norm(mv)
        out[m] = float(np.dot(profile, mv / nm)) if nm > 1e-12 else 0.0
    return out



@st.cache_resource
def train_models():
    ratings, items, X_movies, all_movie_ids, mid_to_row = load_data()

    # 1. SVD
    reader = Reader(rating_scale=(1, 5))
    train_surprise = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = train_surprise.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=25, lr_all=0.005, reg_all=0.02, random_state=RANDOM_STATE)
    svd.fit(trainset)

    # 2. WPR components
    rdict = ratings_dict(ratings)
    gmean = float(ratings["rating"].mean())
    users_sorted = sorted(rdict.keys())
    sim_mat, uid_to_idx, mean_u = precompute_user_sim(rdict, users_sorted, MIN_COMMON)

    def wpr_score(u, movie_id):
        if u not in uid_to_idx:
            return gmean
        sim_row = sim_mat[uid_to_idx[u]]
        return wpr_predict(u, movie_id, rdict, mean_u, sim_row, uid_to_idx, users_sorted, K=K_WPR, gmean=gmean)

    # 3. Blend model (Hybrid)
    rng = np.random.RandomState(RANDOM_STATE)
    users = ratings["userId"].drop_duplicates().astype(int).values
    picked = rng.choice(users, size=min(250, len(users)), replace=False)
    X_feat, y_feat = [], []
    for u in picked:
        u_df = ratings[ratings["userId"] == int(u)][["movieId", "rating"]]
        if u_df.empty:
            continue
        if len(u_df) > 20:
            u_df = u_df.sample(n=20, random_state=RANDOM_STATE)
        prof = user_profile_vector(int(u), ratings, X_movies, mid_to_row)
        for _, rr in u_df.iterrows():
            m = int(rr["movieId"])
            y_true = float(rr["rating"])
            s_svd = float(svd.predict(int(u), m).est)
            s_wpr = float(wpr_score(int(u), m))
            r = mid_to_row.get(m)
            if prof is None or r is None:
                s_tfidf = 0.0
            else:
                mv = np.asarray(X_movies[r].toarray()).ravel()
                nm = np.linalg.norm(mv)
                s_tfidf = float(np.dot(prof, mv / nm)) if nm > 1e-12 else 0.0
            X_feat.append([s_svd, s_tfidf, s_wpr])
            y_feat.append(y_true)

    blend_model = Ridge(alpha=1.0)
    if X_feat:
        blend_model.fit(np.array(X_feat, dtype=np.float64), np.array(y_feat, dtype=np.float64))
    else:
        blend_model.coef_ = np.array([1.0, 0.0, 0.0])
        blend_model.intercept_ = 0.0

    return svd, blend_model, rdict, gmean, users_sorted, sim_mat, uid_to_idx, mean_u, wpr_score



def recommend_svd(user_id, seen_movies, svd, all_movie_ids, items, top_n=50):
    scored = []
    for m in all_movie_ids:
        if m in seen_movies:
            continue
        est = svd.predict(user_id, m).est
        scored.append((m, round(float(est), 2)))
    scored.sort(key=lambda x: -x[1])
    top = scored[:top_n]
    result = []
    for rank, (mid, score) in enumerate(top, 1):
        r = items[items["movieId"] == mid]
        title      = r["title"].values[0]      if not r.empty else f"Movie {mid}"
        genres     = r["genres"].values[0]     if not r.empty else "N/A"
        avg_rating = r["avg_rating"].values[0] if not r.empty else 0.0
        num_rat    = r["num_ratings"].values[0] if not r.empty else 0
        result.append({"Hạng": rank, "Tên phim": title, "Thể loại": genres,
                       "Điểm dự đoán": score, "avg_rating": avg_rating, "num_ratings": num_rat})
    return pd.DataFrame(result)


def recommend_tfidf(user_id, seen_movies, ratings, X_movies, mid_to_row, all_movie_ids, items, top_n=50):
    prof = user_profile_vector(user_id, ratings, X_movies, mid_to_row)
    if prof is None:
        return pd.DataFrame({"Hạng": [], "Tên phim": [], "Thể loại": [], "Điểm dự đoán": [], "avg_rating": [], "num_ratings": []})
    scored = []
    for m in all_movie_ids:
        if m in seen_movies:
            continue
        r = mid_to_row.get(m)
        if r is None:
            continue
        mv = np.asarray(X_movies[r].toarray()).ravel()
        nm = np.linalg.norm(mv)
        cos = float(np.dot(prof, mv / nm)) if nm > 1e-12 else 0.0
        # Scale cosine [0,1] to [1,5] for unified display
        score = round(1.0 + cos * 4.0, 2)
        scored.append((m, score))
    scored.sort(key=lambda x: -x[1])
    top = scored[:top_n]
    result = []
    for rank, (mid, score) in enumerate(top, 1):
        r = items[items["movieId"] == mid]
        title      = r["title"].values[0]      if not r.empty else f"Movie {mid}"
        genres     = r["genres"].values[0]     if not r.empty else "N/A"
        avg_rating = r["avg_rating"].values[0] if not r.empty else 0.0
        num_rat    = r["num_ratings"].values[0] if not r.empty else 0
        result.append({"Hạng": rank, "Tên phim": title, "Thể loại": genres,
                       "Điểm dự đoán": score, "avg_rating": avg_rating, "num_ratings": num_rat})
    return pd.DataFrame(result)


def recommend_hybrid(user_id, seen_movies, svd, blend_model, wpr_score,
                     ratings, X_movies, mid_to_row, all_movie_ids, items, top_n=50):
    # Candidate pool
    svd_scored = []
    for m in all_movie_ids:
        if m in seen_movies:
            continue
        svd_scored.append((m, float(svd.predict(user_id, m).est)))
    svd_scored.sort(key=lambda x: -x[1])
    candidates = [m for m, _ in svd_scored[:TOP_SVD]]
    svd_scores = {m: s for m, s in svd_scored[:TOP_SVD]}

    prof = user_profile_vector(user_id, ratings, X_movies, mid_to_row)
    cos_map = taste_cosines(candidates, prof, X_movies, mid_to_row)

    feats, mids = [], []
    for m in candidates:
        s_svd   = svd_scores.get(m, 0.0)
        s_tfidf = cos_map.get(m, 0.0)
        s_wpr   = float(wpr_score(user_id, m))
        feats.append([s_svd, s_tfidf, s_wpr])
        mids.append(m)

    preds = blend_model.predict(np.array(feats, dtype=np.float64))
    preds = np.clip(preds, 1.0, 5.0)
    scored = sorted(zip(mids, preds.tolist()), key=lambda x: -x[1])[:top_n]

    result = []
    for rank, (mid, score) in enumerate(scored, 1):
        r = items[items["movieId"] == mid]
        title      = r["title"].values[0]      if not r.empty else f"Movie {mid}"
        genres     = r["genres"].values[0]     if not r.empty else "N/A"
        avg_rating = r["avg_rating"].values[0] if not r.empty else 0.0
        num_rat    = r["num_ratings"].values[0] if not r.empty else 0
        result.append({"Hạng": rank, "Tên phim": title, "Thể loại": genres,
                       "Điểm dự đoán": round(score, 2), "avg_rating": avg_rating, "num_ratings": num_rat})
    return pd.DataFrame(result)



def render_movie_list(df):
    if df.empty:
        st.warning("Không có đề xuất nào.")
        return
    for _, row in df.iterrows():
        # Tạo HTML cho từng genre tag
        genre_tags = ""
        genres = str(row.get("Thể loại", "N/A"))
        if genres and genres != "N/A":
            for g in genres.split(" | "):
                genre_tags += f'<span class="genre-tag">{g}</span>'
        else:
            genre_tags = '<span class="genre-tag">N/A</span>'

        avg_r   = row.get("avg_rating", 0.0)
        num_r   = row.get("num_ratings", 0)
        stars   = "★" * round(avg_r) + "☆" * (5 - round(avg_r)) if avg_r > 0 else "—"

        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-tooltip">
                <div class="tooltip-row">
                    <span class="tooltip-label">⭐ Rating TB cộng đồng</span>
                    <span class="tooltip-value">{avg_r}/5</span>
                </div>
                <hr class="tooltip-divider">
                <div class="tooltip-row">
                    <span class="tooltip-label">📊 Số lượt đánh giá</span>
                    <span class="tooltip-value">{num_r:,} votes</span>
                </div>
            </div>
            <span class="movie-rank">#{int(row['Hạng'])}</span>
            <div class="movie-info">
                <span class="movie-title">{row['Tên phim']}</span>
                <div class="movie-genres">{genre_tags}</div>
            </div>
            <span class="movie-score">⭐ {row['Điểm dự đoán']}</span>
        </div>
        """, unsafe_allow_html=True)



def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 CineSense</h1>
        <p>Hệ thống gợi ý phim thông minh – So sánh 3 phương pháp</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load dữ liệu & Train mô hình ──
    with st.spinner("⏳ Đang tải dữ liệu và huấn luyện mô hình... (chỉ lần đầu)"):
        ratings, items, X_movies, all_movie_ids, mid_to_row = load_data()
        svd, blend_model, rdict, gmean, users_sorted, sim_mat, uid_to_idx, mean_u, wpr_score = train_models()

    st.success("✅ Mô hình đã sẵn sàng!")

    # ── Sidebar – Nhập thông tin ──
    with st.sidebar:
        st.markdown("## ⚙️ Cài đặt")
        st.markdown("---")

        all_users = sorted(ratings["userId"].unique().tolist())
        user_id = st.number_input(
            "🔢 Nhập User ID",
            min_value=int(min(all_users)),
            max_value=int(max(all_users)),
            value=1,
            step=1,
        )
        top_n = st.slider("📋 Số phim tối đa mỗi mô hình", min_value=10, max_value=50, value=50, step=5)
        run_btn = st.button("🚀 Lấy đề xuất", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("### 📊 Mô hình")
        st.markdown("""
        - **SVD** – Lọc cộng tác dựa trên hành vi cộng đồng  
        - **TF-IDF** – Dựa trên nội dung thể loại phim  
        - **Hybrid** – Kết hợp tối ưu cả hai phương pháp  
        """)

    # ── Hiển thị lịch sử xem ──
    seen_movies = set(ratings[ratings["userId"] == user_id]["movieId"].astype(int))
    user_history = ratings[ratings["userId"] == user_id].merge(
        items[["movieId", "title", "genres"]], on="movieId"
    ).sort_values("rating", ascending=False)

    with st.expander(f"📽️ Phim đã xem của User {user_id} ({len(seen_movies)} phim)", expanded=False):
        st.dataframe(
            user_history[["title", "genres", "rating"]].rename(
                columns={"title": "Tên phim", "genres": "Thể loại", "rating": "Điểm đã chấm"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    if run_btn:
        st.markdown(f"### 🎯 Đề xuất phim cho **User {user_id}** (Top {top_n})")

        tab_svd, tab_tfidf, tab_hybrid = st.tabs([
            "🤖 SVD – Lọc cộng tác",
            "📄 TF-IDF – Nội dung",
            "⚡ Hybrid – Kết hợp tối ưu",
        ])

        with tab_svd:
            st.markdown('<div class="badge badge-blue">Collaborative Filtering</div>', unsafe_allow_html=True)
            st.caption("Gợi ý dựa trên hành vi của những người dùng có gu giống bạn.")
            with st.spinner("Đang tính toán..."):
                df_svd = recommend_svd(user_id, seen_movies, svd, all_movie_ids, items, top_n=top_n)
            render_movie_list(df_svd)

        with tab_tfidf:
            st.markdown('<div class="badge badge-green">Content-based Filtering</div>', unsafe_allow_html=True)
            st.caption("Gợi ý những phim có nội dung và thể loại gần với sở thích của bạn.")
            with st.spinner("Đang tính toán..."):
                df_tfidf = recommend_tfidf(user_id, seen_movies, ratings, X_movies, mid_to_row, all_movie_ids, items, top_n=top_n)
            render_movie_list(df_tfidf)

        with tab_hybrid:
            st.markdown('<div class="badge badge-red">Hybrid Model (Linear Learned)</div>', unsafe_allow_html=True)
            st.caption("Kết hợp SVD + TF-IDF + WPR thông qua mô hình Ridge Regression được huấn luyện.")
            with st.spinner("Đang tính toán..."):
                df_hybrid = recommend_hybrid(
                    user_id, seen_movies, svd, blend_model, wpr_score,
                    ratings, X_movies, mid_to_row, all_movie_ids, items, top_n=top_n
                )
            render_movie_list(df_hybrid)
    else:
        st.info("👈 Nhập User ID ở thanh bên và nhấn **Lấy đề xuất** để bắt đầu!")


if __name__ == "__main__":
    main()

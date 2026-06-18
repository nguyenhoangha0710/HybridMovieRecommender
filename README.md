# 🎬 CineSense – Movie Recommendation System

CineSense là một hệ thống gợi ý phim thông minh được thiết kế nhằm tối ưu hóa sự cân bằng giữa **độ chính xác dự đoán (Accuracy)** và **tính đa dạng (Diversity)** của danh sách đề xuất. Dự án kết hợp các phương pháp Lọc cộng tác (Collaborative Filtering), Lọc dựa trên nội dung (Content-based Filtering) và thuật toán láng giềng để xây dựng mô hình Hybrid tối ưu.

---

## 📌 Giới thiệu dự án

Trong các hệ thống gợi ý hiện đại, việc chỉ tập trung vào độ chính xác (ví dụ: đề xuất các phim có điểm đánh giá dự đoán cao nhất) thường dẫn đến hiện tượng **Popularity Bias** (chỉ gợi ý các phim quá phổ biến) và làm giảm trải nghiệm khám phá của người dùng. 

Hệ thống **CineSense** được xây dựng nhằm giải quyết bài toán này bằng cách:
1. Dự đoán hành vi dựa trên cộng đồng (SVD).
2. Phân tích nội dung và thể loại phim mà người dùng yêu thích (TF-IDF).
3. Đánh giá độ tương đồng giữa các người dùng thông qua hệ số tương quan Pearson (WPR).
4. **Học máy lai (Hybrid Model)**: Sử dụng mô hình hồi quy Ridge Regression để tìm ra tỷ lệ kết hợp tối ưu giữa các tín hiệu trên, giúp tăng hiệu năng gợi ý và cải thiện tính đa dạng của danh sách đề xuất.

Dữ liệu thử nghiệm được sử dụng là tập dữ liệu chuẩn **MovieLens 100K** gồm 100,000 lượt đánh giá từ 943 người dùng cho 1,682 bộ phim.

---

## ⚙️ Các mô hình trong hệ thống

Dự án triển khai và so sánh 4 mô hình gợi ý khác nhau:

1. **SVD Only (Collaborative Filtering)**: 
   * Sử dụng thư viện `surprise` để phân rã ma trận tương tác User-Item thành các nhân tử ẩn (Latent Factors).
   * Cấu hình: $N_{factors}=50$, $N_{epochs}=25$, học suất $\eta = 0.005$, hệ số phạt L2 $\lambda = 0.02$.
   * Ưu điểm: Nắm bắt tốt các mối quan hệ ẩn dựa trên hành vi chấm điểm của cộng đồng.

2. **TF-IDF Only (Content-based Filtering)**:
   * Trích xuất đặc trưng của phim từ văn bản kết hợp tên phim và thể loại (`movie_text`) bằng bộ vector hóa TF-IDF với tối đa 300 đặc trưng.
   * Xây dựng hồ sơ sở thích (User Profile Vector) bằng cách lấy trung bình cộng vector TF-IDF của những bộ phim người dùng đã đánh giá cao ($\ge 4.0$).
   * Gợi ý dựa trên độ tương đồng Cosine giữa User Profile và danh sách phim chưa xem.

3. **SVD + WPR (Weighted Pearson Rating)**:
   * Tính toán độ tương đồng giữa các người dùng thông qua hệ số tương quan Pearson trên các phim họ đồng đánh giá (ngưỡng tối thiểu đồng đánh giá = 3).
   * Rerank (xếp hạng lại) các ứng viên tiềm năng từ SVD bằng điểm số dự đoán từ thuật toán WPR.

4. **Linear Learned Hybrid (Mô hình lai tối ưu)**:
   * Sử dụng thuật toán hồi quy Ridge Regression ($L_2$ Regularization) để kết hợp tuyến tính 3 nguồn điểm dự đoán: $S_{SVD}$, $S_{TFIDF}$, và $S_{WPR}$.
   * Mô hình được huấn luyện trên mẫu hành vi thực tế của người dùng để tìm ra trọng số tối ưu giúp tối đa hóa độ chính xác xếp hạng thực tế.

---

## 🔄 Quá trình thực hiện & Đánh giá (Process)

Để đánh giá một cách khách quan và toàn diện, dự án áp dụng phương pháp kiểm thử chéo **5-Fold Cross Validation** trên hai không gian xếp hạng: **Global Ranking** (trên toàn bộ catalog phim chưa xem) và **Restricted Ranking** (trên tập phim xuất hiện trong tập Test).

Hệ thống sử dụng hai nhóm chỉ số đánh giá cốt lõi:

### 1. Nhóm chỉ số chính xác (Accuracy Metrics)
* **NDCG@k (Normalized Discounted Cumulative Gain)**: Đánh giá chất lượng sắp xếp thứ tự của các đề xuất (phim hay hơn nên được xếp ở vị trí cao hơn).
* **Recall@k**: Đo lường tỷ lệ các bộ phim thực sự được người dùng yêu thích (rating tập test $\ge 4$) được hệ thống tìm thấy trong top-k.
* **MAP@k (Mean Average Precision)**: Đo lường độ chính xác trung bình trên toàn bộ danh sách kết quả.

### 2. Nhóm chỉ số đa dạng & Khám phá (Diversity & Exploration Metrics)
* **Coverage@k**: Tỷ lệ bao phủ catalog phim (tổng số phim độc nhất được gợi ý cho toàn bộ người dùng chia cho tổng số phim trong hệ thống).
* **Novelty@k**: Độ mới lạ của danh sách gợi ý, tính bằng lượng tin tự thân (Self-Information) trung bình dựa trên xác suất Laplace của số lượt đánh giá phim trong tập huấn luyện ($-\log_2 p$). Chỉ số cao chứng tỏ mô hình gợi ý được nhiều bộ phim chất lượng nằm ở phần đuôi dài (Long-Tail), tránh tập trung quá mức vào các phim hot/bom tấn.
* **Intra-List Diversity (ILD)@k**: Độ đa dạng nội tại của danh sách đề xuất đối với từng người dùng cá nhân, được tính bằng trung bình khoảng cách Cosine ($1 - \text{Cosine Similarity}$) giữa các vector đặc trưng TF-IDF của các cặp phim nằm trong danh sách top-k.

---

## 📊 Kết quả đạt được (Results)

Kết quả trung bình thu được sau quá trình kiểm thử chéo 5-Fold Cross Validation thể hiện rõ rệt sức mạnh của mô hình Hybrid:

### 1. Kết quả độ chính xác (Restricted Ranking @ k=10)

| Mô hình | NDCG@10 (Mean ± Std) | Recall@10 (Mean ± Std) | MAP@10 (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| **TF-IDF Only** | 0.6852 ± 0.0172 | 0.6597 ± 0.0332 | 0.4691 ± 0.0242 |
| **SVD Only** | 0.8177 ± 0.0169 | 0.7254 ± 0.0300 | 0.5965 ± 0.0282 |
| **SVD + WPR** | 0.8227 ± 0.0143 | 0.7267 ± 0.0262 | 0.6034 ± 0.0224 |
| **Linear Learned (Hybrid)** | **0.8247 ± 0.0162** | **0.7276 ± 0.0288** | **0.6085 ± 0.0288** |

> **Nhận xét**: Mô hình lai tối ưu **Linear Learned** đạt điểm số vượt trội nhất ở tất cả các chỉ số độ chính xác (NDCG, Recall và MAP), chứng tỏ việc kết hợp tín hiệu nội dung và tương đồng láng giềng vào mô hình cộng tác SVD giúp nâng cao độ chính xác đáng kể.

### 2. Kết quả tính đa dạng (Global Ranking @ k=100)

* **Coverage (Khả năng bao phủ)**: 
  * **TF-IDF Only** đạt độ bao phủ lớn nhất (**45.5%**) do đặc thù phân tán theo từ khóa thể loại phim.
  * **SVD Only** và **SVD+WPR** có xu hướng tập trung vào các phim phổ biến nên độ bao phủ thấp hơn (lần lượt là **26.3%** và **23.4%**).
  * **Linear Learned** đã cải thiện độ bao phủ của SVD đáng kể lên mức **29.1%** nhờ có thêm tín hiệu từ mô hình nội dung.
* **Novelty (Độ mới lạ - Khám phá)**:
  * Điểm số Novelty của **Linear Learned** đạt **10.25**, cải thiện rõ rệt so với **SVD Only** (**9.94**).
* **ILD (Độ đa dạng thể loại trong danh sách cá nhân)**:
  * **Linear Learned** duy trì được chỉ số ILD cực kỳ cao ở mức **0.914**, vượt trội so với việc chỉ dùng đơn thuần **TF-IDF Only** (**0.545**).

---

## 💻 Ứng dụng Web Demo (Streamlit App)

Dự án cung cấp một giao diện web trực quan được viết bằng **Streamlit** (`app.py`) cho phép:
* Nhập **User ID** bất kỳ để xem lịch sử những phim người dùng đã chấm điểm kèm theo thể loại.
* So sánh danh sách đề xuất trực tiếp giữa 3 tab đại diện cho 3 trường phái thuật toán khác nhau: **SVD (Lọc cộng tác)**, **TF-IDF (Dựa trên nội dung)** và **Hybrid (Mô hình lai tối ưu)**.
* Giao diện phong cách tối hiện đại (Glassmorphism), có tích hợp tooltips hiển thị điểm đánh giá trung bình của cộng đồng và số lượt bình chọn thực tế khi người dùng di chuột qua thẻ phim.

### Hướng dẫn chạy ứng dụng Streamlit:
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install numpy pandas streamlit scikit-surprise scikit-learn matplotlib
   ```
2. Khởi chạy ứng dụng Streamlit từ thư mục dự án:
   ```bash
   streamlit run app.py
   ```

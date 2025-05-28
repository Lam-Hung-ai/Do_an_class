## Môn đồ án trí tuệ nhân tạo
- Họ và tên: Nguyễn Văn Lâm Hùng
- MSV: 2351260653
- Lớp học phần: 64TTNT2
  
### Thực hành 1: Improved Prediction Accuracy of House Price Using Decision Tree Algorithm over Linear Regression Algorithm

**a) Ý chính của Nghiên cứu:**

Bài báo này tập trung vào việc nâng cao độ chính xác trong dự đoán giá nhà bằng cách đề xuất và đánh giá một thuật toán Cây quyết định (Decision Tree - DT) mới, so sánh hiệu quả của nó với thuật toán Hồi quy Tuyến tính (Linear Regression - LR) truyền thống. Mục tiêu cốt lõi là chứng minh rằng thuật toán Cây quyết định có thể mang lại kết quả dự đoán tốt hơn trong lĩnh vực bất động sản.

Kết quả chính cho thấy thuật toán Cây quyết định mới đạt độ chính xác 90% trong việc dự đoán giá nhà, trong khi thuật toán Hồi quy Tuyến tính đạt 80%. Mặc dù có sự chênh lệch về độ chính xác, phân tích thống kê sử dụng SPSS cho thấy sự khác biệt này không có ý nghĩa thống kê (p=0.618, với p > 0.05). Tuy nhiên, các tác giả kết luận rằng thuật toán Cây quyết định đổi mới vượt trội hơn phương pháp Hồi quy Tuyến tính trong việc ước tính giá trị bất động sản.

**b) Phương thức Thực hiện:**
1. Thiết kế Nghiên cứu và Mẫu:
   1. Nghiên cứu được thực hiện tại Phòng thí nghiệm DBMS, Khoa Khoa học Máy tính và Kỹ thuật, Trường Kỹ thuật Saveetha.
   1. Kích thước mẫu được xác định bằng G power Calculator, với mục tiêu đạt 80% năng lực phân tích.
   1. Tổng cộng có 20 điểm dữ liệu được sử dụng cho mỗi thuật toán (N=20 cho mỗi nhóm trong phân tích thống kê).
   1. Dữ liệu được chia thành hai nhóm: Nhóm 1 sử dụng phương pháp Hồi quy Tuyến tính và Nhóm 2 sử dụng thuật toán Cây quyết định mới.
   1. Các thông số thống kê bao gồm: mức alpha (α) = 0.05, mức beta (β) = 0.2, và khoảng tin cậy 95%.
1. Thu thập và Tiền xử lý Dữ liệu:
   1. Bộ dữ liệu được lấy từ Kaggle, có tên "House Prices - Advanced Regression Techniques", dưới dạng tệp CSV.
   1. Bộ dữ liệu chứa khoảng 80 thông số (thuộc tính) khác nhau có ảnh hưởng đến giá nhà.
   1. Dữ liệu được chia thành 80% cho tập huấn luyện (train data) và 20% cho tập kiểm tra (test data), với random\_state được đặt là 42 để đảm bảo tính ngẫu nhiên có thể tái tạo.
   1. Các thuộc tính được xác định là có ảnh hưởng lớn nhất đến dự đoán giá nhà bao gồm: SalePrice, OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt.
1. Mô hình Thuật toán:
   1. Thuật toán Cây Quyết định Mới (Novel Decision Tree Algorithm):
      1. Là một phương pháp học có giám sát, không tham số, được sử dụng cho cả bài toán phân loại và hồi quy.
      1. Hoạt động dựa trên nguyên tắc chia nhị phân đệ quy (recursive binary splitting).
      1. Các nhánh được chọn dựa trên hàm chi phí (cost function), cụ thể là tối thiểu hóa chỉ số Gini.
      1. Các đặc trưng không liên quan được tỉa bớt (pruning) để cải thiện hiệu suất.
   1. Thuật toán Hồi quy Tuyến tính (Linear Regression):
      1. Dựa trên học có giám sát, mô hình hóa mối quan hệ tuyến tính giữa biến phụ thuộc (giá nhà) và các biến độc lập (các thuộc tính của nhà).
      1. Mô hình đơn giản có dạng: y = A0 + A1\*x.
      1. Hàm chi phí được sử dụng để đánh giá là RMSE (Root Mean Squared Error).


### Thực hành 2: HeartInspect Heart Disease Prediction of an Individual Using Nave Bayes Algorithm

**a) Ý chính của Nghiên cứu:**

Bài báo này trình bày về "HeartInspect", một hệ thống dự đoán bệnh tim dựa trên thuật toán Naïve Bayes. Mục tiêu chính là giải quyết các thách thức trong việc dự đoán sớm bệnh tim, đặc biệt ở những khu vực hạn chế về nguồn lực y tế và công cụ chẩn đoán. Nghiên cứu nhấn mạnh tầm quan trọng của việc phát hiện sớm để can thiệp kịp thời và cải thiện kết quả điều trị.

Hệ thống sử dụng một bộ dữ liệu bao gồm các thông tin như tình trạng bệnh tim, chiều cao, cân nặng, sức khỏe thể chất, khó khăn khi đi lại, nhóm tuổi, hoạt động thể chất, sức khỏe tổng quát và thời gian ngủ của một cá nhân. Mặc dù thuật toán Naïve Bayes giả định tính độc lập giữa các đặc trưng, hệ thống vẫn cho thấy hiệu suất hứa hẹn, đạt độ chính xác khoảng 71-73% trong dự đoán bệnh tim. Các tác giả cho rằng đây là một thành tựu đáng chú ý khi xem xét các khó khăn đã nêu, đặc biệt là trong bối cảnh hạn chế về nguồn lực.

**b) Phương thức Thực hiện:**
1. Thu thập và Chuẩn bị Dữ liệu:
   1. Bộ dữ liệu được thu thập từ Kaggle, chứa các thông tin liên quan đến tình trạng sức khỏe và bệnh tim của các cá nhân.
   1. Các thuộc tính quan trọng được lựa chọn bao gồm: tình trạng bệnh tim, chiều cao, cân nặng, sức khỏe thể chất, khó khăn khi đi lại, nhóm tuổi, hoạt động thể chất, sức khỏe tổng quát và thời gian ngủ.
   1. Dữ liệu được chia thành hai phần: 80% cho tập huấn luyện (training) và 20% cho tập kiểm tra (testing).
1. Tiền xử lý Dữ liệu:
   1. Tính toán BMI: Chỉ số khối cơ thể (BMI) được tính toán dựa trên chiều cao và cân nặng của người dùng.
   1. Mã hóa Biến: Do thuật toán CategoricalNB (một biến thể của Naïve Bayes) được sử dụng và phù hợp với dữ liệu dạng danh mục, các biến liên tục đã được chuyển đổi thành biến danh mục. Các biến danh mục sau đó được mã hóa thành dạng số sử dụng LabelEncoder từ thư viện Scikit-learn để tương thích với thuật toán học máy.
   1. Xử lý Mất cân bằng Lớp (Class Imbalance): Phương pháp RandomUnderSampler từ gói imbalanced-learn được sử dụng để giải quyết vấn đề mất cân bằng giữa số lượng người có bệnh tim và không có bệnh tim trong bộ dữ liệu. Kỹ thuật này giảm ngẫu nhiên số lượng mẫu của lớp đa số để tạo ra một bộ dữ liệu cân bằng hơn.
1. Xây dựng Mô hình Dự đoán:
   1. Thuật toán Naïve Bayes (CategoricalNB): Đây là thuật toán chính được sử dụng để xây dựng mô hình dự đoán. CategoricalNB được chọn vì hầu hết các thuộc tính trong bộ dữ liệu là biến danh mục.
   1. Mô hình được huấn luyện trên tập dữ liệu huấn luyện đã qua tiền xử lý.
1. Phát triển Ứng dụng Web:
   1. Một ứng dụng web được phát triển bằng Flask, một framework web phổ biến của Python, để cung cấp giao diện người dùng cho hệ thống dự đoán.
   1. Visual Studio Code (VS Code) được sử dụng làm môi trường phát triển.
   1. Giao diện người dùng (UI) được thiết kế trực quan để người dùng nhập các thông tin cần thiết (chiều cao, cân nặng, tình trạng sức khỏe, v.v.).
   1. Kết quả dự đoán (nguy cơ mắc bệnh tim thấp hoặc cao) cùng với chỉ số BMI được hiển thị cho người dùng.
1. Đánh giá Mô hình và Hệ thống:
   1. Độ chính xác (Accuracy): Mô hình được đánh giá dựa trên độ chính xác dự đoán trên tập dữ liệu kiểm tra. Kết quả cho thấy độ chính xác dao động trong khoảng 71-73% sau khi áp dụng kỹ thuật cân bằng dữ liệu. (Lưu ý: Trước khi cân bằng, độ chính xác cao hơn (88%) nhưng mô hình bị thiên lệch về dự đoán "Không có bệnh tim").
   1. Các chỉ số khác: Precision, Recall, F1-score cũng được sử dụng để đánh giá hiệu suất của mô hình, đặc biệt là sau khi cân bằng dữ liệu để đảm bảo mô hình không bị thiên lệch.
   1. Đánh giá Tính khả dụng của Hệ thống (Usability Testing): Thang đo Khả năng sử dụng Hệ thống (System Usability Scale - SUS) được sử dụng để đánh giá trải nghiệm người dùng. Hệ thống đạt điểm SUS trung bình là 85%, cho thấy tính khả dụng tốt.



### Thực hành 3: Predicting Order Lead Time for Just in Time production system using various Machine Learning Algorithms: A Case Study

**a) Ý chính của Nghiên cứu:**

Bài báo này tập trung vào việc dự đoán thời gian giao hàng (order lead time) trong môi trường sản xuất Just-in-Time (JIT), cụ thể là cho một nhà hàng. Việc dự đoán chính xác thời gian giao hàng có ý nghĩa quan trọng đối với Quản lý Chuỗi Cung ứng (SCM), bao gồm giảm thiểu hàng tồn kho và chi phí liên quan, tăng thông lượng và năng suất. Nghiên cứu này so sánh hiệu quả của nhiều thuật toán học máy hồi quy có giám sát (supervised regression machine learning algorithms) trong việc dự đoán thời gian giao hàng dựa trên một tập hợp các đặc trưng.

Kết quả cho thấy các mô hình học máy có khả năng dự đoán thời gian giao hàng ở một mức độ đáng kể. Tuy nhiên, bài báo cũng chỉ ra những hạn chế như chưa xem xét các yếu tố thay đổi theo thời gian thực (ví dụ: tình hình giao thông) và đề xuất các hướng nghiên cứu trong tương lai bao gồm việc tích hợp các yếu tố này và sử dụng công nghệ IoT để cải thiện độ chính xác.

**b) Phương thức Thực hiện:**
1. Thu thập và Mô tả Dữ liệu:
   1. Dữ liệu được thu thập liên quan đến các đơn hàng của một nhà hàng.
   1. Các đặc trưng (features) trong dữ liệu thô bao gồm: ID Khách hàng, Thời gian đặt hàng lần đầu, Thời gian đặt hàng gần nhất, Số lượng đơn hàng, Số lượng đơn hàng trong 7 ngày qua, Số lượng đơn hàng trong 4 tuần qua, Tổng số tiền, Số tiền trong 7 ngày qua, Số tiền trong 4 tuần qua, Khoảng cách trung bình từ nhà hàng, và Thời gian giao hàng trung bình (đây là biến mục tiêu - target variable).
   1. Kiểu dữ liệu của các đặc trưng được xác định (số nguyên, dấu thời gian, số thực).
1. Tiền xử lý Dữ liệu:
   1. Xử lý giá trị thiếu (Null/Empty Values): Các giá trị rỗng hoặc thiếu được điền bằng các giá trị phù hợp như giá trị trung bình (mean) hoặc trung vị (median) của cột dữ liệu tương ứng.
   1. Trực quan hóa Dữ liệu: Các biểu đồ như histogram được sử dụng để hiểu rõ hơn về phân phối của dữ liệu, ví dụ như biểu đồ tần suất của Thời gian Giao hàng.
   1. Lựa chọn Đặc trưng (Feature Selection):
      1. Để tránh "lời nguyền chiều dữ liệu" (curse of dimensionality), chỉ các đặc trưng có tương quan cao với biến mục tiêu (thời gian giao hàng) mới được chọn.
      1. Hệ số tương quan Pearson, Kendall và Spearman được sử dụng để đo lường mức độ tương quan tuyến tính và phi tuyến tính giữa các đặc trưng và biến mục tiêu.
      1. Dựa trên kết quả phân tích tương quan (Bảng II trong bài báo), "Average Distance from Restaurant" (Khoảng cách trung bình từ nhà hàng) là đặc trưng có hệ số tương quan Pearson cao nhất (0.0886) với "Average Delivery Time" (Thời gian giao hàng trung bình). Các đặc trưng khác có tương quan rất thấp.
1. Xây dựng và Huấn luyện Mô hình Học máy:
   1. Nghiên cứu sử dụng các thuật toán hồi quy có giám sát sau:
      1. Hồi quy Tuyến tính (Linear Regression)
      1. Hồi quy Ridge (Ridge Regression)
      1. Hồi quy Lasso (Lasso Regression)
      1. Hồi quy Cây Quyết định (Decision Tree Regression)
      1. Hồi quy Rừng Ngẫu nhiên (Random Forest Regression)
      1. Hồi quy K Láng giềng Gần nhất (k-Nearest Neighbors Regression)
   1. Dữ liệu được chia thành tập huấn luyện và tập kiểm tra.
   1. Các tham số của mô hình (ví dụ: giá trị k trong k-NN, số lượng lá tối đa trong Cây Quyết định) được chọn dựa trên việc tối ưu hóa và giảm thiểu lỗi trên tập kiểm tra.
1. Đánh giá Mô hình:
   1. Hiệu suất của các mô hình được đánh giá dựa trên các độ đo lỗi sau trên tập kiểm tra:
      1. Sai số Tuyệt đối Trung bình (Mean Absolute Error - MAE)
      1. Sai số Bình phương Trung bình (Mean SquaredError - MSE)
      1. Căn bậc hai Sai số Bình phương Trung bình (Root Mean Squared Error - RMSE)
      1. Sai số Phần trăm Tuyệt đối Trung bình (Mean Absolute Percentage Error - MAPE)
   1. Kết quả so sánh hiệu quả của các thuật toán được trình bày trong Bảng III của bài báo. Nhìn chung, các thuật toán cho kết quả khá tương đồng nhau về các chỉ số lỗi. Ví dụ, Random Forest Regression cho MAPE thấp nhất (38.062%), trong khi Linear Regression và Ridge Regression có các giá trị MAE, RMSE, MSE gần như nhau và thấp hơn một chút so với các mô hình khác.

### Thực hành 4: A Comparative Study of Machine Learning Algorithms and Explainable AI for Fitness Club Attendance Classification

**a) Ý chính của Nghiên cứu:**

Nghiên cứu này tập trung vào việc xây dựng một mô hình phân loại nhị phân để dự đoán khả năng một thành viên sẽ tham gia (điểm danh) tại một câu lạc bộ thể hình. Mục tiêu là giải quyết thách thức trong việc duy trì sự gắn kết của thành viên, một vấn đề phổ biến đối với các nhà điều hành câu lạc bộ thể hình. Bài báo đề xuất một khung làm việc để đánh giá hiệu suất phân loại điểm danh bằng cách sử dụng nhiều mô hình học máy khác nhau và hiểu rõ các chỉ số đánh giá của chúng.

Nghiên cứu đã sử dụng và so sánh hiệu suất của nhiều thuật toán phân loại như Random Forest, Gradient Boosting, LightGBM, v.v. Kết quả nổi bật cho thấy thuật toán Random Forest đạt độ chính xác cao nhất là 0.87. Bên cạnh đó, nghiên cứu còn ứng dụng AI Giải thích được (Explainable AI - XAI) để xác định các đặc trưng có ảnh hưởng lớn nhất đến kết quả dự đoán.

**b) Phương thức Thực hiện:**
1. Thu thập Dữ liệu (Data Collection):
   1. Sử dụng bộ dữ liệu công khai từ Kaggle.
   1. Bộ dữ liệu bao gồm 7 đặc trưng (features) và 1 biến mục tiêu (target variable - liệu thành viên có tham gia hay không).
   1. Tổng cộng có 1501 mẫu (instances).
   1. Dữ liệu được chia thành 80% cho tập huấn luyện (training) và 20% cho tập kiểm tra (testing).
   1. Các đặc trưng bao gồm: months\_as\_member (số tháng là thành viên), weight (cân nặng), days\_before (số ngày trước buổi tập mà thành viên đăng ký), day\_of\_week (ngày trong tuần của buổi tập), time (thời gian trong ngày của buổi tập - AM/PM), category (loại hình lớp tập như Strength, HIIT, Cycling, Yoga), và biến mục tiêu attended (0 là không tham gia, 1 là có tham gia).
1. Tiền xử lý Dữ liệu (Data Preprocessing):
   1. Kiểm tra và xác nhận bộ dữ liệu không có giá trị bị thiếu (missing values).
   1. Sử dụng kỹ thuật Mã hóa Nhãn (Label Encoding) để chuyển đổi các đặc trưng dạng danh mục (categorical features) sang dạng số, nhằm cải thiện hiệu suất của các mô hình phân loại.
   1. Phân tích tương quan giữa các đặc trưng bằng biểu đồ nhiệt (heatmap) cho thấy không có đặc trưng nào có tương quan cao với nhau, do đó tất cả các đặc trưng đều được giữ lại.
1. Cân bằng Dữ liệu (Data Balancing):
   1. Phân tích phân phối của biến mục tiêu cho thấy sự mất cân bằng giữa hai lớp (số lượng thành viên "không tham gia" nhiều hơn "có tham gia").
   1. Áp dụng kỹ thuật SMOTE (Synthetic Minority Over-sampling Technique) để giải quyết vấn đề mất cân bằng dữ liệu bằng cách tạo ra các mẫu tổng hợp cho lớp thiểu số.
1. Xây dựng Mô hình (Model Building):
   1. Đây là bài toán phân loại nhị phân (attended/not-attended).
   1. Sử dụng phương pháp kiểm định chéo k-fold (k-fold cross-validation) với k=5 để chia dữ liệu và đánh giá mô hình một cách khách quan hơn, giảm thiểu sự phụ thuộc vào một cách chia dữ liệu cụ thể.
   1. Triển khai và so sánh tổng cộng 12 thuật toán phân loại khác nhau, bao gồm:
      1. Random Forest
      1. Gradient Boosting
      1. Decision Tree
      1. K-Nearest Neighbors (KNN)
      1. XGBoost
      1. AdaBoost
      1. LightGBM
      1. Perceptron
      1. CatBoost
      1. Support Vector Machine (SVM)
      1. Logistic Regression
      1. Naïve Bayes
1. Đánh giá Mô hình và AI Giải thích được (Model Evaluation & Explainable AI):
   1. Hiệu suất của mỗi mô hình được đánh giá dựa trên các chỉ số: Accuracy (Độ chính xác), Precision (Độ chuẩn), Recall (Độ bao phủ), F1-score, và Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
   1. Do tính chất nhị phân của bài toán, các chỉ số micro-averaged recall, accuracy, và F1-score được đặc biệt chú trọng.
   1. Sử dụng các kỹ thuật XAI là LIME (Local Interpretable Model-agnostic Explanations) và SHAP (SHapley Additive exPlanations) để giải thích các dự đoán và quyết định của mô hình học máy, cụ thể là làm nổi bật tầm quan trọng của các đặc trưng.
      1. LIME được sử dụng để giải thích các dự đoán riêng lẻ.
      1. SHAP được sử dụng để hiểu tầm quan trọng của đặc trưng trên toàn cục.

Kết quả cho thấy sau khi áp dụng SMOTE, thuật toán Random Forest đạt hiệu suất tốt nhất với độ chính xác 0.87 và AUC là 0.88. Phân tích XAI bằng SHAP chỉ ra rằng months\_as\_member (số tháng là thành viên) là đặc trưng quan trọng nhất, tiếp theo là weight (cân nặng).
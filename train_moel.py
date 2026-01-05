from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time

# 1. Cấu hình Spark (Đã chỉnh lại RAM cho phù hợp máy ảo)
spark = SparkSession.builder \
    .appName("Housing_Price_Prediction_Model") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR") # Giảm bớt log rác

# 2. Đọc dữ liệu TỪ HDFS (Quan trọng: Đã sửa đường dẫn tại đây)
print(">>> Dang doc du lieu tu HDFS...")
path = "hdfs://localhost:9000/user/housing_data/housing_data_cleaned_v3.csv"
df = spark.read.csv(path, header=True, inferSchema=True)

# ---------------------------------------------------------
# BƯỚC 1: XỬ LÝ DỮ LIỆU & LỌC NHIỄU
# ---------------------------------------------------------

# Tạo biến mục tiêu (Label)
# Tổng giá = Diện tích * Giá/m2
df = df.withColumn("total_price", F.col("squared_meter_area") * F.col("price_in_million_per_square_meter"))
# Lấy Logarit để giảm độ lệch của dữ liệu giá (giúp mô hình học tốt hơn)
df = df.withColumn("label_log", F.log1p("total_price"))

# Lọc nhiễu (Outliers)
print(">>> Dang tinh toan nguong loc nhieu...")
q_price = df.stat.approxQuantile("price_in_million_per_square_meter", [0.05, 0.95], 0.0)
q_area = df.stat.approxQuantile("squared_meter_area", [0.05, 0.95], 0.0)

iqr_price = q_price[1] - q_price[0]
iqr_area = q_area[1] - q_area[0]

# Lọc chặt tay
df_clean = df.filter(
    (F.col("price_in_million_per_square_meter").between(q_price[0] - 0.5*iqr_price, q_price[1] + 0.5*iqr_price)) &
    (F.col("squared_meter_area").between(q_area[0] - 0.5*iqr_area, q_area[1] + 0.5*iqr_area))
)

# Chia tập Train/Test
train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)

# Cache dữ liệu để chạy nhanh hơn
train_df = train_df.repartition(4).cache()
test_df = test_df.repartition(4).cache()
print(f"Size: Train={train_df.count()}, Test={test_df.count()}")

# ---------------------------------------------------------
# BƯỚC 2: TARGET ENCODING (MÃ HÓA CỘT CHỮ THÀNH SỐ)
# ---------------------------------------------------------
cat_cols = ['district', 'ward', 'street', 'type_of_housing', 'legal_paper']
global_mean = train_df.select(F.mean("label_log")).first()[0]
ALPHA = 10 

print(">>> Dang thuc hien Encoding (Ma hoa du lieu)...")
mappings = {}

for col in cat_cols:
    agg_df = train_df.groupBy(col).agg(
        F.sum("label_log").alias("sum_log"),
        F.count("label_log").alias("count_log")
    )
    agg_df = agg_df.withColumn(
        col + "_encoded",
        (F.col("sum_log") + (F.lit(global_mean) * ALPHA)) / (F.col("count_log") + ALPHA)
    ).select(col, col + "_encoded")

    mappings[col] = F.broadcast(agg_df)

def apply_encoding(df_in, mappings):
    res = df_in
    for col in cat_cols:
        res = res.join(mappings[col], on=col, how="left")
        res = res.fillna({col + "_encoded": global_mean})
    return res

train_encoded = apply_encoding(train_df, mappings)
test_encoded = apply_encoding(test_df, mappings)

train_encoded = train_encoded.localCheckpoint()
test_encoded = test_encoded.localCheckpoint()

# ---------------------------------------------------------
# BƯỚC 3: HUẤN LUYỆN MODEL
# ---------------------------------------------------------
num_cols = ['num_floors', 'num_bed_rooms', 'squared_meter_area']
assembler = VectorAssembler(inputCols=[c + "_encoded" for c in cat_cols] + num_cols, outputCol="features")

# Chuyển đổi dữ liệu
train_vec = assembler.transform(train_encoded).select("features", "label_log")
test_vec = assembler.transform(test_encoded).select("features", "label_log", "total_price")

print(">>> Dang huan luyen GBT (Gradient Boosted Trees)...")
gbt = GBTRegressor(featuresCol="features", labelCol="label_log",
                   maxIter=100, maxDepth=8, stepSize=0.1, seed=42) # Giảm iter xuống 100 cho nhẹ máy
gbt_model = gbt.fit(train_vec)

print(">>> Dang huan luyen Random Forest...")
rf = RandomForestRegressor(featuresCol="features", labelCol="label_log",
                           numTrees=50, maxDepth=10, seed=42) # Giảm cây xuống 50 cho nhẹ máy
rf_model = rf.fit(train_vec)

# ... (Phần code trên giữ nguyên) ...

# ---------------------------------------------------------
# BƯỚC 4: DỰ ĐOÁN & ĐÁNH GIÁ (ENSEMBLE)
# ---------------------------------------------------------
pred_gbt = gbt_model.transform(test_vec).withColumnRenamed("prediction", "p_gbt")
pred_final = rf_model.transform(pred_gbt).withColumnRenamed("prediction", "p_rf")

# Kết hợp kết quả 2 model (60% GBT + 40% RF)
pred_final = pred_final.withColumn("avg_log", (F.col("p_gbt") * 0.6 + F.col("p_rf") * 0.4))

# Chuyển ngược từ Log về giá tiền thực tế
pred_final = pred_final.withColumn("pred_real", F.expm1("avg_log"))

# 1. Đánh giá R2
evaluator_r2 = RegressionEvaluator(labelCol="total_price", predictionCol="pred_real", metricName="r2")
r2 = evaluator_r2.evaluate(pred_final)

# 2. Đánh giá MAE (Mean Absolute Error) - Sai số tuyệt đối trung bình
evaluator_mae = RegressionEvaluator(labelCol="total_price", predictionCol="pred_real", metricName="mae")
mae = evaluator_mae.evaluate(pred_final)

# 3. Đánh giá RMSE (Root Mean Squared Error) - Căn bậc hai sai số bình phương trung bình
evaluator_rmse = RegressionEvaluator(labelCol="total_price", predictionCol="pred_real", metricName="rmse")
rmse = evaluator_rmse.evaluate(pred_final)

print(f"\n{'='*40}")
print(f" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
print(f"{'='*40}")
print(f" R2 Score (Độ phù hợp): {r2:.4f} ({r2*100:.2f}%)")
print(f" MAE (Sai số trung bình): {mae:,.2f} (triệu VND)")
print(f" RMSE (Độ lệch chuẩn sai số): {rmse:,.2f} (triệu VND)")
print(f"{'='*40}")

# Dừng Spark
spark.stop()

# ... (Đoạn code tính toán R2, MAE, MAPE ở trên giữ nguyên) ...

print(">>> Dang ve bieu do bao cao...")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------
# BƯỚC 1: LẤY DỮ LIỆU ĐỂ VẼ
# ---------------------------------------------------------
# Chỉ lấy khoảng 5% dữ liệu Test để vẽ Scatter Plot cho nhẹ (tránh treo máy)
# Chuyển từ Spark DataFrame -> Pandas DataFrame
df_plot = pred_final.select("total_price", "pred_real").sample(withReplacement=False, fraction=0.05, seed=42).toPandas()

# ---------------------------------------------------------
# BIỂU ĐỒ 1: SO SÁNH GIÁ THỰC TẾ vs GIÁ DỰ ĐOÁN (SCATTER PLOT)
# Đây là biểu đồ quan trọng nhất để chứng minh độ chính xác (R2)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Vẽ các điểm dự đoán
sns.scatterplot(x=df_plot["total_price"], y=df_plot["pred_real"], alpha=0.6, color="blue", label="Dữ liệu kiểm thử")

# Vẽ đường chéo hoàn hảo (Ideal Line) - Nơi Giá Thực = Giá Dự Đoán
max_val = max(df_plot["total_price"].max(), df_plot["pred_real"].max())
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label="Đường lý tưởng (Hoàn hảo)")

plt.title(f"Biểu đồ Tương quan: Giá Thực tế vs Dự đoán (R2={r2*100:.1f}%)", fontsize=14)
plt.xlabel("Giá Thực Tế (Triệu VNĐ)", fontsize=12)
plt.ylabel("Giá Mô Hình Dự Đoán (Triệu VNĐ)", fontsize=12)
plt.legend()
plt.tight_layout()

# Lưu ảnh 1
plt.savefig('chart_scatter_r2.png')
print(">>> Da luu bieu do 1: chart_scatter_r2.png")

# ---------------------------------------------------------
# BIỂU ĐỒ 2: TRỰC QUAN HÓA SAI SỐ (MAE & RMSE)
# Dạng biểu đồ cột để so sánh mức độ sai số
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

# Dữ liệu vẽ
metrics = ['MAE (Trung bình)', 'RMSE (Bị phạt nặng)']
values = [mae, rmse]
colors = ['#2ecc71', '#e74c3c'] # Xanh lá và Đỏ

bars = plt.bar(metrics, values, color=colors, width=0.5)

# Viết số lên đầu cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 20, f'{yval:,.0f} tr', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("Biên độ Sai số của Mô hình (Thấp hơn là Tốt hơn)", fontsize=14)
plt.ylabel("Sai số (Triệu VNĐ)", fontsize=12)
plt.ylim(0, max(values) * 1.2) # Tăng chiều cao biểu đồ để số không bị cắt
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Lưu ảnh 2
plt.savefig('chart_error_metrics.png')
print(">>> Da luu bieu do 2: chart_error_metrics.png")

# Dừng Spark
spark.stop()

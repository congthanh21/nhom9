from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sys

# 1. Cấu hình Spark
spark = SparkSession.builder \
    .appName("Housing_Price_Prediction_With_Percentage") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Đọc dữ liệu
path = "hdfs://localhost:9000/user/housing_data/housing_data_cleaned_v3.csv"
print(f">>> Dang doc du lieu tu: {path}")

try:
    df = spark.read.csv(path, header=True, inferSchema=True)
except:
    print("LOI: Khong doc duoc file. Kiem tra lai duong dan.")
    sys.exit(1)

# ---------------------------------------------------------
# XỬ LÝ DỮ LIỆU
# ---------------------------------------------------------
df = df.withColumn("total_price", F.col("squared_meter_area") * F.col("price_in_million_per_square_meter"))
df = df.withColumn("label_log", F.log1p("total_price"))

# Lọc nhiễu
q_price = df.stat.approxQuantile("price_in_million_per_square_meter", [0.05, 0.95], 0.0)
q_area = df.stat.approxQuantile("squared_meter_area", [0.05, 0.95], 0.0)
iqr_price = q_price[1] - q_price[0]
iqr_area = q_area[1] - q_area[0]

df_clean = df.filter(
    (F.col("price_in_million_per_square_meter").between(q_price[0] - 0.5*iqr_price, q_price[1] + 0.5*iqr_price)) & 
    (F.col("squared_meter_area").between(q_area[0] - 0.5*iqr_area, q_area[1] + 0.5*iqr_area))
)

train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)
train_df = train_df.repartition(4).cache()
test_df = test_df.repartition(4).cache()

# ---------------------------------------------------------
# ENCODING
# ---------------------------------------------------------
cat_cols = ['district', 'ward', 'street', 'type_of_housing', 'legal_paper']
global_mean = train_df.select(F.mean("label_log")).first()[0]
ALPHA = 10 
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

train_encoded = apply_encoding(train_df, mappings).localCheckpoint()
test_encoded = apply_encoding(test_df, mappings).localCheckpoint()

# ---------------------------------------------------------
# TRAINING (60% RF - 40% GBT)
# ---------------------------------------------------------
num_cols = ['num_floors', 'num_bed_rooms', 'squared_meter_area']
assembler = VectorAssembler(inputCols=[c + "_encoded" for c in cat_cols] + num_cols, outputCol="features")

train_vec = assembler.transform(train_encoded).select("features", "label_log")
test_vec = assembler.transform(test_encoded).select("features", "label_log", "total_price")

print(">>> Dang huan luyen Random Forest...")
rf = RandomForestRegressor(featuresCol="features", labelCol="label_log", numTrees=50, maxDepth=10, seed=42)
rf_model = rf.fit(train_vec)

print(">>> Dang huan luyen GBT...")
gbt = GBTRegressor(featuresCol="features", labelCol="label_log", maxIter=100, maxDepth=8, stepSize=0.1, seed=42)
gbt_model = gbt.fit(train_vec)

# ---------------------------------------------------------
# EVALUATION & TÍNH PHẦN TRĂM SAI SỐ
# ---------------------------------------------------------
pred_gbt = gbt_model.transform(test_vec).withColumnRenamed("prediction", "p_gbt")
pred_final = rf_model.transform(pred_gbt).withColumnRenamed("prediction", "p_rf")

# Trọng số: 0.6 RF + 0.4 GBT
pred_final = pred_final.withColumn("avg_log", (F.col("p_rf") * 0.6 + F.col("p_gbt") * 0.4))
pred_final = pred_final.withColumn("pred_real", F.expm1("avg_log"))

# 1. Tính các chỉ số cơ bản
evaluator = RegressionEvaluator(labelCol="total_price", predictionCol="pred_real", metricName="r2")
r2 = evaluator.evaluate(pred_final)

mae = evaluator.setMetricName("mae").evaluate(pred_final)
rmse = evaluator.setMetricName("rmse").evaluate(pred_final)

# 2. TÍNH MAPE (Mean Absolute Percentage Error) - SAI SỐ %
# Công thức: Trung bình của |(Giá Thực - Giá Dự Đoán) / Giá Thực| * 100
pred_final = pred_final.withColumn("ape", F.abs((F.col("total_price") - F.col("pred_real")) / F.col("total_price")) * 100)
mape = pred_final.select(F.mean("ape")).first()[0]

# 3. Tính Giá nhà trung bình để so sánh
mean_price = test_vec.select(F.mean("total_price")).first()[0]
mae_percent = (mae / mean_price) * 100
rmse_percent = (rmse / mean_price) * 100

print(f"\n{'='*50}")
print(f" KET QUA CHI TIET (DON VI: TRIEU VND & %)")
print(f"{'='*50}")
print(f" 1. DO CHINH XAC (Accuracy):")
print(f"    - R2 Score:        {r2*100:.2f} %  (Giai thich duoc {r2*100:.1f}% bien dong gia)")
print(f"")
print(f" 2. SAI SO TUYET DOI (Error in Money):")
print(f"    - MAE (Trung binh):  {mae:,.2f} Trieu VND")
print(f"    - RMSE (Binh phuong):{rmse:,.2f} Trieu VND")
print(f"")
print(f" 3. SAI SO TUONG DOI (Error in Percentage):")
print(f"    - MAPE (Sai so %):   {mape:.2f} %  (Trung binh moi can lech khoang {mape:.1f}%)")
print(f"    - MAE / Gia TB:      {mae_percent:.2f} %")
print(f"{'='*50}")
print(f" Gia nha trung binh trong tap Test: {mean_price:,.2f} Trieu VND")
print(f"{'='*50}")

spark.stop()

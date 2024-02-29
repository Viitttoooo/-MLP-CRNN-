# 此代码为多层感知机分类代码
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.linalg import DenseVector
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# 注意修改为自己的主机名
conf = SparkConf().setAppName("mlp").setMaster("spark://master:7077")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
spark = SparkSession(sc)

# 注意修改路径
TRAINPATH = "/mlp/output1.csv"
TESTPATH = "/mlp/test1.csv"

def GetParts(line):
    parts = line.split(',')
    return float(parts[-1]), DenseVector([float(x) for x in parts[:-1]])

rdd_train = sc.textFile(TRAINPATH)
rdd_test = sc.textFile(TESTPATH)

rdd_train = rdd_train.map(lambda line: GetParts(line))
rdd_test = rdd_test.map(lambda line: GetParts(line))

df_train = spark.createDataFrame(rdd_train, schema=["label", "features"])
df_test = spark.createDataFrame(rdd_test, schema=["label", "features"])

assembler = VectorAssembler(inputCols=["features"], outputCol="dense_features")
df_train = assembler.transform(df_train).select("label", "dense_features")
df_train = df_train.withColumnRenamed("dense_features", "features")  # Rename the features column
df_test = assembler.transform(df_test).select("label", "dense_features")
df_test = df_test.withColumnRenamed("dense_features", "features")  # Rename the features column

indexer = StringIndexer(inputCol="label", outputCol="label_index", handleInvalid="keep")
indexer_model = indexer.fit(df_train)
df_train = indexer_model.transform(df_train)
df_test = indexer_model.transform(df_test)

encoder = OneHotEncoder(inputCol="label_index", outputCol="label_encoded", handleInvalid="keep")
encoder_model = encoder.fit(df_train)
df_train = encoder_model.transform(df_train)
df_test = encoder_model.transform(df_test)

# Convert the label column data type to DoubleType
df_train = df_train.withColumn("label", df_train["label"].cast(DoubleType()))

# 此处由于字分类器和数字字母分类器的分类数不同，所以对于不同的分类器要修改最后一层神经元数量
layers = [len(df_train.select("features").first().features), 1024, 512, 256, 34]
mlp = MultilayerPerceptronClassifier(layers=layers, seed=1234)
model = mlp.fit(df_train)

predictions = model.transform(df_test)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: ", accuracy)
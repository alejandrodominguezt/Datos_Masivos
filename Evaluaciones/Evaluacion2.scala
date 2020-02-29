//1 
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

//2
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

//3
df.columns
//Date,Open,High,Low,Close,Volume,adj Close.

//4
df.printSchema()

//5
df.head(5)

//6
df.describe().show()

//7
val df2 = df.withColumn("HV Ratio",(df("High") / df("Volume")));

//8
df.select(max("Close")).show()
df.select(day(df("Date"))).show()

//9
val total = df.filter($"High">0).count()
val result = df.filter($"High">500).count()
val porcenta = result * 100;
val procentaje =  porcenta / total;
val dohhh = procentaje  * 1.0;


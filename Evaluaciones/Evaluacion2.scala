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
val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))
df2.show()

//8
df.orderBy($"Close".desc).show(1)

//9
// "Close" column represents the closing price per day
df.select(mean("Close")).show()

//10
df.select(max("Volume")).show()
df.select(min("Volume")).show

//11
  //a. 
df.filter($"Close"<600).count()

  //b. 
(df.filter($"High" > 500).count * 1.0 / df.count())*100

  //c.
df.select(corr("High", "Volume")).show()

  //d.
val ydf = df.withColumn("Year", year(df("Date")))
val maxdf = ydf.select($"Year", $"High").groupBy("Year").max()
maxdf.select($"Year", $"max(High)").orderBy("Year").show()

  //e.
val mdf = df.withColumn("Month", month(df("Date")))
val promdf = mdf.select($"Month", $"Close").groupBy("Month").mean()
promdf.select($"Month", $"avg(Close)").orderBy("Month").show()



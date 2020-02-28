// Practice 1 U2 20 functions
// Example df.select(corr("High", "Low")).show()

import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")


// Test 20 functions for the variable "df"
//1
df.select("Low").show()

//2
df.count()

//3
df.describe()

//4
df.filter($"Open" < 470 && $"Close" < 470).show()

//5
df.select(mean("Low")).show()

//6
df.select(max("High")).show()

//7
df.select(min("High")).show()

//8
df.select(month(df("Date"))).show()

//9
df.select(year(df("Date"))).show()

//10
df.filter($"Low"===482.7).show()

//11
df.head(5)

//12
df.select(avg("Close")).show()

//13
df.first()

//14
df.last()

//15
df.select(sum("High")).show()

//16
df.select(variance("High")).show()

//17
df.select(kurtosis("High")).show()

//18
df.select(bin("High")).show()

//19
df.groupBy("Low").agg(max("Low")).show()

//20
df.select(collect_list("High")).show()

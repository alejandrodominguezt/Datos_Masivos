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

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Sales.csv")

//11
df.select(approx_count_distinct("Company")).show()

//12
df.select(collect_set("Company")).show()

//13
df.select(sum("Sales")).show()

//14
df.select(bin("open")).show()

//15
df.groupBy("Company").agg(max("Sales")).show()

//16
df.groupBy("Company").agg(min("Sales")).show()

//17 
df.groupBy("Company").agg(avg("Sales")).show()

//18
df.groupBy($"Company").min().show

//19
df.groupBy($"Company").max().show

//20
df.groupBy($"Company").avg().show

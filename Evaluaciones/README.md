# Evaluation 1
**Domínguez Tabardillo David Alejandro - 15211698** 

 **Soto Pino Saúl - 15211705**
 
 **Index**
* [1.-Evaluation 1](#e1)
* [2.-Evaluation 2](#e2)

<a name="e1"></a>

**Assessment 1/Evaluation 1:**

### INSTRUCCIONS

Given a square matrix, calculate the absolute difference between the sums of your diagonals.
For example, the square matrix is ​​shown below:

**arr** [[11, 2, 4],[4, 5, 6], [10, 8, -12]]

**diagonal_1** = 11 + 5 -12 = 4

**diagonal_2**  = 4 + 5 + 10 = 19

**Absolute difference** = | 4 -19 | = 15
	
### DESCRIPTIVE FUNCTION

*Develop a function called Diagonal Difference in a script with the Scala programming language. This must return an integer that represents the difference of the absolute diagonal*

diagonalDifference takes the following parameter:

arr


### DEVELOPMENT

*We created the matrix using the example*
```
val arr = Array(Array(11,2,4),Array(4,5,6),Array(10,8,-12));
```
*We created the "diagonalDifference" function that receives the example array, n and the Boolean values according to the diagonal*

```
def diagonalDifference(arr:Array[Array[Int]], n:Int, diagonal_1:Boolean, diagonal_2: Boolean):Int=
{

```
* The sum variable was declared that will keep the path of the example array by adding the diagonal 1*
```
    var sum:Int=0
```
*We created a conditional for diagonal 1, that recieves the boolean paratemer from the diagonalDifference function*
```
   if (diagonal_1){
 ```
 *A for cycle was used to traverse the example array and with Range to obtain the ordered sequence of integers that are equally spaced on diagonal 1, the sum variable keeps the elements obtained from the matrix.*
 ```
       for(i<-Range(0,n))
       {
           sum = sum + arr(i)(i)
       }
   }
   ```
*The sum2 variable was declared that will keep the path of the example array by adding the diagonal 2*
   ```
   var sum2:Int=0
   ```
 *We created a conditional for diagonal 2, that recieves the boolean paratemer from the diagonalDifference function* 
 ```
   if (diagonal_2){
 ```
 *A for cycle was used to traverse the example array and with Range to obtain the ordered sequence of integers that are equally spaced on diagonal 2, the sum2 variable keeps the elements obtained from the matrix.*
 ```
       for(i <- Range(0,n)){
           sum2 = sum2 + arr(i)((n-1)-i)
       }
   }
 ```
  *We return the ** absolute difference ** using the Math.abs function of the subtraction of the variables that saved the sum of the diagonals*
 ```
   return Math.abs(sum - sum2)
  } 
 ```
 *Finally, the diagonalDifference function is called and sent as parameters: the example array "arr", the value of the length of the matrix "n" and the Boolean values true for both diagonals*
  ```
diagonalDifference(arr,3,true, true)

```

<a name="e2"></a>

# Evaluation 2

### 1. Start a simple Spark session
*A spark session was imported*
```
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
```
### 2. Load the Netflix Stock CSV file, have Spark infer the data types
*The file is loaded with data, inferSchema indicates the inference of the data types*
```
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
```

### 3. What are the names of the columns?
*Result: Date, Open, High, Low, Close, Volume, Adj Close*
```
df.columns
```

### 4. How is the scheme?
*PrintSchema shows the complete structure, column names and data types*
```
df.printSchema()
```
### 5. Print the first 5 columns
*Head shows the column data, 5 represents the number of columns it will take*
```
df.head(5)
```
### 6. Use describe () to learn about the DataFrame
*Describe shows different rows with the functions applied as min, max and mean*
```
df.describe().show()
```
### 7. Create a new dataframe with a new column called "HV Ratio" which is the relationship between the price of the "High" column versus the "Volume" column of shares traded for a day
*A value called hrdf is created that will save the creation of the new column "HV Ratio". This column will have the division of the "High" and "Volume" columns*
```
val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))
df2.show()
```
### 8. What day had the highest peak in the "Close" column?
*With order By, the values in the "Close" column are sorted in descending order. With Show (1) it is indicated that it only shows the first value, which will be the highest on that day.*
```
df.orderBy($"Close".desc).show(1)
```
### 9. What is the meaning of the Close column?
 *"Close" column represents the closing price per day*
 ```
df.select(mean("Close")).show()
```
### 10. What is the maximum and minimum of the "Volume" column?
*max gets the maximum value of the Volume column, respectively min the minimum. Select is similar to an SQL query with Scala code and with show it shows*
```
df.select(max("Volume")).show()
df.select(min("Volume")).show
```
### 11. With Scala / Spark $ syntax answer the following:
  **a. ¿Cuántos días fue la columna "Close" inferior a $600?** 
  *Se filtran los datos que coincidan con la regla (valores menores a 600 de la columna "Close")
   Count muestra la cantidad de resultados coincidentes con la regla*
  ```
df.filter($"Close"<600).count()
```
  **b. ¿Qué porcentaje del tiempo fue la columna "High" mayor que $500?** 
  *Se filtran los datos que coincidan con la regla (Valores mayores a 500 de la columna High)
   Se cuentan con count la cantidad de resultados
   Se implementa una multiplicación * 1.0 para convertir la operación en Double
   Se divide entre el número de registros del DataFrame
   El resultado obtenido se multiplica por 100 para obtener el porcentaje*
  ```
(df.filter($"High" > 500).count * 1.0 / df.count())*100
```
  **c. ¿Cuál es la correlación de pearson entre la columna "High" y la culumna "Volume"?**
  *Se selecciona y muestra la correlación de "High" y "Volume "utilizando la funcion corr *
  ```
df.select(corr("High", "Volume")).show()
```
  **d. ¿Cuál es el máximo de la columna "High" por Año?**
  *Se creo un value que guardará la creación de la nueva columna "Year"
   La función year obtiene de la columna "Date" los años y los guarda en la nueva columna
   Se creo otra variable para multiples columnas, en la que la columna de los años se junta con la columna "High"
   Se agrupan con groupBy por año y se busca el máx
   Se obtienen 2 columnas "Year" y "max(High)" las cuales son ordenadas por año y mostradas*
  ```
val ydf = df.withColumn("Year", year(df("Date")))
val maxdf = ydf.select($"Year", $"High").groupBy("Year").max()
maxdf.select($"Year", $"max(High)").orderBy("Year").show()
```
  **e. ¿Cuál es el promedio de la columna "Close" para cada mes del calendario?**
  *Se creo un value que guardará la creación de la columna "Month"
   La función month obtiene el mes de la columna "Date" y los guarda en la nueva columna
   Se creo otra variable para multiples columnas, en la que la columna de los meses se junta con la columna "High"
   Se agrupan con groupBy por mes y se busca el promedio con mean()
   Se obtienen 2 columnas "Month" y "avg(Close)" las cuales son ordenadas por mes y mostradas*
  ```
val mdf = df.withColumn("Month", month(df("Date")))
val promdf = mdf.select($"Month", $"Close").groupBy("Month").mean()
promdf.select($"Month", $"avg(Close)").orderBy("Month").show()
```



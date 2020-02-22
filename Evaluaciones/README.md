# Evaluación 1
**Domínguez Tabardillo David Alejandro - 15211698** 
 **Soto Pino Saúl - 1521____**

Instructions:




*Se creó el arreglo usado de ejemplo*
```
val arr = Array(Array(11,2,4),Array(4,5,6),Array(10,8,-12));
```
*Se definió la función "diagonalDifference" que recibe el arreglo ejemplo, n y los valores booleanos de acuerdo a la diagonal*

```
def diagonalDifference(arr:Array[Array[Int]], n:Int, diagonal_1:Boolean, diagonal_2: Boolean):Int=
{

```
* Se declaró la variable sum que guardará el recorrido del arreglo ejemplo sumando la diagonal 1*
```
    var sum:Int=0
```
*Se condicionó la diagonal 1, que recibe el parámetro booleano de la función diagonalDifference*
```
   if (diagonal_1){
 ```
 *Se usó un ciclo for para recorrer el arreglo ejemplo y con Range obtener la secuencia ordenada de enteros que están igualmente espaciados en la diagonal 1, la variable sum va guardando los elementos obtenidos de la matriz.*
 ```
       for(i<-Range(0,n))
       {
           sum = sum + arr(i)(i)
       }
   }
   ```
*Se declaró la variable sum que guardará el recorrido del arreglo ejemplo sumando la diagonal 2*
   ```
   var sum2:Int=0
   ```
 *Se condicionó la diagonal 2, que recibe el parámetro booleano de la función diagonalDifference* 
 ```
   if (diagonal_2){
 ```
 *Se usó un ciclo for para recorrer el arreglo ejemplo y con Range obtener la secuencia ordenada de enteros que están igualmente espaciados en la diagonal 2, la variable sum va guardando los elementos obtenidos de la matriz.*
 ```
       for(i <- Range(0,n)){
           sum2 = sum2 + arr(i)((n-1)-i)
       }
   }
 ```
  *Retornamos la **diferencia absoluta** utilizando la función Math.abs de la resta de las variables que guardaron la suma de las diagonales*
 ```
   return Math.abs(sum - sum2)
  } 
 ```
 *Finalmente se manda a llamar a la función diagonalDifference y se le mandan como parámetros: el arreglo ejemplo "arr", el valor de la longitud de la matriz "n" y los valores booleanos true para ambas diagonales*
  ```
diagonalDifference(arr,3,true, true)

```

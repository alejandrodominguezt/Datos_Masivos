# Evaluation 1
**Domínguez Tabardillo David Alejandro - 15211698** 

 **Soto Pino Saúl - 15211705**


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

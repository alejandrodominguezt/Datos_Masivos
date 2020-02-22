
// Evaluación 1
// Domínguez Tabardillo David Alejandro
// Soto Pino Saúl


val arr = Array(Array(11,2,4),Array(4,5,6),Array(10,8,-12));

def diagonalDifference(arr:Array[Array[Int]], n:Int, diagonal_1:Boolean, diagonal_2: Boolean):Int=
{
    var sum:Int=0
   if (diagonal_1){
       for(i<-Range(0,n))
       {
           sum = sum + arr(i)(i)
       }
   }
   var sum2:Int=0
   if (diagonal_2){
       for(i <- Range(0,n)){
           sum2 = sum2 + arr(i)((n-1)-i)
       }
   }
  
   return Math.abs(sum - sum2)
  } 
diagonalDifference(arr,3,true, true)

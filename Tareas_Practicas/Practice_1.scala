// Assessment 1/Practice 1

//1. Develop an algorithm in scala that calculates the radius of a circle

println("Enter Diameter: ")
val diam: Double = scala.io.StdIn.readLine.toInt 
val rad: Double = diam/2
println("The radius is: " + rad) 

//2.Develop an algorithm in scala that tells me if a number is prime
def prime(i :Int) : Boolean = {
if (i <= 1)
false
else if (i == 2)
true
else
!(2 to (i-1)).exists(x => i % x == 0)
}
print("Enter upper limit: ")
val up: Int = scala.io.StdIn.readLine.toInt

print("Enter lower limit: ")
val low: Int = scala.io.StdIn.readLine.toInt

(low to up).foreach(i => if (prime(i)) println("%d is prime.".format(i)))

//3. Given the variable bird = "tweet", use string interpolation to
// print "I am writing a tweet" 
var bird = "tweet"
println(s"I am writing a $bird") 

//4. Given the variable message = "Hi Luke, I'm your father!" use slice to extract the sequence "Luke"


//5.


//6. 



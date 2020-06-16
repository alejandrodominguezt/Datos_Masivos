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
var starWars = "Hola Luke yo soy tu padre!"
println(starWars.slice(5,9))

//5.What's the difference between Value and Variable
println("val defines a constant, a fixed value which cannot be modified once declared and assigned while var defines a variable, which can be modified or reassigned.")

//6. Given the tuple ((2,4,5),(1,2,3),(3.1416,23))) return the number 3.1416 
val my_tuple = ((2,4,5),(1,2,3),(3.1416,23));
my_tuple._3._1


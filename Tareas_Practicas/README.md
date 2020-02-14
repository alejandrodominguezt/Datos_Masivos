# PRACTICES (Description)
### Domínguez Tabardillo David Alejandro 15211698
### Saúl Soto Pino 1521

## Practice 1 
**// Assessment 1/Practice 1**

//1. Develop an algorithm in scala that calculates the radius of a circle
*// User enters diameter data. Radio is the double the diameter diameter is divided between two.*

println("Enter Diameter: ")
val diam: Double = scala.io.StdIn.readLine.toInt // Enter data
val rad: Double = diam/2
println("The radius is: " + rad) 

//2.Develop an algorithm in scala that tells me if a number is prime
*// Determine whether it is prime or not Boolean values. User enters a range of data and the existing prime values are displayed*

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
*// variable bird saves the word "tweet" and is implied with the symbol $ contemplating the use string "s" *
var bird = "tweet"
println(s"I am writing a $bird") 

//4. Given the variable message = "Hi Luke, I'm your father!" use slice to extract the sequence "Luke"


//5.


//6. 

## Practice 2 
**//Assessment 1/ Practice 2**
// 1. Create a list called "list" with the elements "red", "white", "black"
        *// Option 1  Using ListBuffer helps to list according to the order in which the elements are added.*
val list = collection.mutable.ListBuffer("red","white","black")
        *// Option 2 Create a simple list*
val list = List("red","white","black")

// 2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
        *//Option 1 Add item one by one*
list += "green"
list +="yellow"
list +="blue"
list +="orange"
list += "pearl"
        *//Option 2 Creating a new list and add multiple item*
val list2 = "green" ::"yellow" :: "blue" :: "orange" :: "pearl" :: list

// 3. Bring the elements of "list" "green", "yellow", "blue"
        *//For option 1 Using slice and giving the coordinates*
list slice(3,6)
        *//For option 2 Using slice and giving the coordinates*
list2 slice(0,3)

// 4. Create a number array in the 1-1000 range in 5-in-5 steps
*//The first value indicates where the fix will start, the second value indicates the limit and the third value jumps*
Array.range(1, 1000,5)

// 5. What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets
*// toSet extracts the unique values from the list*

val list = List(1,3,3,4,6,7,3,7)
list.toSet

// 6. Create a mutable map called names that contains the following "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"

// 6 a .

// 7 b .

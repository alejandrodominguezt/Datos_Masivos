//Assessment 1/ Practice 2
// 1. Create a list called "list" with the elements "red", "white", "black"
        // Option 1  
val list = collection.mutable.ListBuffer("red","white","black")
        // Option 2
val list = List("red","white","black")

// 2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
        //Option 1 
list += "green"
list +="yellow"
list +="blue"
list +="orange"
list += "pearl"
        //Option 2
val list2 = "green" ::"yellow" :: "blue" :: "orange" :: "pearl" :: list

// 3. Bring the elements of "list" "green", "yellow", "blue"
        //For option 1
list slice(3,6)
        //For option 2
list2 slice(0,3)

// 4. Create a number array in the 1-1000 range in 5-in-5 steps
Array.range(1, 1000,5)

// 5. What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets
// toSet extracts the unique values from the list
val list = List(1,3,3,4,6,7,3,7)
list.toSet

// 6. Create a mutable map called names that contains the following "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"

// 6 a .

// 7 b .

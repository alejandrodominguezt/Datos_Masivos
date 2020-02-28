// Practice 5

// 1 
df.select(approx_count_distinct("Sales")).show()

// 2 
df.select(avg("Sales")).show()

// 3 
df.select(collect_list("Sales")).show()

// 4 
df.select(collect_set("Sales")).show()

// 5
df.select(first("Sales")).show()

// 6 
df.select(kurtosis("Sales")).show()

// 7
df.select(last("Sales")).show()

// 8 
df.select(skewness("Sales")).show()

//9
df.select(bin("open")).show()

//10
df.groupBy("Company").agg(max("Sales")).show()

//11
df.groupBy("Company").agg(min("Sales")).show()

//12 
df.groupBy("Company").agg(avg("Sales")).show()

//13
df.groupBy($"Company").min().show

//14
df.groupBy($"Company").max().show

//15
df.groupBy($"Company").avg().show
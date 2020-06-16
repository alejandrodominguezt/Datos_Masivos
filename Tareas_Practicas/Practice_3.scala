// Practice 3

// Method 1

  def fib1(n: Int): Int = n match {
    case 0 | 1 => n
    case _ => fib1(n - 1) + fib1(n - 2)
  }
  fib1(5)
  
  // Method 2

   def fib2(n: Int): Int = {
    
    var first = 0
    var second = 1
    var count = 0
    
    while(count < n){
      val sum = first + second
      first = second
      second = sum
      count = count + 1
    }
    
    return first
  }
  fib2(18)
  
  // Method 3
 
  def fib3(n: Int): Int = {
    def fib_tail(n: Int, a: Int, b: Int): Int = n match {
      case 0 => a
      case _ => fib_tail(n - 1, b, a + b)
    }  
    return fib_tail(n, 0 , 1)
  }
  fib3(21)
  
    // Method 4

def fib4(num: Int): Double = {
    if(num < 2){
        return num; 
    }else{
        var i = ((1 + math.sqrt(5))/2);
        var j = ((math.pow(i,num)) - (math.pow((1-i),num)))/ math.sqrt(5);
        return j;
    }
}
fibFormula(5)

  // Method 5

  def fib5( n : Int) : Int = { 
    def fib_tail( n: Int, a:Int, b:Int): Int = n match {
      case 0 => a 
      case _ => fib_tail( n-1, b, (a+b)%1000000 )
    }
    return fib_tail( n%1500000, 0, 1)
  }
}

fib5(8)

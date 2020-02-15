// Practice 3

// Method 1

  def fib1(n: Int): Int = n match {
    case 0 | 1 => n
    case _ => fib1(n - 1) + fib1(n - 2)
  }
  fib1(5)
  
  // Method 2

 
 

 

  
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

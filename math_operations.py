"""
Python Math Operations and Calculations Module
===============================================
Comprehensive module demonstrating basic mathematical operations,
calculations, and problem-solving using Python.

Author: Math Operations Module
Version: 1.0
"""

import math


def basic_arithmetic():
    """
    Demonstrate basic arithmetic operations.
    
    Returns:
    --------
    dict
        Dictionary containing results of arithmetic operations
    """
    print("=" * 70)
    print("BASIC ARITHMETIC OPERATIONS")
    print("=" * 70)
    
    a = 15
    b = 4
    
    # Addition
    addition = a + b
    print("\nAddition: {} + {} = {}".format(a, b, addition))
    
    # Subtraction
    subtraction = a - b
    print("Subtraction: {} - {} = {}".format(a, b, subtraction))
    
    # Multiplication
    multiplication = a * b
    print("Multiplication: {} * {} = {}".format(a, b, multiplication))
    
    # Division
    division = a / b
    print("Division: {} / {} = {:.2f}".format(a, b, division))
    
    # Floor Division
    floor_division = a // b
    print("Floor Division: {} // {} = {}".format(a, b, floor_division))
    
    # Modulo (Remainder)
    modulo = a % b
    print("Modulo: {} % {} = {}".format(a, b, modulo))
    
    # Exponentiation
    exponentiation = a ** b
    print("Exponentiation: {} ** {} = {}".format(a, b, exponentiation))
    
    return {
        "addition": addition,
        "subtraction": subtraction,
        "multiplication": multiplication,
        "division": division,
        "floor_division": floor_division,
        "modulo": modulo,
        "exponentiation": exponentiation
    }


def calculate_statistics(numbers):
    """
    Calculate statistical measures for a list of numbers.
    
    Parameters:
    -----------
    numbers : list
        List of numerical values
    
    Returns:
    --------
    dict
        Dictionary containing statistical measures
    """
    print("\n" + "=" * 70)
    print("STATISTICAL CALCULATIONS")
    print("=" * 70)
    print("\nNumbers: {}".format(numbers))
    
    # Sum
    total = sum(numbers)
    print("\nSum: {}".format(total))
    
    # Count
    count = len(numbers)
    print("Count: {}".format(count))
    
    # Mean (Average)
    mean = total / count if count > 0 else 0
    print("Mean (Average): {:.2f}".format(mean))
    
    # Minimum and Maximum
    minimum = min(numbers)
    maximum = max(numbers)
    print("Minimum: {}".format(minimum))
    print("Maximum: {}".format(maximum))
    
    # Range
    range_value = maximum - minimum
    print("Range: {}".format(range_value))
    
    # Variance
    variance = sum((x - mean) ** 2 for x in numbers) / count if count > 0 else 0
    print("Variance: {:.2f}".format(variance))
    
    # Standard Deviation
    std_dev = math.sqrt(variance)
    print("Standard Deviation: {:.2f}".format(std_dev))
    
    return {
        "sum": total,
        "count": count,
        "mean": mean,
        "min": minimum,
        "max": maximum,
        "range": range_value,
        "variance": variance,
        "std_dev": std_dev
    }


def calculate_geometry():
    """
    Calculate geometric properties.
    
    Returns:
    --------
    dict
        Dictionary containing geometric calculations
    """
    print("\n" + "=" * 70)
    print("GEOMETRIC CALCULATIONS")
    print("=" * 70)
    
    # Circle calculations
    radius = 5
    circle_area = math.pi * radius ** 2
    circle_circumference = 2 * math.pi * radius
    print("\nCircle (radius = {}):".format(radius))
    print("  Area: {:.2f}".format(circle_area))
    print("  Circumference: {:.2f}".format(circle_circumference))
    
    # Rectangle calculations
    length = 10
    width = 6
    rect_area = length * width
    rect_perimeter = 2 * (length + width)
    print("\nRectangle (length = {}, width = {}):".format(length, width))
    print("  Area: {}".format(rect_area))
    print("  Perimeter: {}".format(rect_perimeter))
    
    # Triangle calculations (using Heron's formula)
    side_a = 7
    side_b = 8
    side_c = 9
    semi_perimeter = (side_a + side_b + side_c) / 2
    tri_area = math.sqrt(
        semi_perimeter * (semi_perimeter - side_a) * 
        (semi_perimeter - side_b) * (semi_perimeter - side_c)
    )
    tri_perimeter = side_a + side_b + side_c
    print("\nTriangle (sides = {}, {}, {}):".format(side_a, side_b, side_c))
    print("  Area (Heron's formula): {:.2f}".format(tri_area))
    print("  Perimeter: {}".format(tri_perimeter))
    
    # Sphere calculations
    sphere_radius = 3
    sphere_volume = (4/3) * math.pi * sphere_radius ** 3
    sphere_surface = 4 * math.pi * sphere_radius ** 2
    print("\nSphere (radius = {}):".format(sphere_radius))
    print("  Volume: {:.2f}".format(sphere_volume))
    print("  Surface Area: {:.2f}".format(sphere_surface))
    
    return {
        "circle_area": circle_area,
        "circle_circumference": circle_circumference,
        "rectangle_area": rect_area,
        "rectangle_perimeter": rect_perimeter,
        "triangle_area": tri_area,
        "triangle_perimeter": tri_perimeter,
        "sphere_volume": sphere_volume,
        "sphere_surface": sphere_surface
    }


def solve_quadratic_equation(a, b, c):
    """
    Solve a quadratic equation: ax^2 + bx + c = 0
    
    Parameters:
    -----------
    a : float
        Coefficient of x^2
    b : float
        Coefficient of x
    c : float
        Constant term
    
    Returns:
    --------
    dict
        Dictionary containing roots and discriminant
    """
    print("\n" + "=" * 70)
    print("QUADRATIC EQUATION SOLVER")
    print("=" * 70)
    print("\nEquation: {}x^2 + {}x + {} = 0".format(a, b, c))
    
    # Calculate discriminant
    discriminant = b ** 2 - 4 * a * c
    print("Discriminant: {}".format(discriminant))
    
    if discriminant < 0:
        print("No real solutions (discriminant is negative)")
        return {"discriminant": discriminant, "roots": None}
    
    elif discriminant == 0:
        root = -b / (2 * a)
        print("One solution: x = {:.2f}".format(root))
        return {"discriminant": discriminant, "roots": [root]}
    
    else:
        sqrt_discriminant = math.sqrt(discriminant)
        root1 = (-b + sqrt_discriminant) / (2 * a)
        root2 = (-b - sqrt_discriminant) / (2 * a)
        print("Two solutions:")
        print("  x1 = {:.2f}".format(root1))
        print("  x2 = {:.2f}".format(root2))
        return {"discriminant": discriminant, "roots": [root1, root2]}


def calculate_compound_interest(principal, rate, time, compounds_per_year=1):
    """
    Calculate compound interest.
    
    Formula: A = P(1 + r/n)^(nt)
    
    Parameters:
    -----------
    principal : float
        Initial amount (dollars)
    rate : float
        Annual interest rate (as decimal, e.g., 0.05 for 5%)
    time : float
        Time period in years
    compounds_per_year : int
        Number of times interest compounds per year
    
    Returns:
    --------
    dict
        Dictionary containing interest calculations
    """
    print("\n" + "=" * 70)
    print("COMPOUND INTEREST CALCULATOR")
    print("=" * 70)
    print("\nPrincipal: ${}".format(principal))
    print("Annual Rate: {}%".format(rate * 100))
    print("Time Period: {} years".format(time))
    print("Compounds per year: {}".format(compounds_per_year))
    
    # Calculate final amount
    final_amount = principal * (1 + rate / compounds_per_year) ** (compounds_per_year * time)
    
    # Calculate interest earned
    interest_earned = final_amount - principal
    
    print("\nFinal Amount: ${:.2f}".format(final_amount))
    print("Interest Earned: ${:.2f}".format(interest_earned))
    
    return {
        "principal": principal,
        "rate": rate,
        "time": time,
        "final_amount": final_amount,
        "interest_earned": interest_earned
    }


def calculate_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points.
    
    Formula: distance = sqrt((x2-x1)^2 + (y2-y1)^2)
    
    Parameters:
    -----------
    x1, y1 : float
        Coordinates of first point
    x2, y2 : float
        Coordinates of second point
    
    Returns:
    --------
    float
        Euclidean distance
    """
    print("\n" + "=" * 70)
    print("DISTANCE CALCULATION")
    print("=" * 70)
    print("\nPoint 1: ({}, {})".format(x1, y1))
    print("Point 2: ({}, {})".format(x2, y2))
    
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print("Distance: {:.2f}".format(distance))
    
    return distance


def calculate_percentage_change(original, new):
    """
    Calculate percentage change between two values.
    
    Formula: percent_change = ((new - original) / original) * 100
    
    Parameters:
    -----------
    original : float
        Original value
    new : float
        New value
    
    Returns:
    --------
    float
        Percentage change
    """
    print("\n" + "=" * 70)
    print("PERCENTAGE CHANGE CALCULATION")
    print("=" * 70)
    print("\nOriginal Value: {}".format(original))
    print("New Value: {}".format(new))
    
    if original == 0:
        print("Error: Original value cannot be zero")
        return None
    
    percent_change = ((new - original) / original) * 100
    print("Percentage Change: {:.2f}%".format(percent_change))
    
    if percent_change > 0:
        print("Status: Increase")
    elif percent_change < 0:
        print("Status: Decrease")
    else:
        print("Status: No change")
    
    return percent_change


def calculate_factorial(n):
    """
    Calculate factorial of a number.
    
    Formula: n! = n * (n-1) * (n-2) * ... * 1
    
    Parameters:
    -----------
    n : int
        Non-negative integer
    
    Returns:
    --------
    int
        Factorial of n
    """
    print("\n" + "=" * 70)
    print("FACTORIAL CALCULATION")
    print("=" * 70)
    print("\nCalculating factorial of {}".format(n))
    
    if n < 0:
        print("Error: Factorial undefined for negative numbers")
        return None
    
    if n == 0 or n == 1:
        result = 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
    
    print("{}! = {}".format(n, result))
    return result


def calculate_fibonacci(n):
    """
    Generate Fibonacci sequence up to n terms.
    
    Parameters:
    -----------
    n : int
        Number of Fibonacci terms to generate
    
    Returns:
    --------
    list
        List of Fibonacci numbers
    """
    print("\n" + "=" * 70)
    print("FIBONACCI SEQUENCE")
    print("=" * 70)
    print("\nGenerating first {} Fibonacci numbers".format(n))
    
    if n <= 0:
        return []
    
    fibonacci_sequence = []
    a, b = 0, 1
    
    for _ in range(n):
        fibonacci_sequence.append(a)
        a, b = b, a + b
    
    print("Sequence: {}".format(fibonacci_sequence))
    return fibonacci_sequence


def find_prime_numbers(start, end):
    """
    Find all prime numbers in a given range.
    
    Parameters:
    -----------
    start : int
        Start of range
    end : int
        End of range
    
    Returns:
    --------
    list
        List of prime numbers
    """
    print("\n" + "=" * 70)
    print("PRIME NUMBER FINDER")
    print("=" * 70)
    print("\nFinding primes between {} and {}".format(start, end))
    
    primes = []
    
    for num in range(max(2, start), end + 1):
        is_prime = True
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    
    print("Prime numbers: {}".format(primes))
    print("Total: {}".format(len(primes)))
    return primes


def calculate_gcd(a, b):
    """
    Calculate Greatest Common Divisor using Euclidean algorithm.
    
    Parameters:
    -----------
    a, b : int
        Two integers
    
    Returns:
    --------
    int
        Greatest Common Divisor
    """
    print("\n" + "=" * 70)
    print("GREATEST COMMON DIVISOR (GCD)")
    print("=" * 70)
    print("\nFinding GCD of {} and {}".format(a, b))
    
    while b != 0:
        a, b = b, a % b
    
    gcd_result = a
    print("GCD: {}".format(gcd_result))
    return gcd_result


def main():
    """
    Main function demonstrating all mathematical operations.
    """
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  COMPREHENSIVE PYTHON MATH OPERATIONS DEMONSTRATION".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # 1. Basic Arithmetic
    basic_results = basic_arithmetic()
    
    # 2. Statistical Calculations
    test_numbers = [10, 25, 30, 45, 50, 65, 75, 85, 90, 100]
    stats_results = calculate_statistics(test_numbers)
    
    # 3. Geometric Calculations
    geometry_results = calculate_geometry()
    
    # 4. Quadratic Equation Solver
    solve_quadratic_equation(1, -5, 6)  # x^2 - 5x + 6 = 0
    
    # 5. Compound Interest
    compound_results = calculate_compound_interest(1000, 0.05, 5, 12)
    
    # 6. Distance Calculation
    distance = calculate_distance(0, 0, 3, 4)
    
    # 7. Percentage Change
    percent_change = calculate_percentage_change(100, 150)
    
    # 8. Factorial
    factorial = calculate_factorial(5)
    
    # 9. Fibonacci Sequence
    fibonacci = calculate_fibonacci(10)
    
    # 10. Prime Numbers
    primes = find_prime_numbers(1, 50)
    
    # 11. GCD
    gcd = calculate_gcd(48, 18)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nAll mathematical operations completed successfully!")
    print("\nOperations performed:")
    print("  1. Basic Arithmetic (7 operations)")
    print("  2. Statistics (8 measures)")
    print("  3. Geometry (8 calculations)")
    print("  4. Quadratic Equation Solver")
    print("  5. Compound Interest Calculator")
    print("  6. Euclidean Distance")
    print("  7. Percentage Change")
    print("  8. Factorial")
    print("  9. Fibonacci Sequence")
    print("  10. Prime Number Finder")
    print("  11. Greatest Common Divisor")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

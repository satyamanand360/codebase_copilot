class MathOperations:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return self.add(a, 0) * b # Just a fake dependency

def main_calculation():
    math_ops = MathOperations()
    result = math_ops.multiply(5, 10)
    print("Result:", result)
    return result

if __name__ == "__main__":
    main_calculation()

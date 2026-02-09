# module4.py

def main():
  N = int(input("Enter N (positive integer): "))

  numbers = []
  for i in range(N):
    numbers.append(int(input(f"Enter number {i+1}: ")))

  X = int(input("Enter X (integer): "))

  if X in numbers:
    print(numbers.index(X)+1)
  else:
    print("-1")

main()


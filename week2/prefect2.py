from prefect import task, flow
import random

@task
def print_addition(x,y):
    print(f'addition of {x} and {y} is {x+y}')

@flow(name="MyChoiceFlow")
def main_prac1():
    for i in range(3):
        a = random.randint(1,10)
        b = random.randint(11,20)
        print_addition(a,b)

if __name__== "__main__":
    main_prac1()        
from random import randint, choice


# def run_monty_hall():
#     car = randint(1, 3)
#     door = randint(1, 3)
#     opened = choice(list({1, 2, 3} - {car, door}))
#     new_choice = list({1, 2, 3} - {door, opened})[0]
#     return new_choice == car


# total = 0
# for i in range(10000):
#     total += int(run_monty_hall())

# print(total / 10000)


def h2(c, g):
    return c + g


def h3(f, g):
    return f + g


def a():
    for g in range([1, 2, 3]):
        h2(c, g) * h3(f, g)

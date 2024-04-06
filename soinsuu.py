year = 2024
prime = []
for i in range(2, year//2):
    if i == 2:
        prime.append(i)
    flag = 0
    for j in range(2, i):
        if (i % j) == 0:
            flag = 1
            break
    if flag == 0:
        prime.append(i)

form = {}
for num in prime:
    form[num] = 0
while year != 1:
    for num in prime:
        if year % num == 0:
            form[num] = form[num] + 1
            year = year // num
for beki in form:
    if form[beki] != 0:
        print("(%d^%d)"%(beki, form[beki]), end="")

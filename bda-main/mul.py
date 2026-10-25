from collections import defaultdict

ra, ca = int(input("Enter rows of Matrix A: ")), int(input("Enter columns of Matrix A: "))
print("Enter Matrix A:")
A = [list(map(int, input().split())) for _ in range(ra)]
rb, cb = int(input("Enter rows of Matrix B: ")), int(input("Enter columns of Matrix B: "))
if ca != rb:
    raise SystemExit("Matrix multiplication not possible.")
print("Enter Matrix B:")
B = [list(map(int, input().split())) for _ in range(rb)]

groups = defaultdict(lambda: ([0] * ca, [0] * ca))
for i in range(ra):
    for k in range(ca):
        for j in range(cb):
            groups[i, j][0][k] = A[i][k]
for k in range(rb):
    for j in range(cb):
        for i in range(ra):
            groups[i, j][1][k] = B[k][j]

print("Result Matrix:")
for i in range(ra):
    print(*(sum(x * y for x, y in zip(*groups[i, j])) for j in range(cb)), sep="\t")
print("Matrix multiplication completed!")
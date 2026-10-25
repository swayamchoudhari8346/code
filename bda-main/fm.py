import hashlib

data = list(map(int, input("Enter numbers: ").split()))
R = 0

for n in set(data):
    h = int(hashlib.md5(str(n).encode()).hexdigest(), 16)
    b = bin(h)[2:]
    z = len(b) - len(b.rstrip('0'))
    R = max(R, z)

print("Dataset:", data)
print("Exact distinct count:", len(set(data)))
print("Maximum trailing zeros (R):", R)
print("Estimated distinct count:", round(2**R / 0.77351))
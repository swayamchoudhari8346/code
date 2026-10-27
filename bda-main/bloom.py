import hashlib

class BloomFilter:
    def __init__(self, size=100, k=3):
        self.size, self.k = size, k
        self.bits = [0] * size

    def hashes(self, n):
        return [int(hashlib.sha256(f"{n}{i}".encode()).hexdigest(), 16) % self.size for i in range(self.k)]

    def add(self, n):
        for i in self.hashes(n):
            self.bits[i] = 1

    def check(self, n):
        return all(self.bits[i] for i in self.hashes(n))


bloom = BloomFilter()
numbers = [10, 20, 30, 40, 50, 60]

for n in numbers:
    bloom.add(n)

print("Hash Positions:")
for n in numbers:
    print(n, "->", bloom.hashes(n))

print("Index:", list(range(100)))
print("Bit Array:", bloom.bits)

search = int(input("Enter number to search: "))
print("Hash Positions:", bloom.hashes(search))
print("Result:", "Possibly Present" if bloom.check(search) else "Definitely Not Present")

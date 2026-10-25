class DGIM:
    def __init__(self):
        self.t, self.b = 0, []

    def update(self, bit):
        self.t += 1
        if bit:
            self.b.insert(0, [self.t, 1])

        while True:
            for s in sorted(set(x[1] for x in self.b)):
                i = [j for j, x in enumerate(self.b) if x[1] == s]
                if len(i) > 2:
                    self.b[i[-2]][1] *= 2
                    self.b.pop(i[-1])
                    break
            else:
                return

    def estimate(self):
        return sum(x[1] for x in self.b[:-1]) + (self.b[-1][1] // 2 if self.b else 0)


stream = list(map(int, input("Enter binary stream: ").split()))
d = DGIM()

for bit in stream:
    d.update(bit)
    print("Bit:", bit, "| Buckets:", [x[1] for x in d.b],
          "| Estimate:", d.estimate())

print("Final buckets:", [x[1] for x in d.b])
print("Estimated count:", d.estimate())
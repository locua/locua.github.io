import os

R = []
for i in range(100):
  r = os.urandom(100)
  R.append(r)
  print(r)

for r in R:
  print(r.decode('latin-1'))
  


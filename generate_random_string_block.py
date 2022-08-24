import string
import random

out = ""
for i in range(20*100):
  if(i%100==0):
    out+="\n"
  else:
    out+=random.choice(string.ascii_letters)
print(out)
    

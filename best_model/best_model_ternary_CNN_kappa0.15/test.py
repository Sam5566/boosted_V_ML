import os

#s.system("sed -i -r 's/.*(.{150})/\1/g' test.log"+" && sed -i -r 's/\r//g' test.log")
os.system("cat test.log | tr -d '\b\r' > test2.log")

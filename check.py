
# Python code to
# demonstrate readlines()
  




# Using readlines()
file1 = open('requirements.txt', 'r')
Lines = file1.readlines()
print(Lines)

# # writing to file
# file1 = open('requirements.txt', 'w')
# file1.writelines(L)
# file1.close()
  

  
# count = 0
# # Strips the newline character
# for line in Lines:
#     count += 1
#     print("Line{}: {}".format(count, line.strip())
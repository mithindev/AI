name = "Mithin Dev"

# CONTEXT MANAGER
#   |-> closes automatically!
# with open("test.txt", "w") as f:
#     f.write(name)

fp = open("demo.txt", "w")
fp.write(name)
fp.close()


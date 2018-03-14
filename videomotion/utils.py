def out(msg=""):
    print(str(msg))
    with open("output.txt", "a") as dump_file:
        dump_file.write(str(msg) + "\n")


def err(msg):
    print("ERROR: " + str(msg))


def warn(msg):
    print("WARNING: " + str(msg))

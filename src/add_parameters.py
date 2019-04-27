import sys

input_file = sys.argv[-1]
my_lines = []
with open(input_file, "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        my_lines.append(lines[i])
        if("<edge" in lines[i] and "from=" in lines[i]):
            my_params = lines[i].split("<edge")[0]
            junction = lines[i].split("from=")[1].split("\"")[1]
            my_params+="\t<param key=\"from\" value=\"{}\"/>\n".format(junction)
            my_lines.append(my_params)
with open(input_file,'w') as f:
    for line in my_lines:
        f.write(line)

import json

def load_parameters(name):
    with open(name, "r") as file:
        pars = json.load(file)

    return pars["intensity"], pars["angle"], pars["length"]

if __name__ == "__main__":
    ints, angles, lengths = load_parameters("old_data/parameters.json")
    with open("old_data/params.txt","w")  as file:
        for i in ints:
            for a in angles:
                for l in lengths:
                    file.write("{},{},{}\n".format(i, l, a))
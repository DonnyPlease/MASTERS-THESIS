from collections import UserDict


class DatasetRecord():
    def __init__(self, text_line = ''):
        self.I = 0
        self.L = 0
        self.alpha = 0
        self.t_hot = 0
        self.min_energy = 0
        self.max_energy = 0
        self.type = ''
        self.a = 0
        self.b = '0'
        self.c = '0'
        self.d = '0'
        self.e = '0'
        self.f = '0'
        self.g = '0'
        self.a_stdev = '0'
        self.b_stdev = '0'
        self.c_stdev = '0'
        self.d_stdev = '0'
        self.e_stdev = '0'
        self.f_stdev = '0'
        self.g_stdev = '0'
        self.t_hot_stdev = '0'
        if text_line != '':
            self.initialize_from_text(text_line)
            

    def make_key(self):
        return (self.I, "{:.2f}".format(float(self.L)), self.alpha)
    
    def __hash__(self):
        return hash((self.I, self.L, self.alpha))
    
    def __eq__(self, other):
        # Check for equality based on the three keys
        return (
            isinstance(other, DatasetRecord) and
            self.key1 == other.key1 and
            self.key2 == other.key2 and
            self.key3 == other.key3
        )
    
    def initialize_from_text(self, text_line):
        params = text_line.strip('\n').split(',')
        self.I = params[0]
        self.L = params[1]
        self.alpha =  params[2]
        self.t_hot = params[3]
        self.min_energy = params[4]
        self.max_energy = params[5]
        self.type =  params[6]
        self.a = params[7]
        self.b = params[8]
        self.c = params[9]
        self.d = params[10]
        self.e = params[11]
        self.f = params[12]
        self.g = params[13]
        self.a_stdev = params[14]
        self.b_stdev = params[15]
        self.c_stdev = params[16]
        self.d_stdev = params[17]
        self.e_stdev = params[18]
        self.f_stdev = params[19]
        self.g_stdev = params[20]
        self.t_hot_stdev = params[21]
         

    def to_text(self):
        params = [self.I, 
                  "{:.2f}".format(float(self.L)),
                  self.alpha,
                  self.t_hot,
                  self.min_energy,
                  self.max_energy,
                  self.type,
                  self.a,
                  self.b,
                  self.c,
                  self.d,
                  self.e,
                  self.f,
                  self.g,
                  self.a_stdev,
                  self.b_stdev,
                  self.c_stdev,
                  self.d_stdev,
                  self.e_stdev,
                  self.f_stdev,
                  self.g_stdev,
                  self.t_hot_stdev]
        line = ''
        for p in params:
            line += p + ','
        line = line[:-1] + '\n'
        return line


class DatasetUtils():
    def load_datasets_to_dicts(folder):
        final_dataset = {}
        with open(folder+'/'+'final_dataset.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                one_data = DatasetRecord(line)
                final_dataset[one_data.make_key()] = one_data
                
        autofit_dataset = {}
        with open(folder+'/'+'auto_fit.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                one_data = DatasetRecord(line)
                autofit_dataset[one_data.make_key()] = one_data
                # print(one_data.make_key())
        return final_dataset, autofit_dataset
    
    def overwrite_final(folder, final_dataset):
        with open(folder+'/'+'final_dataset.txt', 'w') as f:
            for key, value in final_dataset.items():
                f.write(value.to_text())
    
    def dataset_to_dict(dataset):
        data = {"1e19": [], "1e18": [], "1e17": [], "5e17": [], "5e18": []}
        for key, item in dataset.items():
            i, l, alpha = key
            l = float(l)
            alpha = float(alpha)
            data[i].append((l, alpha, item.t_hot, item.t_hot_stdev))
        return data 
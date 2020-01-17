from preprocess import *
import os



file = './data/processed/batch4.data'
if os.path.exists(file):

    for b_idx in range(1, 5):

        bin_file = './data/processed/batch%d.data' % b_idx
        batch = 'b%d' % b_idx
        data_path = r'./data/raw/batch%d' % b_idx
        D = data(bin_file)
        print('Process batch : %d' % b_idx)

        if os.path.exists(D.bin_file):
            with open(D.bin_file, 'rb') as f:
                D = pickle.load(f)
                D.bin_file = bin_file
        else:
            D.read_data(batch, data_path)
            D.save_data()
            D.produce_minidata()
            del D

bin_file = './data/processed/total.data_Mini'
D = data(bin_file)
if not os.path.exists(D.bin_file):
    D1 = data('./data/processed/batch1.data_Mini')
    D2 = data('./data/processed/batch2.data_Mini')
    D3 = data('./data/processed/batch3.data_Mini')
    D4 = data('./data/processed/batch4.data_Mini')

    with open(D1.bin_file, 'rb') as f:
        D1 = pickle.load(f)
    for cap in D1.caps:
        D.caps.append(cap)
    del D1
    with open(D2.bin_file, 'rb') as f:
        D2 = pickle.load(f)
    for cap in D2.caps:
        D.caps.append(cap)
    del D2
    with open(D3.bin_file, 'rb') as f:
        D3 = pickle.load(f)
    for cap in D3.caps:
        D.caps.append(cap)
    del D3
    with open(D4.bin_file, 'rb') as f:
        D4 = pickle.load(f)
    for cap in D4.caps:
        D.caps.append(cap)
    del D4
    D.save_data()

else:
    with open(D.bin_file, 'rb') as f:
        D = pickle.load(f)




# The index we use
train_index, test_index = index_split(D.caps)

# The methods to get data or feature
# cap.get_data_from_cycle('cycle', 'discharge_capacitance(F)', 657)
# cap.V_drop_with_cycle(657,2)



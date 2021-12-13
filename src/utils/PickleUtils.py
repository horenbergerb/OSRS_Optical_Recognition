import pickle


def LoadPickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def SavePickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

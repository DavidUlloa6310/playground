import torch


CONTEXT_LEN = 8
BATCH_SIZE = 4

def translate(c2i, i2c):

    def encode(input):
        result = []
        for item in input:
            result.append(c2i[item])
        return result

    def decode(input):
        result = []
        for item in input:
            result.append(i2c[item])
        return ''.join(result)

    return encode, decode 


def read_data(filename):
    text = None
    with open(filename, 'r') as f:
        text = f.read()

    unique_chars = sorted(list(set(text)))
    c2i, i2c = {}, {} 
    for index, char in enumerate(unique_chars):
        c2i[char] = index
        i2c[index] = char

    encode, decode = translate(c2i, i2c)
    data = torch.tensor(encode(text))
    split = int(0.9*len(data))
    train = data[:split]
    val = data[split:]

    def get_batch(type):
        data = train if type == "train" else val
        indices = torch.randint(high = len(data) - CONTEXT_LEN, size = (BATCH_SIZE,))
        x = torch.stack([data[i:i+CONTEXT_LEN] for i in indices])
        y = torch.stack([data[i+1:i+CONTEXT_LEN+1] for i in indices])
        return x, y
    
    return get_batch, encode, decode


def main():
    get_batch = read_data('input.txt')
    x, y = get_batch('train')
    print('x values')
    print(x)
    print('y values')
    print(y)


if __name__ == '__main__':
    main()

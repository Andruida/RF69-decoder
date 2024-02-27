from bitarray import bitarray
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import kodolo
#import json

sampling_rate = 240000
bit_rate = 2_400
preamble_length = 4
sync_word = 0x2dd4
min_correlation = 7*0.8
whitening = True

lfsr_bits = 9
bit_length = sampling_rate//bit_rate
print("bit_length:", bit_length)
packet_preamble1 = np.repeat([0, 1]*4*preamble_length, bit_length)
packet_preamble2 = np.repeat([1, 0]*4*preamble_length, bit_length)

sync_word_bits = [sync_word >> i & 1 for i in range(15, -1, -1)]
#print(sync_word_bits)
packet_sync_word = np.repeat(sync_word_bits, bit_length)
packet_start1 = np.concatenate([packet_preamble1, packet_sync_word]) * 2 - 1
packet_start2 = np.concatenate([packet_preamble2, packet_sync_word]) * 2 - 1

#packet_start1 *= 1
#packet_start2 *= 1

def gen_lfsr_sequence():
    state = 0
    n = 0
    while True:
        # if n > lfsr_bits-1:
        yield state & 1
        inp = (state & 1) ^ ((state & (1 << 5)) >> 5) ^ 1
        state = (state >> 1) | (inp << lfsr_bits-1)
        n += 1

def convert_to_dsignal(data: np.ndarray, clusters):
    one = min(range(len(clusters)), key=lambda i: clusters[i])
    zero = max(range(len(clusters)), key=lambda i: clusters[i])
    dsig = data.copy()
    dsig[data == one] = 1
    dsig[data == zero] = -1
    dsig[(data != one) & (data != zero)] = 0
    # print("one:", one, "zero:", zero)
    
    return dsig
    
    

def convert_to_bits(dsignal):
    bits = np.zeros(int(np.ceil(len(dsignal)/bit_length)), dtype=np.uint8)
    for i in range(len(bits)):
        begin = i*bit_length
        end = i*bit_length + bit_length
        bitsum = np.sum(dsignal[begin:end])
        if bitsum > 0:
            bits[i] = 1
        else:
            bits[i] = 0

    return bits

def convert_bits_to_int(bits):
    data = np.zeros(int(np.ceil(len(bits)/8)), dtype=np.uint8)
    for i in range(len(bits)):
        data_i = i // 8
        bit_shift = 7 - (i % 8)
        data[data_i] += bits[i] << bit_shift
    return data

def descramble(bits, lfsr=None):
    bits_out = np.zeros(len(bits), dtype=np.uint8)
    if lfsr == None:
        lfsr = gen_lfsr_sequence()
    for i in range(len(bits)):
        bits_out[i] = bits[i] ^ next(lfsr)

    return bits_out

def calculate_error(data_decoded, data_encoded):
    if len(data_decoded) != 18:
        return -1
    bits_in = bitarray(endian="little")
    bits_in.frombytes(data_decoded)
    bits_with_error = bitarray(endian="little")
    bits_with_error.frombytes(data_encoded)
    bits_encoded = kodolo.encode(bits_in)
    return (bits_with_error ^ bits_encoded).count()
    

while True:
    raw_bytes = sys.stdin.buffer.read(sampling_rate)
    if len(raw_bytes) == 0:
        break
    data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int16) - 128

    # find packet start
    packets = []
    for packet_start in [packet_start1, packet_start2]:
        corr = signal.correlate(data, packet_start, mode="same") / len(packet_start)
        best_above_limit = -1
        # plt.plot(data)
        # plt.plot(corr)
        # plt.show()
        # print("max corr: ",round(np.max(corr), 2))
        for i in range(len(corr)):
            if corr[i] > min_correlation:
                if best_above_limit == -1:
                    best_above_limit = i
                if best_above_limit + len(packet_start)//2 < i:
                    best_above_limit = i
                if corr[i] > corr[best_above_limit]:
                    best_above_limit = i
            else:
                if best_above_limit != -1:
                    packets.append(best_above_limit - len(packet_start)//2)
                best_above_limit = -1

    #print(packets)
    for begin in packets:
        end_preamble = begin + len(packet_start1)
        end_length_byte = end_preamble + 8*bit_length
        if end_length_byte > len(data) or end_preamble > len(data) or len(data[begin:end_preamble]) == 0:
            print("Packet got cropped")
            newdata = np.frombuffer(sys.stdin.buffer.read(end_length_byte - len(data)), dtype=np.uint8).astype(np.int16) - 128
            data = np.concatenate((data, newdata))
            del newdata
        lfsr = gen_lfsr_sequence()
        kmeans = KMeans(n_clusters=3, random_state=0).fit(data[begin:end_preamble].reshape(-1, 1))
        #inverted = (kmeans.cluster_centers_[0][0] < kmeans.cluster_centers_[1][0])
        # plt.plot(data[begin:end_length_byte])
        # plt.plot(convert_to_dsignal(kmeans.predict(data[begin:end_length_byte].reshape(-1, 1)), kmeans.cluster_centers_))
        # plt.show()
        #print(inverted)
        # print(kmeans.cluster_centers_)
        length_predict = kmeans.predict(data[end_preamble:end_length_byte].reshape(-1, 1))
        length_dsig = convert_to_dsignal(length_predict, kmeans.cluster_centers_)
        length_bits = convert_to_bits(length_dsig)
        length_bits = descramble(length_bits, lfsr)
        packet_length = convert_bits_to_int(length_bits)[0]
        if packet_length == 0:
            print("Packet length is 0\n")
            continue
        print("Packet length:", packet_length)

        end_data = end_length_byte + packet_length*8*bit_length
        if end_data > len(data):
            print("Packet got cropped")
            newdata = np.frombuffer(sys.stdin.buffer.read(end_data - len(data)), dtype=np.uint8).astype(np.int16) - 128
            data = np.concatenate((data, newdata))
            del newdata
        data_predict = kmeans.predict(data[end_length_byte:end_data].reshape(-1, 1))
        data_dsig = convert_to_dsignal(data_predict, kmeans.cluster_centers_)
        data_bits = convert_to_bits(data_dsig)
        data_bits = descramble(data_bits, lfsr)
        data_bytes = convert_bits_to_int(data_bits)
        
        
        print(":".join([hex(b)[2:] if len(hex(b)[2:]) == 2 else "0"+hex(b)[2:] for b in data_bytes]))
        #print("".join([chr(b) for b in data_bytes[4:]]))
        
        
        if packet_length == 32:
            print("Hibajavított:")
            data_decoded = kodolo.decode_bytes(data_bytes.tobytes())
            print(":".join([hex(b)[2:] if len(hex(b)[2:]) == 2 else "0"+hex(b)[2:] for b in data_decoded]))
            print("".join([chr(b) for b in data_decoded]))
            print("Hibás bitek: ", calculate_error(data_decoded, data_bytes.tobytes()))
        else:
            print("".join([chr(b) for b in data_bytes]))
            
        print()




    #plt.savefig("fig.png")


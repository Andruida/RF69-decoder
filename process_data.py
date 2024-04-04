from bitarray import bitarray
import crc
import kodolo
from sklearn.utils.murmurhash import murmurhash3_32
import numpy as np
import os
import sys
import time
# import pandas as pd

RUNID = sys.argv[1]
if len(sys.argv) > 2:
    static_data = sys.argv[2] == "static"
else:
    static_data = False
    
if len(sys.argv) > 3:
    static_serial = sys.argv[3] == "static"
else:
    static_serial = False
write_to_file = True

crc_config = crc.Configuration(
    width=32,
    polynomial=0x4C11DB7,
)
internal_serial = 0

def gen_murmur_sequence(seed = 42):
    outData = bytes()
    for _ in range(31):
        # seedData = bytes([(seed >> (3-i)*8) & 0xFF for i in range(4)])
        seedData = seed.to_bytes(4, "big")
        r = murmurhash3_32(seedData, seed, positive=True)
        seed = r
        # outData += bytes([(r >> (3-i)*8) & 0xFF for i in range(4)])        
        outData += r.to_bytes(4, "big")
    return outData

previous_serial = -1
def process_data(data_bytes: np.ndarray[np.uint8], data_bits: np.ndarray[np.uint8], detected_packet_length: int):
    global previous_serial
    global internal_serial
    packet_length = len(data_bytes)
    if packet_length % 32 == 0 and packet_length == 7*32:
        print("Hibajavított:")
        data_decoded = bytes()
        for i in range(packet_length//32):
            data_decoded += kodolo.decode_bytes(data_bytes[i*32:(i+1)*32].tobytes())
        # print(":".join([hex(b)[2:] if len(hex(b)[2:]) == 2 else "0"+hex(b)[2:] for b in data_decoded]))
        #print("".join([chr(b) for b in data_decoded[:-8]]))
        # print("Hibás bitek: ", calculate_error(data_decoded, data_bytes.tobytes()).count())
        crc_calc = crc.Calculator(crc_config)
        chksum_calc = crc_calc.checksum(data_decoded[:-4])
        chksum = int.from_bytes(data_decoded[-4:], "big")
        if static_serial:
            serial = 42
            read_serial = int.from_bytes(data_decoded[-8:-4], "big")
            if serial != read_serial:
                print("Serial mismatch, read:", read_serial, "expected:", serial)
        else:
            serial = int.from_bytes(data_decoded[-8:-4], "big")
        if static_data:
            reconstructed = gen_murmur_sequence(42)[:7*18-8] + serial.to_bytes(4, "big")
        else:
            reconstructed = gen_murmur_sequence(serial)[:7*18-8] + serial.to_bytes(4, "big")
        reconstructed_crc = crc_calc.checksum(reconstructed)
        reconstructed += reconstructed_crc.to_bytes(4, "big")
        reconstructed_encoded = bytes()
        for i in range(len(reconstructed)//18):
            reconstructed_encoded += kodolo.encode_bytes(reconstructed[i*18:(i+1)*18])
        # print(":".join([hex(b)[2:] if len(hex(b)[2:]) == 2 else "0"+hex(b)[2:] for b in reconstructed]))
        # print("Számolt ellenőrző összeg: ", chksum_calc)
        # print("Ellenőrző összeg: ", chksum)
        print("Sorszám: ", serial)
        print("Ellenőrző összeg egyezik: ", chksum == chksum_calc, chksum == reconstructed_crc)
        print("Várt és kapott egyezik: ", reconstructed == data_decoded, reconstructed_encoded == data_bytes.tobytes())
        
        # print(data_bits)
        incoming_bits = bitarray(list(data_bits))
        incoming_bits2 = bitarray()
        incoming_bits2.frombytes(data_bytes.tobytes())
        print("Sanity check: ",incoming_bits == incoming_bits2)
        
        decoded_bits = bitarray()
        decoded_bits.frombytes(data_decoded)
        
        reconstructed_incoming_bits = bitarray()
        reconstructed_incoming_bits.frombytes(reconstructed_encoded)
        
        reconstructed_bits = bitarray()
        reconstructed_bits.frombytes(reconstructed)
        
        incoming_bit_error = incoming_bits ^ reconstructed_incoming_bits
        decoded_bit_error = decoded_bits ^ reconstructed_bits
        
        
        timestamp = "" #time.time()
        
        print("Várt vs kapott (bitek): " , (incoming_bits ^ reconstructed_incoming_bits).count(), (decoded_bits ^ reconstructed_bits).count())
        
        if (serial - previous_serial) != 1:
            print("Serial jump: ", serial - previous_serial)    
        
        if write_to_file:
            if (os.path.exists(f"outdata_{RUNID}.csv") == False):
                with open(f"outdata_{RUNID}.csv", "w") as f:
                    print("timestamp,serial,detected_packet_length,incoming_bit_error,incoming_bit_error_count,decoded_bit_error,decoded_bit_error_count,crc_valid", file=f)
            
            with open(f"outdata_{RUNID}.csv", "a") as f:
                # if serial > previous_serial+1 and serial - previous_serial < 100:
                #     for i in range(previous_serial+1, serial):
                #         print(f"{timestamp},{i},,,,,,", file=f)
                if static_serial:
                    print("Belső sorszám: ", internal_serial)
                    print(f"{timestamp},{internal_serial},{detected_packet_length},{incoming_bit_error.to01()},{incoming_bit_error.count()},{decoded_bit_error.to01()},{decoded_bit_error.count()},{chksum == chksum_calc}", file=f)
                else:
                    print(f"{timestamp},{serial},{detected_packet_length},{incoming_bit_error.to01()},{incoming_bit_error.count()},{decoded_bit_error.to01()},{decoded_bit_error.count()},{chksum == chksum_calc}", file=f)
            
        previous_serial = serial
        internal_serial += 1
        

        # print(incoming_bits)
        
        
    else:
        print("Hibás csomag")
        #print("".join([chr(b) for b in data_bytes]))
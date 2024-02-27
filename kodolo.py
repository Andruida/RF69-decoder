from typing import Union
from bitarray import bitarray
from bitarray import util as bitarrayutil

epic_list = [
    0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240,
    1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241,
    2, 18, 34, 50, 66, 82, 98, 114, 130, 146, 162, 178, 194, 210, 226, 242,
    3, 19, 35, 51, 67, 83, 99, 115, 131, 147, 163, 179, 195, 211, 227, 243,
    4, 20, 36, 52, 68, 84, 100, 116, 132, 148, 164, 180, 196, 212, 228, 244,
    5, 21, 37, 53, 69, 85, 101, 117, 133, 149, 165, 181, 197, 213, 229, 245,
    6, 22, 38, 54, 70, 86, 102, 118, 134, 150, 166, 182, 198, 214, 230, 246,
    7, 23, 39, 55, 71, 87, 103, 119, 135, 151, 167, 183, 199, 215, 231, 247,
    8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 232, 248,
    9, 25, 41, 57, 73, 89, 105, 121, 137, 153, 169, 185, 201, 217, 233, 249,
    10, 26, 42, 58, 74, 90, 106, 122, 138, 154, 170, 186, 202, 218, 234, 250,
    11, 27, 43, 59, 75, 91, 107, 123, 139, 155, 171, 187, 203, 219, 235, 251,
    12, 28, 44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252,
    13, 29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253,
    14, 30, 46, 62, 78, 94, 110, 126, 142, 158, 174, 190, 206, 222, 238, 254,
    15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255]

kod_szavak_str = "0000000000101100101010100111100011000111100110010110000110100111001101010110011101001101010101100001110011111111"
kod_szavak = bitarray(kod_szavak_str[::-1], endian="little")
del kod_szavak_str

def kodolo_matrix(c_chunk: bitarray) -> bitarray:
    if not isinstance(c_chunk, bitarray) or len(c_chunk) != 4:
        raise ValueError("Input bitarray must be 4 bits long, got " + str(len(c_chunk)))
    
    kodolt = bitarrayutil.zeros(7, endian="little")
    for i in range(0, 4, 1):
        kodolt[i+3] = c_chunk[i]
    
    # e: 1110
    # f: 1101
    # g: 0111
    
    # e:
    kodolt[2] = c_chunk[3] ^ c_chunk[2] ^ c_chunk[1]
    
    # f:
    kodolt[1] = c_chunk[3] ^ c_chunk[2] ^ c_chunk[0]
    
    # g:
    kodolt[0] = c_chunk[2] ^ c_chunk[1] ^ c_chunk[0]
    
    return kodolt

def encode(bit_in : bitarray) -> bitarray:
    if not isinstance(bit_in, bitarray) or len(bit_in) != 144:
        raise ValueError("Input bitarray must be 144 bits long, got " + str(len(bit_in)))
    
    _bit_out = bitarrayutil.zeros(256, endian="little")
    for i in range(252, 256, 1):
        _bit_out[i] = 1
    
    current_chunk = bitarrayutil.zeros(4, endian="little")
    chunk_index: int = 0
    bit_out_index: int = 0
    
    for i in range(144):
        current_chunk[chunk_index] = bit_in[i]
        chunk_index += 1
        
        if chunk_index == 4:
            kodolt = kodolo_matrix(current_chunk)
            for j in range(7):
                _bit_out[bit_out_index] = kodolt[j]
                bit_out_index += 1
            chunk_index = 0
    
    bit_out = bitarrayutil.zeros(256, endian="little")
    for i in range(256):
        bit_out[i] = _bit_out[epic_list[i]]
        
    return bit_out

def kodolo_javitas(c_bit: bitarray) -> bitarray:
    if not isinstance(c_bit, bitarray) or len(c_bit) != 7:
        raise ValueError("Input bitarray must be 7 bits long, got " + str(len(c_bit)))
    
    diff: int = 8
    k_index: int = 0
    chunk_index: int = 0
    c_diff: int = 0
    
    for i in range(112):
        if c_bit[chunk_index] != kod_szavak[i]:
            c_diff += 1
        chunk_index += 1
        
        if chunk_index == 7:
            if c_diff == 0:
                return c_bit.copy()
            
            if c_diff < diff:
                diff = c_diff
                k_index = i - 6
            c_diff = 0
            chunk_index = 0
    
    res_bit = bitarrayutil.zeros(7, endian="little")
    index: int = 0
    for i in range(k_index, k_index + 7, 1):
        res_bit[index] = kod_szavak[i]
        index += 1
    
    return res_bit

def decode(bit_in: bitarray) -> bitarray:
    if not isinstance(bit_in, bitarray) or len(bit_in) != 256:
        raise ValueError("Input bitarray must be 256 bits long got " + str(len(bit_in)))
    
    un_bit = bitarrayutil.zeros(256, endian="little")
    
    for i in range(256):
        un_bit[i] = bit_in[epic_list[i]]
        
    res_bit = bitarrayutil.zeros(144, endian="little")
    chunk_index: int = 0
    res_bit_index: int = 0
    javitando = bitarrayutil.zeros(7, endian="little")
    javitott = bitarrayutil.zeros(7, endian="little")
    
    for i in range(252):
        javitando[chunk_index] = un_bit[i]
        chunk_index += 1
        
        if chunk_index == 7:
            javitott = kodolo_javitas(javitando)
            chunk_index = 0
            
            for j in range(3, 7):
                res_bit[res_bit_index] = javitott[j]
                res_bit_index += 1
            
    return res_bit

def encode_bytes(bytes_in: bytes) -> bytes:
    if not isinstance(bytes_in, bytes) or len(bytes_in) != 18:
        raise ValueError("Input bytes must be 18 bytes long, got " + str(len(bytes_in)))
    
    bits = bitarray(endian="little")
    bits.frombytes(bytes_in)
    return encode(bits).tobytes()

def decode_bytes(bytes_in: bytes) -> bytes:
    if not isinstance(bytes_in, bytes) or len(bytes_in) != 32:
        raise ValueError("Input bytes must be 32 bytes long, got " + str(len(bytes_in)))
    
    bits = bitarray(endian="little")
    bits.frombytes(bytes_in)
    return decode(bits).tobytes()
    

import numpy as np
import scipy.signal as signal
import sys

diff_threshold = 20
signal_count_threshold = 20
logic_change_threshold = 10
calibration_samples = 10
preamble_length = 4
sync_word = 0x2dd4
whitening = True
lfsr_bits = 9

def gen_lfsr_sequence():
    state = 0
    n = 0
    while True:
        # if n > lfsr_bits-1:
        yield state & 1
        inp = (state & 1) ^ ((state & (1 << 5)) >> 5) ^ 1
        state = (state >> 1) | (inp << lfsr_bits-1)
        n += 1

prev = 0
n = -1
def reset_vars():
    global zero_sum, zero_count, one_sum, one_count, count, current_bit_value, last_logic_change, timings, n, zero_value, one_value
    
    zero_sum = 0
    zero_count = 0
    zero_value = 0
    one_sum = 0
    one_count = 0
    one_value = 1
    count = 0
    timings = []
    
    current_bit_value = -1
    last_logic_change = n
    
def decode_timings(timings, zero_value):
    if zero_value == 0:
        timings = timings[2:]
    else:
        timings = timings[1:]
    if len(timings) < 4*8-4:
        return []
    bit_length = np.mean(timings[:4*8-4])
    print(file=sys.stderr)
    print("Bit length: ", bit_length, file=sys.stderr)
    #print([round(t/bit_length, 2) for t in timings], file=sys.stderr)
    timings_np = np.array(timings)
    bit_array = []
    t = 0
    i = 4
    current_bit_value = 0
    one_count = 0
    zero_count = 0
    while True:
        chunk = min(bit_length-t, timings_np[i])
        t += chunk
        timings_np[i] -= chunk
        if current_bit_value == 0:
            zero_count += chunk
        else:
            one_count += chunk
        
        if timings_np[i] == 0:
            i += 1
            if i == len(timings_np):
                bit_array.append(1 if one_count > zero_count else 0)
                break
            current_bit_value = 1 - current_bit_value
        if t >= bit_length:
            bit_array.append(1 if one_count > zero_count else 0)
            t = 0
            one_count = 0
            zero_count = 0
    
    data_begins = 0
    #print("".join([str(b) for b in bit_array]), file=sys.stderr)
    for i in range(0, len(bit_array)-(2*8)):
        word = 0
        for j in range(2*8):
            word = word << 1
            word += bit_array[i+j]
        if word == sync_word:
            print("Sync word found at: ", i, file=sys.stderr)
            data_begins = i+2*8
            break
    if data_begins == 0:
        print("No sync word found!", file=sys.stderr)
        return []
    if whitening:
        lfsr = gen_lfsr_sequence()
        for i in range(data_begins, len(bit_array)):
            bit_array[i] = bit_array[i] ^ next(lfsr)
            bit_array[i] = 1 - bit_array[i]
    byte_array = []
    for i in range(data_begins-16, len(bit_array)-1, 8):
        byte = 0
        for j in range(8):
            byte = byte << 1
            next_bit = 0 if i+j >= len(bit_array) else bit_array[i+j]
            byte += next_bit
        byte_array.append(byte)
    
    print(":".join([hex(b)[2:] if len(hex(b)[2:]) == 2 else "0"+hex(b)[2:] for b in byte_array]), file=sys.stderr)
    print("".join([chr(b) for b in byte_array[5:-4]]), file=sys.stderr)
    return bit_array
    
    

reset_vars()
#buffer = np.zeros(buffer_size)
while True:
    n += 1
    raw_byte = sys.stdin.buffer.read(1)
    if len(raw_byte) == 0:
        break
    data = int.from_bytes(raw_byte, byteorder='big', signed=False)
    #buffer = np.roll(buffer, -1)
    #buffer[-1] = data
    diff = data - prev
    prev = data
    if abs(diff) < diff_threshold:
        count += 1
        #if abs(diff) > diff_threshold/2:
            #data = np.median(buffer)
    else:
        if count > signal_count_threshold:
            decode_timings(timings, zero_value)
        reset_vars()

    
    if zero_count < calibration_samples or one_count < calibration_samples:
        if zero_count < calibration_samples:
            zero_sum += data
            zero_count += 1
        else: 
            zero_diff = abs(data - zero_sum / zero_count)   
            if zero_diff > logic_change_threshold:
                if one_count < calibration_samples:
                    one_sum += data
                    one_count += 1
    else:
        if current_bit_value == -1:
            zero_value = 0 if zero_sum / zero_count < one_sum / one_count else 1
            one_value = 1 - zero_value
            current_bit_value = zero_value
        zero_diff = abs(data - zero_sum / zero_count)
        one_diff = abs(data - one_sum / one_count)
        if zero_diff < one_diff:
            if current_bit_value != zero_value and n-last_logic_change > 5:
                timings.append(n-last_logic_change)
                last_logic_change = n
                current_bit_value = zero_value
        else:
            if current_bit_value != one_value and n-last_logic_change > 5:
                timings.append(n-last_logic_change)
                last_logic_change = n
                current_bit_value = one_value
        
    if current_bit_value == 0:
        sys.stdout.buffer.write((-100+127).to_bytes(1, byteorder='big', signed=False))
    elif current_bit_value == 1:
        sys.stdout.buffer.write((100+127).to_bytes(1, byteorder='big', signed=False))
    else:
        sys.stdout.buffer.write((0+127).to_bytes(1, byteorder='big', signed=False))
        
    
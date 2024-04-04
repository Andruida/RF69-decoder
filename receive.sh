#!/bin/bash

rtl_sdr -s 240000 -f 868000000 -g 0 - | tcc -lm -run fm.c | python3 decode_stat.py -1 static static
# cat 0328_240000_without_antennae.raw | tcc -lm -run fm.c | py decode_stat.py
# cat gqrx_20240327_235110_868084600_240000_fc.raw | py convert_gqrx.py | tcc -lm -run fm.c | py decode_stat.py

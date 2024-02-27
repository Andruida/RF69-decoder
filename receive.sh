#!/bin/bash

rtl_sdr -s 240000 -f 868000000 -g 0 - | tcc -lm -run fm.c | py decode_stat.py
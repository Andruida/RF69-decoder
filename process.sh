cat "$1" | tcc -lm -run fm.c | python3 decode_stat.py $2 $3 $4
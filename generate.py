# python generate.py 10 20 6 24 119238 out.txt
# <airplane id>,<src_airport_id>,<dst_airport_id>,<depart_time>,<arrival_time>

import sys
import random

flight_n = int(sys.argv[1])
airport_n = int(sys.argv[2])
time = int(sys.argv[3])
seed = int(sys.argv[4])
filename = str(sys.argv[5])

random.seed(seed)

plane_ids = [i for i in range(flight_n)]

file = open(filename, "w+")
for i in range(len(plane_ids)):
	start_time = random.randint(0, time-1)
	end_time = random.randint(start_time+1, time)
	
	airports = random.sample(range(0, airport_n), 2)
	random.shuffle(airports)
	src_airport = airports[0]
	dst_airport = airports[1]
	line = str(i) + "," + str(src_airport) + "," + str(dst_airport) + "," + str(start_time) + "," + str(end_time)
	num_spaces = " " * (127 - len(line))
	file.write(line + num_spaces + "\n")

file.close()

# no flights don't go after the 0:00 mark

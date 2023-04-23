import sys, random
# python generate.py 10 20 6 24 119238 out.txt
# <airplane id>,<src_airport_id>,<dst_airport_id>,<depart_time>,<arrival_time>

# Get the command line arguments
num_deliveries = int(sys.argv[1])
num_locations = int(sys.argv[2])
simulation_duration = int(sys.argv[3])
seed = int(sys.argv[4])
filename = str(sys.argv[5])

# Set seed for the random delivery generator
random.seed(seed)

file = open(filename, "w+")
for i in range(num_deliveries):
	start_time = random.randint(0, simulation_duration-1)
	end_time = random.randint(start_time+1, simulation_duration)
	
	locations = random.sample(range(0, num_locations), 2)
	random.shuffle(locations)
	source = airports[0]
	destination = airports[1]
	line = str(i) + "," + str(source) + "," + str(destination) + "," + str(start_time) + "," + str(end_time)
	padding = " " * (127 - len(line)) # line length is fixed to 127 + 1 for convenience
	file.write(line + padding + "\n")
file.close()

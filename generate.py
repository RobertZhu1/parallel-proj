import sys, random
# Execution: python generate.py 1572864 12288 24 100 delivery_test (1572864 deliveries within 12288 transit centers in 24 hours)
# Delivery format: <delivery id>,<source>,<destination>,<departure time>,<arrival time>

class deliveries_generator:
	"""
    Generator of a list of random deliveries given simultation duration
    """
	def __init__(self, filename, num_deliveries, num_transit_centers, simulation_duration):
		"""
        deliveries_generator class contsructor.

		:attr filename: name of the output file
		:type filename: string
		:attr num_deliveries: number of deliveries randomly generated
		:type num_deliveries: integer
		:attr num_transit_centers: number of transit centers for delivery
		:type num_transit_centers: integer
		:attr simulation_duration: duration of the discrete event simulation
		:type simulation_duration: integer
        """
		self.filename = filename
		self.num_deliveries = num_deliveries
		self.num_transit_centers = num_transit_centers
		self.simulation_duration = simulation_duration

	def generate_file(self):
		"""
        Randomly generate a list of deliveries according to attributes and write to output file
        """
		file = open(self.filename, "w+")
		for i in range(self.num_deliveries):
			start_time = random.randint(0, self.simulation_duration-1)
			end_time = random.randint(start_time+1, self.simulation_duration) # End time is later than start time
			trnansit_centers = random.sample(range(0, self.num_transit_centers), 2)
			random.shuffle(trnansit_centers)
			source = trnansit_centers[0]
			destination = trnansit_centers[1]
			line = str(i) + "," + str(source) + "," + str(destination) + "," + str(start_time) + "," + str(end_time)
			padding = " " * (127 - len(line)) # line length is fixed to 127 + 1 for convenience
			file.write(line + padding + "\n")
		file.close()

if __name__ == '__main__':
	# Get the command line arguments
	num_deliveries = int(sys.argv[1])
	num_transit_centers = int(sys.argv[2])
	simulation_duration = int(sys.argv[3])
	seed = int(sys.argv[4])
	filename = str(sys.argv[5])

	# Set seed for the random delivery generator
	random.seed(seed)
	gen = deliveries_generator(filename, num_deliveries, num_transit_centers, simulation_duration)
	gen.generate_file()
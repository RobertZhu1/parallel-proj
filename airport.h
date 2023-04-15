
// A representation of an airport, containing an id and a variable number of runways
struct airport{
	int id;
	struct runway *runways;
	int num_runways;
};

// Stages to represent the different stages that a flight can be in.
enum stages{
	INVALID_FLIGHT = -1,
	WAITING = 0,
	PREPARING_TO_TAKEOFF = 1,
	TAKEOFF = 2,
	CRUISING = 3,
	PREPARING_TO_LAND = 4,
	LANDING = 5,
	LANDED = 6
};

// A representation of a flight, containing information to use in simulation
struct flight{
	int id;
	int source_id;
	int destination_id;
	int stage;
	int current_runway_id;
	unsigned int starting_time;
	unsigned int landing_time;
	unsigned int travel_time;
	unsigned int taxi_time;
	unsigned int wait_time;
};

// A single runway in an airport. Status represents a mutex to ensure one flight
// per runway.
struct runway{
	int id;
	int status;
	int last_access;
};

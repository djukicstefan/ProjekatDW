create table Passanger (
	passanger_id integer NOT NULL PRIMARY KEY,
	first_name varchar(50) NOT NULL,
	last_name varchar(50) NOT NULL,
	birthdate date,
	street varchar(80),
	city varchar(50),
	phone varchar(30),
	email varchar(50)
);

create table Plane (
	plane_id integer NOT NULL PRIMARY KEY,
	model varchar(50) NOT NULL,
	airline varchar(50) NOT NULL,
	capacity integer
);

create table Flight (
	flight_id integer NOT NULL PRIMARY KEY,
	plane_id integer NOT NULL,
	arrival_date timestamp,	
	departure_date timestamp,
	departure_city varchar(80),
	arrival_city varchar(80),
	FOREIGN KEY (plane_id) REFERENCES Plane (plane_id)
);

create table Reservation (
	reservation_id integer NOT NULL PRIMARY KEY,
	passanger_id integer,
	flight_id integer,
	seat_number integer,
	seat_class varchar(50),
	ticket_price integer,
	food_cost integer,
	reservation_date timestamp,
	FOREIGN KEY (passanger_id) REFERENCES Passanger (passanger_id),
	FOREIGN KEY (flight_id) REFERENCES Flight (flight_id)
);
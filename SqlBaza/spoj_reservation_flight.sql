SELECT reservation.*, flight.plane_id
FROM reservation
INNER JOIN flight ON reservation.flight_id = flight.flight_id 

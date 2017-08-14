#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"
#include <typeinfo>

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2( (map_y-y),(map_x-x) );

	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
	}

	return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

// Conditions to turn:
bool safe_turn(double check_car_s, double car_s, int security, double car_speed, double check_speed, double ref_speed, double front_car_s)
{
		bool turn = true;
		double time_manouver = 3*security/car_speed;
		double speed_diff = check_speed-car_speed;
		double ref_speed_diff = check_speed - ref_speed;

		//if a car is behind within the conflict area and runing faster:
		if ((check_car_s <= car_s) && (car_s - check_car_s) < abs(speed_diff*time_manouver) && (speed_diff>=0)) {
				turn = false;
		}
		//if a car is on front moving slower than the reference car and within the conflict area:
		if (((check_car_s>=front_car_s) &&(check_car_s-front_car_s)<security) && (ref_speed_diff<=0)) {
				turn = false;
		}
		// if a car is between the autonomous car and the reference car
		if ((check_car_s >= car_s) && (check_car_s <= front_car_s)){
				turn = false;
		}

		// if both cars are at the same s or check car is behind the reference car
		if(abs(check_car_s - car_s) < 5 || ((front_car_s - check_car_s) > 5 && (check_car_s>=car_s-5))){
				turn = false;
		}

		return turn;
}

// get line from d:
int get_line_from_d(double d){
		if (d < 8. && d > 4.){
				return 1;
		}
		else if (d >= 8.){
				return 2;
		}
		else{
				return 0;
		}
}

// Check if there is a close car on front:
vector<double> car_front(json sensor_fusion,double lane,double car_speed,double car_s,int security,int prev_size){

		double getting_closer = 0;
		double too_close = 0;
		double between_distance = 3*security;
		double front_car_speed = 0;
		double front_car_s = 0;
		double car_break = 0;


		for(int i = 0; i < sensor_fusion.size(); i++) {
				// retrieve surrounding cars information:
				double vx = sensor_fusion[i][3];
				double vy = sensor_fusion[i][4];
				double check_speed = sqrt(vx * vx + vy * vy);
				double check_car_s = sensor_fusion[i][5];
				float d = sensor_fusion[i][6];

				check_car_s += ((double)prev_size*.02*check_speed);

				// car is in my line:
				if (d < (2+4*lane+2) && d > (2+4*lane-2)){

						if((check_car_s > car_s) && (check_car_s-car_s) < security && (car_speed > check_speed) && (between_distance>check_car_s-car_s)){
								getting_closer = 1;
								between_distance = check_car_s-car_s;
								front_car_speed = check_speed;
								front_car_s = check_car_s;
								car_break = .3 * security/(check_car_s-car_s);
						}
						if((check_car_s > car_s) && (check_car_s-car_s) < security*0.8 && (car_speed > check_speed)){
								too_close = 1;
						}
				}

		}
		return {getting_closer,between_distance,front_car_speed,front_car_s,car_break,too_close};
}

// Check if car can turn:
vector<bool> car_turn(json sensor_fusion,vector<double> front,double lane,double car_speed,double car_s,int security, bool maneuver_completed,int prev_size){

		double front_car_speed = front[2];
		double front_car_s = front[3];

		bool turn_right = true;
		bool turn_left = true;


		for(int i = 0; i < sensor_fusion.size(); i++) {
				// retrieve surrounding cars information:
				double vx = sensor_fusion[i][3];
				double vy = sensor_fusion[i][4];
				double check_speed = sqrt(vx * vx + vy * vy);
				double check_car_s = sensor_fusion[i][5];
				float d = sensor_fusion[i][6];

				check_car_s += ((double)prev_size*.02*check_speed);

				// car is in different line:
				if(abs(lane - get_line_from_d(d))==1 && maneuver_completed){

						// check turn right:
						if(lane == 2){
								turn_right = false;
						}

						if(turn_right && (lane < 2) && (lane + 1 == get_line_from_d(d))){
								turn_right = safe_turn(check_car_s, car_s, security, car_speed, check_speed, front_car_speed,front_car_s);
						}

						// check turn left:
						if(lane == 0){
								turn_left = false;
						}

						if(turn_left && (lane > 0) && (lane - 1 == get_line_from_d(d))){
								turn_left = safe_turn(check_car_s, car_s, security, car_speed, check_speed, front_car_speed,front_car_s);
						}
				}

		}
		return {turn_right, turn_left};
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  // start in line 1:
  int lane = 1;

	// check whether maneuver is completed:
	double maneuver_start_s = 0;
	bool maneuver_completed = true;

  // Define a reference velocity:
  double ref_vel = 0.0; //mph

  h.onMessage([&maneuver_completed,&maneuver_start_s,&lane,&ref_vel,&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

						// Define the previous path size:
						int prev_size = previous_path_x.size();

						//Define the actual (x,y) points that will be used for the planner:
						vector<double> next_x_vals;
						vector<double> next_y_vals;

						if(prev_size > 0)
						{
								car_s = end_path_s;
						}

						// security margin:
						int security = 30; //meters

						// define turns:
						bool turn_right = true;
						bool turn_left = true;

						// check if close car on front:
						vector<double> front = car_front(sensor_fusion,lane,car_speed,car_s,security,prev_size);

						// if car on front, check the available turns:
						double getting_closer = front[0];
						double too_close = front[5];
						if (getting_closer==1){
								vector<bool>turns = car_turn(sensor_fusion,front,lane,car_speed,car_s,security, maneuver_completed,prev_size);
								turn_right = turns[0];
								turn_left = turns[1];
						}

						// since this is a circuit, we need to reset manouver_start_s at every lap:
						if (car_s<maneuver_start_s){
								maneuver_start_s = car_s;
						}

						// check if car is ready to maneuver:
						if (car_s > maneuver_start_s + 3*security){
								maneuver_completed = true;
						}


						// A car is on front and need to turn left:
						if(getting_closer && turn_left && maneuver_completed){
								if (lane>0){
										lane -= 1;
										maneuver_start_s = car_s;
										maneuver_completed = false;
								}
						}
						// A car is in front and need to turn right:
						else if(getting_closer && turn_right && maneuver_completed){
								if (lane<2){
										lane += 1;
										maneuver_start_s = car_s;
										maneuver_completed = false;
								}

						}

						// adjust speed with regards to current speed or cars infront of us
						if(too_close){
								ref_vel -= front[4]; //.224;
						}
						else if(ref_vel < 49.5){ //49.5
								ref_vel += .224;
						}


            // Create a list of widely spaced (x, y) waypoints, spaced 30m:
            vector<double> ptsx;
            vector<double> ptsy;

						// reference x, y, yaw states (either these values comes from the starting points or previous
						// path end points.
						double ref_x = car_x;
						double ref_y = car_y;
						double ref_yaw = deg2rad(car_yaw);

						// if previous size is almost empty, use the car's reference as starting point:
						if(prev_size < 2)
						{
								double prev_car_x = car_x - cos(car_yaw);
								double prev_car_y = car_y - sin(car_yaw);

								// fill first 2 values form the previous positions:
								ptsx.push_back(prev_car_x);
								ptsx.push_back(car_x);

								ptsy.push_back(prev_car_y);
								ptsy.push_back(car_y);

						}
						// use previous path end points as starting reference
						else
						{
								// redefine references state based on previous path end points:
								ref_x = previous_path_x[prev_size-1];
								ref_y = previous_path_y[prev_size-1];

								double ref_x_prev = previous_path_x[prev_size-2];
								double ref_y_prev = previous_path_y[prev_size-2];
								ref_yaw = atan2(ref_y-ref_y_prev,ref_x-ref_x_prev);

								// use these two points as references:
								ptsx.push_back(ref_x_prev);
								ptsx.push_back(ref_x);

								ptsy.push_back(ref_y_prev);
								ptsy.push_back(ref_y);
						}

						// Generate 3 waypoints separated apart 30m in Frenet coordinates:
						vector<double> next_wp0 = getXY(car_s + security,(2+4*lane),map_waypoints_s,map_waypoints_x,map_waypoints_y);
						vector<double> next_wp1 = getXY(car_s + 2*security,(2+4*lane),map_waypoints_s,map_waypoints_x,map_waypoints_y);
						vector<double> next_wp2 = getXY(car_s + 3*security,(2+4*lane),map_waypoints_s,map_waypoints_x,map_waypoints_y);

						ptsx.push_back(next_wp0[0]);
						ptsx.push_back(next_wp1[0]);
						ptsx.push_back(next_wp2[0]);

						ptsy.push_back(next_wp0[1]);
						ptsy.push_back(next_wp1[1]);
						ptsy.push_back(next_wp2[1]);

						// shith car reference angle to 0 degrees (simplify calculations)
						for (int i = 0; i < ptsx.size(); i++)
						{
								double shift_x = ptsx[i]-ref_x;
								double shift_y = ptsy[i]-ref_y;

								ptsx[i] = (shift_x*cos(0-ref_yaw)-shift_y*sin(0-ref_yaw));
								ptsy[i] = (shift_x*sin(0-ref_yaw)+shift_y*cos(0-ref_yaw));
						}

						// create the spline:
						tk::spline s;

						// set (x,y) points for the spline:
						s.set_points(ptsx,ptsy);

						// Start with the previous points from the previous step:
						for (int i = 0; i < previous_path_x.size(); i++){
								next_x_vals.push_back(previous_path_x[i]);
								next_y_vals.push_back(previous_path_y[i]);
						}

						// Define waypoints separation:
						double target_x = 30.0;
						double target_y = s(target_x);
						double target_dist = sqrt((target_x)*(target_x)+(target_y)*(target_y));

						double x_add_on = 0;

						// Fill up the rest of our path planner after filling it with the previous points, the
						// output size is set to 50 points:
						for (int i = 1; i <= 50-previous_path_x.size(); i++){

								// 2.24 conversion from mph to km/h
								// .02 => target distance
								double N = (target_dist/(.02*ref_vel/2.24));
								double x_point = x_add_on+(target_x)/N;
								double y_point = s(x_point);

								x_add_on = x_point;

								double x_ref = x_point;
								double y_ref = y_point;

								// rotate points from local to global:
								x_point = (x_ref*cos(ref_yaw)-y_ref*sin(ref_yaw));
								y_point = (x_ref*sin(ref_yaw)+y_ref*cos(ref_yaw));

								x_point += ref_x;
								y_point += ref_y;

								next_x_vals.push_back(x_point);
								next_y_vals.push_back(y_point);

						}


          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}

















































































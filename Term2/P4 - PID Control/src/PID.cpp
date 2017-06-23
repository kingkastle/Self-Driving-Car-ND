#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;

  p_error = 0;
  i_error = 0;
  d_error = 0;
}

void PID::UpdateError(double cte) {
  d_error = cte - p_error;
  p_error = cte;
  i_error += cte;

}

double PID::TotalError() {
  double steering_a = (-1) * (Kp * p_error + Ki * i_error + Kd * d_error);
  if (steering_a > 1){
    steering_a = +1;
  }
  if (steering_a < -1){
    steering_a = -1;
  }
  return steering_a;
}


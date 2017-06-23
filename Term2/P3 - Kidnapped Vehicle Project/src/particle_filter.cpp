/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles =500;

  default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for the different variables:
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize all particles:
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
  }

  // Initialize all weights to 1.
  weights.clear();
  weights.resize(num_particles);
  for (int i = 0; i < weights.size(); i++) {
    weights[i] = 1.0;
  }

  // set is_initialized to true:
  is_initialized = true;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


  // initialize noises around 0.0
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  // add measurements to particles:
  for (int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) < 0.001){
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      }
    else{
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // Add random noise:
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // calculate variance and covar:
  double variance_x = 2 * pow(std_landmark[0], 2);
  double variance_y = 2 * pow(std_landmark[1], 2);
  double covariance_xy = std_landmark[0] * std_landmark[1];

  // define sum of weights:
  double sum_weights = 0;

  // iterate over all particles:
  for (int i = 0; i < num_particles; ++i) {

    // initialize weight:
    double weight = 1.0;
    for (int j = 0; j < observations.size(); j++) {
      // perform observation transformation from VEHICLE to MAP:
      double predicted_x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
      double predicted_y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;

      // define min distance parameter and initialize associated landmark:
      double min_dist = sensor_range;
      Map::single_landmark_s closest_landmark;

      // identify closest landmark to transformed observation
      for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
        Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
        double current_dist = dist(predicted_x, predicted_y, landmark.x_f, landmark.y_f);
        if (current_dist < min_dist) {
          min_dist = current_dist;
          closest_landmark = landmark;
          }
        }

      // update weight for the associated particle and in weights vector:
      double x_diff = pow(particles[i].x - closest_landmark.x_f, 2);
      double y_diff = pow(particles[i].y - closest_landmark.y_f, 2);
      double exponente = -1 * ((x_diff/variance_x) + (y_diff/variance_y));
      weight *= (1/(2 * M_PI * covariance_xy)) * exp(exponente);
      }

    // update associated weight for the current particle:
    particles[i].weight = weight;
    weights[i] = particles[i].weight;

    sum_weights += particles[i].weight;
    }

//  // Normalize weights:
//  for (int i = 0; i < num_particles; i++) {
//    particles[i].weight = particles[i].weight / sum_weights;
//    weights[i] = weights[i] / sum_weights;
//    }
  }

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;

  // Take a discrete distribution with pmf equal to weights
  discrete_distribution<> weights_distribution(weights.begin(), weights.end());

  // initialise new particle array
  vector<Particle> resampled_Particles;

  // resample particles
  for (int i = 0; i < num_particles; ++i)
    resampled_Particles.push_back(particles[weights_distribution(gen)]);

  particles = resampled_Particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

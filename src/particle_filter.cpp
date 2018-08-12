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

struct lesser {
    bool operator()(const Particle& a, const Particle& b) {
        return a.weight < b.weight;
    }
};

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    const double std_x = std[0];
    const double std_y = std[1];
    const double std_yaw = std[2];
    default_random_engine gen;

    num_particles = 100;
    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_yaw);

    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
        particles.push_back(p);
        weights.push_back(p.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	const double std_x = std_pos[0];
	const double std_y = std_pos[1];
	const double std_yaw = std_pos[2];

    std::normal_distribution<double> dist_x(0, std_x);
    std::normal_distribution<double> dist_y(0, std_y);
    std::normal_distribution<double> dist_theta(0, std_yaw);

    for (int i = 0; i < num_particles; i++) {
        Particle &p = particles[i]; // Be careful to use reference, if not, it will be a new particle
        if (fabs(yaw_rate) - 0.0001 > 0) {
            p.x = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
            p.y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + dist_y(gen);
            p.theta = p.theta + yaw_rate * delta_t + dist_theta(gen);
        } else {
            p.x = p.x + velocity * yaw_rate + dist_x(gen);
            p.y = p.y + velocity * yaw_rate + dist_y(gen);
            p.theta = p.theta + dist_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> landmarks, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    // for  each landmark obs, compare with predicted, find min distTo, and assign the landmarkId of predicted to observation
    for (auto &observation : observations) {
        double minDistTo = 999.0;
        for (int j = 0; j < landmarks.size(); j++) {
            const double distTo = dist(observation.x, observation.y, landmarks[j].x_f, landmarks[j].y_f);
            if (distTo < minDistTo) { minDistTo = distTo; observation.id = landmarks[j].id_i; }
        }
        // cout << "MinDistTo: " << minDistTo << " landmarkId: " << observation.id << endl;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

	double sumOfWeights = 0;
	for (auto &particle: particles) {
	    vector<LandmarkObs> transformedObs;
	    for (auto &obs: observations) {
	        LandmarkObs newObs;
	        // Transform from car coordinate to map coordinate
	        newObs.id = 1; // Some dummy landmark id
	        newObs.x = particle.x + obs.x * cos(particle.theta) - sin(particle.theta) * obs.y;
	        newObs.y = particle.y + obs.x * sin(particle.theta) + cos(particle.theta) * obs.y;
	        transformedObs.push_back(newObs);
	    }
	    vector<Map::single_landmark_s> filteredLandmarks;
	    for (auto &landmark: map_landmarks.landmark_list) {
	        if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) < sensor_range) {
	            filteredLandmarks.push_back(landmark);
	        }
	    }
        // associate each observation to closest landmark for each particle
        dataAssociation(filteredLandmarks, transformedObs);

        // retrieve associations, sense_x, sense_y, calculate weight for this particle
        double prob = 1.0;
        const double sigma_x = std_landmark[0];
        const double sigma_y = std_landmark[1];
        const double norm = 1 / (2 * M_PI * sigma_x * sigma_y);
        vector<int> associations;
        vector<double> sense_x, sense_y;
        for (auto &obs: transformedObs) {
            associations.push_back(obs.id);
            const double x_obs = obs.x;
            const double y_obs = obs.y;
            const double x_mu = map_landmarks.landmark_list[obs.id-1].x_f; // assuming index of landmark_list == landmarkId
            const double y_mu = map_landmarks.landmark_list[obs.id-1].y_f;
            // multivariate gaussian
            const double exponent = (((x_obs - x_mu) * (x_obs - x_mu)) / (2 * sigma_x * sigma_x)) + (((y_obs - y_mu) * (y_obs - y_mu)) / (2 * sigma_y * sigma_y));
            prob *= norm * exp(-exponent);
            sense_x.push_back(x_obs);
            sense_y.push_back(y_obs);
        }
        particle.weight = prob;
        sumOfWeights += prob;

        SetAssociations(particle, associations, sense_x, sense_y);
	}
	for (int i = 0; i < particles.size(); ++i) {
        if (sumOfWeights > 0) {
            particles[i].weight /= sumOfWeights;
            weights[i] = particles[i].weight;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	std::discrete_distribution<int> distribution (weights.begin(), weights.end());

	vector <Particle> temp;
	for (int i = 0; i < num_particles; i++) {
	    Particle new_p;
	    const int random_int = distribution(gen);
        new_p.id = i;
        new_p.x = particles[random_int].x;
        new_p.y = particles[random_int].y;
        new_p.theta = particles[random_int].theta;
        new_p.weight = particles[random_int].weight;
        SetAssociations(new_p, particles[random_int].associations, particles[random_int].sense_x, particles[random_int].sense_y);
        // new_p.associations = particles[random_int].associations;
        // new_p.sense_x = particles[random_int].sense_x;
        // new_p.sense_y = particles[random_int].sense_y;
        temp.push_back(new_p);
	}
    particles = temp;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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

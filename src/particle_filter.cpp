/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang, Lee Yu
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>

#include <math.h>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Normal distributions for x, y, theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Randomly generate particles
  std::default_random_engine gen;
  for (int i = 0; i < num_particles; i++) {
    particles.push_back(Particle{.id=i, .x=dist_x(gen), .y=dist_y(gen), .theta=dist_theta(gen), .weight=1});
    weights.push_back(1.0);
  }
  is_initialized = true;
  std::cout << "inited" << std::endl;
}

void ParticleFilter::prediction(double delta_t, double std[], double velocity, double yaw_rate) {
  std::normal_distribution<double> dist_noise_x(0, std[0]);
  std::normal_distribution<double> dist_noise_y(0, std[1]);
  std::normal_distribution<double> dist_noise_theta(0, std[2]);
  std::default_random_engine gen;

  for (auto p = particles.begin(); p < particles.end(); p++) {
    // random noises
    double noise_x = dist_noise_x(gen);
    double noise_y = dist_noise_y(gen);
    double noise_theta = dist_noise_theta(gen);
    if (yaw_rate == 0) {
      p->x += velocity * cos(p->theta) * delta_t + noise_x;
      p->y += velocity * sin(p->theta) * delta_t + noise_y;
    } else {
      p->x += velocity * (sin(p->theta + yaw_rate * delta_t) - sin(p->theta)) / yaw_rate + noise_x;
      p->y += velocity * (-cos(p->theta + yaw_rate * delta_t) + cos(p->theta)) / yaw_rate + noise_y;
    }
    p->theta += yaw_rate * delta_t + noise_theta;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  for (auto it = observations.begin(); it < observations.end(); it++) {
    double min_d = INFINITY;
    int associated_id = -1;
    // associate with the closest landmark
    for (LandmarkObs pred : predicted) {
      double d = dist(it->x, it->y, pred.x, pred.y);
      if (d < min_d) {
        min_d = d;
        associated_id = pred.id;
      }
    }
    it->id = associated_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  std::unordered_map<int, Map::single_landmark_s> landmarks;
  for (auto l : map_landmarks.landmark_list) {
    landmarks[l.id_i] = l;
  }

  for (int i = 0; i < particles.size(); i++) {
    Particle *p = &particles[i];
    // compute predicted observations
    std::vector<LandmarkObs> pred_obs;
    for (auto l : map_landmarks.landmark_list) {
      double d = dist(l.x_f, l.y_f, p->x, p->y);
      if (d <= sensor_range) {
        pred_obs.push_back(LandmarkObs{.x=l.x_f, .y=l.y_f, .id=l.id_i});
      }
    }

    // transform the observations from the particle's coordinates to map coordinates
    std::vector<LandmarkObs> map_obs;
    map_obs.reserve(observations.size());
    for (auto obs : observations) {
      map_obs.push_back(LandmarkObs{
        .x = p->x + cos(p->theta) * obs.x - sin(p->theta) * obs.y,
        .y = p->y + sin(p->theta) * obs.x + cos(p->theta) * obs.y,
      });
    }

    // associate observations with landmarks
    dataAssociation(pred_obs, map_obs);

    // compute weight
    double w = !map_obs.empty() ? 1 : 0;
    std::vector<int> asso_id;
    std::vector<double> asso_x;
    std::vector<double> asso_y;
    for (auto obs : map_obs) {
      // associated landmark
      auto landmark = landmarks[obs.id];
      auto g = gaussian(obs.x, obs.y, landmark.x_f, landmark.y_f, std_landmark[0], std_landmark[1]);
      w *= g;

      asso_id.push_back(obs.id);
      asso_x.push_back(obs.x);
      asso_y.push_back(obs.y);
    }
    p->weight = weights[i] = w;

    setAssociations(*p, asso_id, asso_x, asso_y);
  }
}

void ParticleFilter::resample() {
  std::random_device rd;
  std::mt19937 gen(rd());
  // Probability of being drawn is proportional to the weight
  std::discrete_distribution<> d(weights.cbegin(), weights.cend());
  std::vector<Particle> resampled;
  for (int i = 0; i < num_particles; i++) {
    int k = d(gen);
    resampled.push_back(particles[k]);
  }
  particles = resampled;
}

void ParticleFilter::setAssociations(Particle &particle, const std::vector<int> &associations,
                                     const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseX(Particle best) {
  std::vector<double> v = best.sense_x;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseY(Particle best) {
  std::vector<double> v = best.sense_y;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

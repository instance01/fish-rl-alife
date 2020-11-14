
"""
sources:
    speed and angle calculation:
        https://ericleong.me/research/circle-circle/#static-circle-circle-collision-response
        https://stackoverflow.com/questions/30497287/elastic-2d-ball-collision-using-angles
    overlap correction ("sticky problem"):
        http://www.petercollingridge.co.uk/tutorials/pygame-physics-simulation/collisions/
"""
import numpy as np
from itertools import combinations
from env.animal import Animal
import env.util as util


class CollisionSpace:
    DEBUG = False
    OVERLAP_CORRECTION = 0.001
    ELASTICITY = 0.2
    
    def __init__(self, torus, x, y, width, height):
        self.torus = torus
        self.min_x = x
        self.min_y = y
        self.max_x = width
        self.max_y = height
        
    def check_collision(self, a1: Animal, a2: Animal) -> bool:
        if self.torus:
            dx, dy = util.vector_in_torus_space(a1.position, a2.position, self.max_x, self.max_y) 
        else:
            dx, dy = a2.position - a1.position

        distance_squared = dx ** 2 + dy ** 2
        min_distance_squared = (a1.radius + a2.radius) ** 2
        
        return distance_squared < min_distance_squared

    def perform_collision(self, a1: Animal, a2: Animal) -> bool:
        # check for collision
        if self.torus:
            dx, dy = util.vector_in_torus_space(a2.position, a1.position, self.max_x, self.max_y)
        else:
            dx, dy = a1.position - a2.position

        distance_squared = dx ** 2 + dy ** 2
        min_distance_squared = (a1.radius + a2.radius) ** 2

        if distance_squared < min_distance_squared:
            # debug
            if CollisionSpace.DEBUG:
                print("overlap detected:\n positions: [{} {}] and [{} {}]"
                      .format(a1.position[0], a1.position[1], a2.position[0], a2.position[1]))
                print("velocities: [{} {}] and [{} {}]\n"
                      .format(a1.velocity[0], a1.velocity[1], a2.velocity[0], a2.velocity[1]))
                      
            # compute fancy numbers to estimate velocities after collision
            if self.torus:
                dvx, dvy = util.vector_in_torus_space(a1.velocity, a2.velocity, self.max_x, self.max_y)
            else:
                dvx, dvy = a2.velocity - a1.velocity
                
            dot = dx * dvx + dy * dvy
            factor = dot / distance_squared
            
            # set new velocities
            a1.velocity[0] += factor * dx
            a1.velocity[1] += factor * dy
            a2.velocity[0] -= factor * dx
            a2.velocity[1] -= factor * dy
            # apply elasticity
            a1.velocity *= CollisionSpace.ELASTICITY
            a2.velocity *= CollisionSpace.ELASTICITY
            # compute collision angle
            collision_angle = np.arctan2(dy, dx) + np.pi * 0.5
            # compute overlap distance
            overlap_distance = np.sqrt(min_distance_squared) - np.sqrt(distance_squared)
            overlap_correction = 0.5 * (overlap_distance + CollisionSpace.OVERLAP_CORRECTION)
            # correct overlap w.r.t. the collision angle
            a1.position[0] += np.sin(collision_angle) * overlap_correction
            a1.position[1] -= np.cos(collision_angle) * overlap_correction
            a2.position[0] -= np.sin(collision_angle) * overlap_correction
            a2.position[1] += np.cos(collision_angle) * overlap_correction
            # debug
            if CollisionSpace.DEBUG:
                print("overlap resolved:\n new positions: [{} {}] and [{} {}]"
                      .format(a1.position[0], a1.position[1], a2.position[0], a2.position[1]))
                print("new velocities: [{} {}] and [{} {}]\n"
                      .format(a1.velocity[0], a1.velocity[1], a2.velocity[0], a2.velocity[1]))

    def perform_boundary_collision(self, animal: Animal) -> bool:
        """
        UPDATE: the perform_boundary_collision function returns whether a collision has happened!
        """

        """"
        For each animal in animals:
            1. perform collision detection with provided boundaries
            2. eventually perform collision
                2.1 update velocities
                2.2 reposition to avoid sticking
        """
        min_pos_x = self.min_x + animal.radius
        max_pos_x = self.max_x - animal.radius
        min_pos_y = self.min_y + animal.radius
        max_pos_y = self.max_y - animal.radius

        if self.torus:
            
            if animal.position[0] < self.min_x:
                animal.position[0] = self.max_x
            elif animal.position[0] > self.max_x:
                animal.position[0] = self.min_x
        
            if animal.position[1] < self.min_y:
                animal.position[1] = self.max_y
            elif animal.position[1] > self.max_y:
                animal.position[1] = self.min_y
            return False

        else:
            wall_collision: bool = False
            if animal.position[0] < min_pos_x:
                animal.velocity[0] *= -1 * CollisionSpace.ELASTICITY
                animal.position[0] = min_pos_x + CollisionSpace.OVERLAP_CORRECTION
                wall_collision = True
            elif animal.position[0] > max_pos_x:
                animal.velocity[0] *= -1 * CollisionSpace.ELASTICITY
                animal.position[0] = max_pos_x - CollisionSpace.OVERLAP_CORRECTION
                wall_collision = True
            if animal.position[1] < min_pos_y:
                animal.velocity[1] *= -1 * CollisionSpace.ELASTICITY
                animal.position[1] = min_pos_y + CollisionSpace.OVERLAP_CORRECTION
                wall_collision = True
            elif animal.position[1] > max_pos_y:
                animal.velocity[1] *= -1 * CollisionSpace.ELASTICITY
                animal.position[1] = max_pos_y - CollisionSpace.OVERLAP_CORRECTION
                wall_collision = True
            return wall_collision


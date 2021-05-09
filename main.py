import pygame
import os
import math
import sys
import neat
from pygame.locals import *

WIDTH = 1920
HEIGHT = 1080
pygame.display.set_caption("AI self driving car")
FPS = 60

CAR_IMAGE = pygame.image.load(os.path.join("assets", "car.png"))

CAR_SIZE_X = 60
CAR_SIZE_Y = 60
CRASH_COLOR = (255, 255, 255, 255)

class Car:
    def __init__(self):
        self.body = pygame.image.load(os.path.join("assets", "car.png")).convert_alpha()
        self.body = pygame.transform.scale(self.body, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_body = self.body

        self.position = [830,920]
        self.position_center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.angle = 0

        self.speed = 0
        self.set_speed = False

        self.radars = []
        self.alive = True
        self.distane = 0
        self.time = 0

    def get_alive(self):
        return self.alive

    def draw(self, screen):
        screen.blit(self.rotated_body, self.position)
        self.draw_radars(screen)

    def radar(self, degree, map):
        length = 0
        x = int(self.position_center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.position_center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not map.get_at((x,y)) == CRASH_COLOR and length < 300:
            length += 1
            x = int(self.position_center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.position_center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        distance_to_border = int(math.sqrt(math.pow(x - self.position_center[0], 2) + math.pow(y - self.position_center[1], 2)))
        self.radars.append([(x,y), distance_to_border])

    def draw_radars(self, screen):
        for radar in self.radars:
            pos = radar[0]
            pygame.draw.line(screen, (255,255,0), self.position_center, pos, 1)
            pygame.draw.circle(screen, (255,255,0), pos, 5)

    def check_crash(self,map):
        self.alive = True
        for point in self.corners:
            if map.get_at((int(point[0]), int(point[1]))) == CRASH_COLOR:
                self.alive = False
                break

    def state(self, map):
        if not self.set_speed:
            self.speed = 10
            self.set_speed = True

        self.rotated_body = self.rotate_center(self.body, self.angle)

        self.position[0] += math.cos(math.radians(360-self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH-120)

        self.distane += self.speed
        self.time += 1

        self.position[1] += math.sin(math.radians(360-self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH-120)

        self.position_center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        length = 0.5 * CAR_SIZE_X
        left_top = [self.position_center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.position_center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.position_center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.position_center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.position_center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.position_center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.position_center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.position_center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]

        self.corners = [left_top,right_top,left_bottom,left_top]

        self.check_crash(map)
        self.radars.clear()

        for x in range(-90, 120, 45):
            self.radar(x, map)

    def get_data(self):
        radars = self.radars
        ret = [0,0,0,0,0]
        for i, radar in enumerate(radars):
            ret[i] = int(radar[1] / 30)
        return ret

    def get_reward(self):
        return self.distane / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

def main(genomes, config):
    clock = pygame.time.Clock()
    run = True
    cars = []
    networks = []

    pygame.init()
    WINDOW = pygame.display.set_mode((WIDTH,HEIGHT))
       
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        networks.append(net)
        g.fitness = 0

        cars.append(Car())

    TRACK = pygame.image.load(os.path.join("assets", "map2.png")).convert() 
    counter = 0

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
                run = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    run = False

        for i, car in enumerate(cars):
            output = networks[i].activate(car.get_data())
            choice = output.index(max(output))

            if choice == 0:
                car.angle += 10 
            elif choice == 1:
                car.angle -= 10 
            elif choice == 2:
                if(car.speed - 2 >= 12):
                    car.speed -= 2 
            else:
                car.speed += 2 

        count_alive = 0
        
        for i, car in enumerate(cars):
            if car.get_alive():
                count_alive += 1
                car.state(TRACK)
                genomes[i][1].fitness += car.get_reward()

        if count_alive == 0:
            break

        counter += 1
        if counter == 30 * 40: # Stop After About 20 Seconds
            break

        WINDOW.blit(TRACK, (0,0))
        for car in cars:
            if car.get_alive():
                car.draw(WINDOW)

        pygame.display.flip()
        clock.tick(FPS)


def menu():
    pygame.init()
    bg = pygame.image.load(os.path.join("assets", "background.png"))
    bg = pygame.transform.scale(bg, (1920,1080))
    button_draw = pygame.image.load(os.path.join("assets", "draw.png"))
    button_draw = pygame.transform.scale(button_draw, (400,100))
    button_ready_maps = pygame.image.load(os.path.join("assets", "ready.png"))
    button_ready_maps = pygame.transform.scale(button_ready_maps, (400,100))
    button_exit = pygame.image.load(os.path.join("assets", "exit2.png"))
    button_exit = pygame.transform.scale(button_exit, (400,100))
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    menu_clock = pygame.time.Clock()
    click = False

    while True:
        click = False
        screen.fill((255,255,255))
        screen.blit(bg, (0,0))
        screen.blit(button_draw, (740,350))
        screen.blit(button_ready_maps, (740,500))
        screen.blit(button_exit, (740,650))

        b_ready_maps = pygame.Rect(740,500,400,100)
        b_draw = pygame.Rect(740,350,400,100)
        b_exit = pygame.Rect(740,650,400,100)

        mx, my = pygame.mouse.get_pos()

        if b_ready_maps.collidepoint((mx,my)):
            if click:
                game()
        if b_exit.collidepoint((mx,my)):
            if click:
                sys.exit()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    click = True

        pygame.display.update()
        menu_clock.tick(60)
        
def game():
    global repeats
    repeats = 100
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(main, repeats)

menu()



import pygame
import numpy as np
from sklearn.datasets import make_blobs
from collections import Counter
from operator import itemgetter


def dist(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def k_neighbours(k, distances):
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours


def add_predictions(predictions, pred):
    predictions.append(pred)
    return predictions


class KNN:
    def __init__(self):
        self.optimal_k = None
        self.train_points = []
        self.train_clusters = []

    def distances(self, point):
        dists = []
        for i in range(len(self.train_points)):
            dists.append((self.train_clusters[i], dist(self.train_points[i], point)))
        dists.sort(key=lambda x: x[1])
        return dists

    def train(self, x_train, y_train):
        self.train_points = x_train
        self.train_clusters = y_train

    def predict(self, points, optimal):
        predictions = []
        for point in points:
            if not optimal and self.optimal_k is not None:
                distances = self.distances(point)
                neighbours = k_neighbours(self.optimal_k, distances)
                labels = [neighbour for neighbour in neighbours]
                prediction = max(labels, key=labels.count)
                predictions = add_predictions(predictions, prediction)
            else:
                ks = []
                distances = self.distances(point)
                max_k = int(np.ceil(np.sqrt(len(self.train_points))))
                for i in range(3, max_k):
                    neighbours = k_neighbours(i, distances)
                    counter = Counter(neighbours)
                    probabilities = [(count[0], count[1] / len(neighbours) * 100.0) for count in counter.most_common()]
                    ks.append((probabilities[0][0], probabilities[0][1], i))
                prediction = max(ks, key=itemgetter(1))
                self.optimal_k = prediction[2]
                predictions = add_predictions(predictions, prediction[0])
        return predictions, self.optimal_k


class Area:

    def __init__(self):
        self.k = None
        self.training = True
        self.current_point = None
        self.current_cluster = 0
        points, clusters = make_blobs(n_samples=initial_points_num, centers=clusters_num,
                                      cluster_std=30, center_box=(100, 720 - 100))
        self.points = list(map(lambda x: [x[0], x[1]], points))
        self.clusters = list(map(lambda x: x + 1, clusters))

    def run(self):
        s = False
        font = pygame.font.SysFont('Arial', font_size, True)

        while not s:
            clock.tick(30)
            events = pygame.event.get()
            for event in events:
                print event.type
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    s = True

                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.current_point = None
                    self.training = False
                    screen.fill(white)

                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.current_cluster = 0
                    self.current_point = None
                    self.points = []
                    self.k = None
                    knn.optimal_k = None
                    self.training = True
                    points, clusters = make_blobs(n_samples=initial_points_num, centers=clusters_num,
                                                  cluster_std=30, center_box=(100, 720 - 100))
                    self.points = list(map(lambda x: [x[0], x[1]], points))
                    self.clusters = list(map(lambda x: x + 1, clusters))
                    start_colors()
                    screen.fill(white)

                if event.type == pygame.KEYDOWN and '1' <= event.unicode <= str(clusters_num):
                    self.current_cluster = int(event.unicode)

                left = pygame.mouse.get_pressed()[0]
                right = pygame.mouse.get_pressed()[2]
                if left:
                    self.current_point = pygame.mouse.get_pos()
                    if self.training:
                        if self.current_cluster != 0:
                            if self.current_point is not None:
                                self.points.append(self.current_point)
                                self.clusters.append(self.current_cluster)
                                self.current_point = None
                    else:
                        knn.train(self.points, self.clusters)
                        pred, optimal_k = knn.predict([self.current_point], right)
                        self.k = optimal_k
                        self.points.append(self.current_point)
                        self.clusters += pred
                        self.current_point = []

            if self.training:
                surf = font.render('Hello!  S = start, R = restart, Esc = exit',
                                   False, colors[-1])
                screen.blit(surf, (3, 0))
            if not self.training:
                surf = font.render(
                    'UHUUU! KNN start. R = restart, tap on right button - optimal k, use - tap left button'
                    '[Esc] to exit.', False, colors[-1])
                screen.blit(surf, (3, 0))

            for i in range(1, clusters_num + 1):
                surf = font.render(str(i) + ' ', False, colors[i])
                screen.blit(surf, (2 * font_size * (i - 1), font_size))

            if self.k is not None:
                pygame.draw.rect(screen, white, (2 * font_size * clusters_num, font_size, font_size * 16, font_size))
                surf = font.render('Optimal k = ' + str(self.k), False, colors[-1])
                screen.blit(surf, (1000, 700))

            pygame.display.update()

            if len(self.points) > 0:
                for point, cluster in zip(self.points, self.clusters):
                    pygame.draw.circle(screen, colors[int(cluster)], point, 7)


pygame.init()
clusters_num = 6
initial_points_num = clusters_num * 10

font_size = 28
clock = pygame.time.Clock()
white = (255, 255, 255)
colors = []


def start_colors():
    global colors
    colors = []
    for i in range(1, 255):
        colors.append(tuple(np.random.choice(range(256), size=3)))
    colors.append((255, 0, 0))


start_colors()
screen = pygame.display.set_mode((1280, 720))
screen.fill(white)
knn = KNN()
area = Area()
area.run()

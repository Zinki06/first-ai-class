import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pickle
import heapq

class Sensor:
    def __init__(self, range, angles):
        self.range = range  # 센서의 범위
        self.angles = angles  # 센서가 커버하는 각도들

    def sense(self, x, y, direction, grid):
        sensor_data = []
        for angle in self.angles:
            absolute_angle = angle + direction * np.pi / 2
            dx = np.cos(absolute_angle)
            dy = -np.sin(absolute_angle)
            for i in range(1, self.range + 1):
                sense_x = int(x + i * dx)
                sense_y = int(y + i * dy)
                if 0 <= sense_x < grid.shape[1] and 0 <= sense_y < grid.shape[0]:
                    if grid[sense_y, sense_x] != 0.8:  # 도로가 아닌 경우
                        sensor_data.append((i, angle, grid[sense_y, sense_x]))
                        break
                else:
                    sensor_data.append((i, angle, 0))  # 외곽 벽 감지
                    break
            else:
                sensor_data.append((self.range, angle, 1))  # 아무것도 감지되지 않음
        return sensor_data

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = 0  # 0: 오른쪽, 1: 위, 2: 왼쪽, 3: 아래
        self.lidar = Sensor(10, np.linspace(0, 2 * np.pi, 36))  # LiDAR 센서 초기화
        self.radar = Sensor(5, np.linspace(-np.pi / 4, np.pi / 4, 5))  # 레이더 센서 초기화
        self.path = []

    def move(self, action=None):
        if action is not None:
            if action == 0:  # 전진
                if self.direction == 0:
                    self.x += 1
                elif self.direction == 1:
                    self.y -= 1
                elif self.direction == 2:
                    self.x -= 1
                elif self.direction == 3:
                    self.y += 1
            elif action == 1:  # 우회전
                self.direction = (self.direction - 1) % 4
            elif action == 2:  # 좌회전
                self.direction = (self.direction + 1) % 4
        elif self.path:
            next_x, next_y = self.path.pop(0)
            dx, dy = next_x - self.x, next_y - self.y
            if dx > 0:
                self.direction = 0
            elif dx < 0:
                self.direction = 2
            elif dy > 0:
                self.direction = 3
            elif dy < 0:
                self.direction = 1
            self.x, self.y = next_x, next_y
        else:
            if self.direction == 0:
                self.x += 1
            elif self.direction == 1:
                self.y -= 1
            elif self.direction == 2:
                self.x -= 1
            elif self.direction == 3:
                self.y += 1

    def sense(self, grid):
        lidar_data = self.lidar.sense(self.x, self.y, self.direction, grid)
        radar_data = self.radar.sense(self.x, self.y, self.direction, grid)
        return lidar_data, radar_data

    def decide_move(self, lidar_data, radar_data):
        if self.path:
            return  # 경로가 있으면 그대로 따라감

        # 외곽 벽 감지
        if any(d[2] == 0 for d in lidar_data):
            self.direction = (self.direction + 2) % 4  # 반대 방향으로 전환
            return

        # 전방 장애물 감지 (LIDAR)
        forward_distances = [d[0] for d in lidar_data if d[1] < np.pi / 4 or d[1] > 7 * np.pi / 4]
        if min(forward_distances) < 3:
            left_distances = [d[0] for d in lidar_data if np.pi / 4 <= d[1] < 3 * np.pi / 4]
            right_distances = [d[0] for d in lidar_data if 5 * np.pi / 4 <= d[1] < 7 * np.pi / 4]
            if min(left_distances) > min(right_distances):
                self.direction = (self.direction - 1) % 4  # 왼쪽 회전
            else:
                self.direction = (self.direction + 1) % 4  # 오른쪽 회전
            return

        # 전방 장애물 감지 (RADAR)
        if any(d[0] < 3 for d in radar_data):
            self.direction = (self.direction + 1) % 4  # 오른쪽 회전
            return

        # 긴 직선 주행 선호
        if min(forward_distances) > 5:
            return  # 현재 방향 유지

class Environment:
    ROAD_VALUE = 0.8
    BUILDING_VALUE = 0.2
    PARKING_LOT_VALUE = 0.6
    CROSSWALK_VALUE = 0.9
    ROAD_WIDTH = 3
    BUILDING_PROBABILITY = 0.7
    PARKING_LOT_PROBABILITY = 0.3

    def __init__(self, size):
        self.size = size
        self.grid = np.ones((size, size))  # 환경 그리드 초기화
        self.buildings = []  # 건물 목록 초기화
        self.parking_lots = []  # 주차장 목록 초기화
        self.crosswalks = []  # 횡단보도 목록 초기화
        self.create_complex_environment()  # 복잡한 환경 생성
        self.car = self.place_car()  # 자동차 배치
        self.start_point = None  # 시작점 초기화
        self.end_point = None  # 끝점 초기화

    def place_car(self):
        while True:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            if self.grid[y, x] == self.ROAD_VALUE:  # 도로 위에 차량을 배치
                return Car(x, y)

    def create_complex_environment(self):
        # 도로 생성
        for i in range(0, self.size, 10):
            self.grid[i:i + self.ROAD_WIDTH, :] = self.ROAD_VALUE  # 수평 도로
            self.grid[:, i:i + self.ROAD_WIDTH] = self.ROAD_VALUE  # 수직 도로

        # 건물 생성
        for i in range(0, self.size, 10):
            for j in range(0, self.size, 10):
                if np.random.rand() < self.BUILDING_PROBABILITY and self.grid[i, j] != self.ROAD_VALUE:
                    building_size = min(np.random.randint(3, 8), 10 - self.ROAD_WIDTH)
                    self.grid[i:i + building_size, j:j + building_size] = self.BUILDING_VALUE
                    self.buildings.append((j, i, building_size, building_size))

        # 주차장 생성
        for i in range(0, self.size, 10):
            for j in range(0, self.size, 10):
                if np.random.rand() < self.PARKING_LOT_PROBABILITY and self.grid[i, j] != self.ROAD_VALUE:
                    self.grid[i:i + 5, j:j + 5] = self.PARKING_LOT_VALUE
                    self.parking_lots.append((j, i, 5, 5))

        # 횡단보도 생성
        for i in range(0, self.size, 10):
            if np.random.rand() < 0.3:
                crosswalk_pos = np.random.randint(0, self.size - self.ROAD_WIDTH)
                self.grid[i:i + self.ROAD_WIDTH, crosswalk_pos:crosswalk_pos + self.ROAD_WIDTH] = self.CROSSWALK_VALUE
                self.crosswalks.append((crosswalk_pos, i, self.ROAD_WIDTH, self.ROAD_WIDTH))

    def step(self, action=None):
        old_distance = self.get_distance_to_goal()
        self.car.move(action)
        self.car.x = max(0, min(self.car.x, self.size - 1))
        self.car.y = max(0, min(self.car.y, self.size - 1))
        new_distance = self.get_distance_to_goal()

        lidar_data, radar_data = self.car.sense(self.grid)

        # 보상 계산
        if self.grid[self.car.y, self.car.x] != self.ROAD_VALUE:
            reward = -10
            done = True
        elif (self.car.x, self.car.y) == self.end_point:
            reward = 100
            done = True
        elif new_distance < old_distance:
            reward = 1
            done = False
        else:
            reward = -1
            done = False

        return self.get_state(), reward, done, lidar_data, radar_data

    def get_state(self):
        return self.car.x * self.size + self.car.y

    def get_distance_to_goal(self):
        if self.end_point is None:
            return 0
        return abs(self.car.x - self.end_point[0]) + abs(self.car.y - self.end_point[1])

    def reset(self):
        if self.start_point:
            self.car.x, self.car.y = self.start_point
        else:
            self.car = self.place_car()
        self.car.direction = 0
        return self.get_state()

    def set_start_point(self, x, y):
        if self.grid[y, x] == self.ROAD_VALUE:
            self.start_point = (x, y)
            self.car.x, self.car.y = x, y
            return True
        return False

    def set_end_point(self, x, y):
        if self.grid[y, x] == self.ROAD_VALUE:
            self.end_point = (x, y)
            return True
        return False

    def calculate_path(self):
        if not self.start_point or not self.end_point:
            return None

        # 휴리스틱 함수: 현재 위치와 목표 위치 사이의 맨해튼 거리 계산
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])

        # 이웃 노드 찾기: 현재 노드에서 이동 가능한 이웃 노드들을 반환
        def get_neighbors(node):
            x, y = node
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            return [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny, nx] == self.ROAD_VALUE]

        start = self.start_point
        goal = self.end_point

        # A* 알고리즘 초기화
        frontier = []  # 우선순위 큐(힙) 초기화
        heapq.heappush(frontier, (0, start))  # 시작 노드를 큐에 추가
        came_from = {start: None}  # 경로 추적을 위한 딕셔너리
        cost_so_far = {start: 0}  # 시작 노드에서 각 노드까지의 비용

        while frontier:
            current = heapq.heappop(frontier)[1]  # 비용이 가장 적은 노드를 선택

            if current == goal:  # 목표 지점에 도달한 경우
                break

            for next in get_neighbors(current):  # 현재 노드의 이웃 노드들을 탐색
                new_cost = cost_so_far[current] + 1  # 현재 노드까지의 비용에 1을 더함
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost  # 새로운 비용이 더 적으면 갱신
                    priority = new_cost + heuristic(goal, next)  # 우선순위 계산
                    heapq.heappush(frontier, (priority, next))  # 큐에 이웃 노드 추가
                    came_from[next] = current  # 경로 추적

        if goal not in came_from:  # 목표 지점에 도달할 수 없는 경우
            return None

        path = []
        current = goal
        while current != start:  # 경로를 역추적하여 path에 추가
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()  # 시작 지점부터 목표 지점까지의 경로로 변환

        self.car.path = path[1:]  # 시작점은 제외한 경로를 자동차 경로로 설정
        return path

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 상태의 크기
        self.action_size = action_size  # 행동의 크기
        self.q_table = np.zeros((state_size, action_size))  # Q 테이블 초기화
        self.learning_rate = 0.1  # 학습률
        self.discount_factor = 0.99  # 할인율
        self.epsilon = 0.1  # 탐험률

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # 랜덤 행동 선택
        return np.argmax(self.q_table[state])  # Q 값이 최대인 행동 선택

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]  # 현재 상태의 Q 값
        next_max_q = np.max(self.q_table[next_state])  # 다음 상태에서의 최대 Q 값
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * next_max_q)  # Q 값 업데이트
        self.q_table[state, action] = new_q  # Q 테이블 갱신

class AutonomousDrivingSimulation:
    def __init__(self):
        self.env = Environment(50)  # 환경 초기화
        self.agent = QLearningAgent(self.env.size ** 2, 3)  # 3 actions: forward, right, left
        self.fig, self.ax = plt.subplots(figsize=(10, 10))  # 시각화 설정
        self.training_mode = False  # 훈련 모드 초기화
        self.init_plot()  # 초기 플롯 설정

    def init_plot(self):
        self.ax.clear()
        self.ax.imshow(self.env.grid, cmap='gray')
        self.ax.set_title('자율주행 시뮬레이션\n'
                          'T: 훈련 시작/중지, S: 모델 저장, L: 모델 불러오기\n'
                          'R: 초기화, 클릭: 시작/도착점 설정')
        self.ax.axis('off')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)  # 마우스 클릭 이벤트 연결
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)  # 키보드 이벤트 연결

    def update_plot(self):
        self.ax.clear()
        self.ax.imshow(self.env.grid, cmap='gray')

        # 건물 시각화
        for building in self.env.buildings:
            self.ax.add_patch(Rectangle((building[0], building[1]), building[2], building[3],
                                        facecolor='#a0a0a0', edgecolor='none'))
        # 주차장 시각화
        for parking in self.env.parking_lots:
            self.ax.add_patch(Rectangle((parking[0], parking[1]), parking[2], parking[3],
                                        facecolor='#c0c0c0', edgecolor='none'))
        # 횡단보도 시각화
        for crosswalk in self.env.crosswalks:
            self.ax.add_patch(Rectangle((crosswalk[0], crosswalk[1]), crosswalk[2], crosswalk[3],
                                        facecolor='#ffffff', edgecolor='none'))

        # 자동차 시각화
        car_rect = Rectangle((self.env.car.x - 0.5, self.env.car.y - 0.25), 1, 0.5, facecolor='red')
        self.ax.add_patch(car_rect)
        self.ax.add_patch(Circle((self.env.car.x - 0.25, self.env.car.y + 0.25), 0.1, facecolor='black'))
        self.ax.add_patch(Circle((self.env.car.x + 0.25, self.env.car.y + 0.25), 0.1, facecolor='black'))

        # LiDAR 데이터 시각화
        lidar_data, radar_data = self.env.car.sense(self.env.grid)
        for distance, angle, intensity in lidar_data:
            end_x = self.env.car.x + distance * np.cos(angle + self.env.car.direction * np.pi / 2)
            end_y = self.env.car.y - distance * np.sin(angle + self.env.car.direction * np.pi / 2)
            self.ax.plot([self.env.car.x, end_x], [self.env.car.y, end_y], 'g-', alpha=0.3)

        # 레이더 데이터 시각화
        for distance, angle, intensity in radar_data:
            end_x = self.env.car.x + distance * np.cos(angle + self.env.car.direction * np.pi / 2)
            end_y = self.env.car.y - distance * np.sin(angle + self.env.car.direction * np.pi / 2)
            self.ax.plot([self.env.car.x, end_x], [self.env.car.y, end_y], 'r-', alpha=0.5)

        # 시작점 시각화
        if self.env.start_point:
            self.ax.plot(self.env.start_point[0], self.env.start_point[1], 'bo', markersize=10)
            self.ax.text(self.env.start_point[0], self.env.start_point[1], 'S', color='white', ha='center', va='center')
        # 도착점 시각화
        if self.env.end_point:
            self.ax.plot(self.env.end_point[0], self.env.end_point[1], 'ro', markersize=10)
            self.ax.text(self.env.end_point[0], self.env.end_point[1], 'E', color='white', ha='center', va='center')

        # 경로 시각화
        if self.env.car.path:
            path_x, path_y = zip(*self.env.car.path)
            self.ax.plot(path_x, path_y, 'y-', linewidth=2)

        self.ax.set_title('자율주행 시뮬레이션\n'
                          'T: 훈련 시작/중지, S: 모델 저장, L: 모델 불러오기\n'
                          'R: 초기화, 클릭: 시작/도착점 설정')
        self.ax.axis('off')
        self.fig.canvas.draw()

        if self.training_mode:
            state = self.env.get_state()
            action = self.agent.get_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.agent.learn(state, action, reward, next_state)
            if done:
                self.env.reset()
        else:
            if self.env.car.path:
                self.env.car.move()
                if not self.env.car.path:
                    print("목적지에 도달했습니다!")

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if not self.env.start_point:
                if self.env.set_start_point(x, y):
                    print("시작점 설정:", x, y)
            elif not self.env.end_point:
                if self.env.set_end_point(x, y):
                    print("도착점 설정:", x, y)
                    path = self.env.calculate_path()
                    if path:
                        print("경로 계산 완료")
                    else:
                        print("유효한 경로를 찾을 수 없습니다.")
            self.update_plot()

    def save_model(self):
        try:
            with open('q_table.pkl', 'wb') as f:
                pickle.dump(self.agent.q_table, f)
            print("모델 저장 완료")
        except Exception as e:
            print(f"모델 저장 중 오류 발생: {e}")

    def load_model(self):
        try:
            with open('q_table.pkl', 'rb') as f:
                self.agent.q_table = pickle.load(f)
            print("모델 불러오기 완료")
        except FileNotFoundError:
            print("저장된 모델을 찾을 수 없습니다.")
        except Exception as e:
            print(f"모델 불러오기 중 오류 발생: {e}")

    def on_key_press(self, event):
        if event.key == 't':
            self.training_mode = not self.training_mode
            print("훈련 모드:", "켜짐" if self.training_mode else "꺼짐")
        elif event.key == 's':
            self.save_model()
        elif event.key == 'l':
            self.load_model()
        elif event.key == 'r':
            self.reset_simulation()
        self.update_plot()

    def reset_simulation(self):
        self.env = Environment(50)
        self.agent = QLearningAgent(self.env.size ** 2, 3)
        self.training_mode = False
        print("시뮬레이션 초기화 완료")

    def run(self):
        timer = self.fig.canvas.new_timer(interval=100)
        timer.add_callback(self.update_plot)
        timer.start()
        plt.show()

if __name__ == '__main__':
    sim = AutonomousDrivingSimulation()
    sim.run()

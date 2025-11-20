import numpy as np

class GenFeatures:    
    def generate_plane_features():
        center = np.array([30, -30])
        z1 = 5
        z2 = 2.5

        n1 = np.arange(-25, 25+5, 5)
        M = 4*len(n1)
        lms_front = np.zeros((5, M))
        lms_front[0, :] = np.arange(0, M, 1)
        lms_front[1, :M] = center[0]
        lms_front[2, :M] = np.tile(n1, 4)
        lms_front[3, :len(n1)] = z1
        lms_front[3, len(n1):2*len(n1)] = z2
        lms_front[3, 2*len(n1):3*len(n1)] = 0
        lms_front[3, 3*len(n1):4*len(n1)] = -z2
        lms_front[4, :] = 10

        lms_back = lms_front.copy()
        lms_back[0, :] = np.arange(M, 2*M, 1)
        lms_back[1, :] = center[1]
        lms_back[4, :] = 11

        lms_right = lms_front.copy()
        lms_right[0, :] = np.arange(2*M, 3*M, 1)
        lms_right[1, :] = lms_front[2, :]
        lms_right[2, :] = center[0]
        lms_right[4, :] = 12

        lms_left = lms_front.copy()
        lms_left[0, :] = np.arange(3*M, 4*M, 1)
        lms_left[1, :] = lms_front[2, :]
        lms_left[2, :] = center[1]
        lms_left[4, :] = 13

        lms_ground = np.zeros(((5, 121)))
        lms_ground[0, :] = np.arange(4*M, 4*M + (M/4)**2, 1)
        b = np.array([])
        for i in range(len(n1)):
            a = np.linspace(n1[i], n1[i], 11)
            b = np.hstack((b, a))
            
        lms_ground[1, :] = b
        lms_ground[2, :] = np.tile(n1, 11)
        lms_ground[3, :] = -z1
        lms_ground[4, :] = 14
            
        lms = np.empty(0)
        lms = np.hstack((lms_front, lms_back, lms_right, lms_left, lms_ground))
        return lms

    def generate_random_points_on_box(args, start_idx_, class_):
        """
        직육면체의 겉면에 랜덤으로 분포된 점들을 생성하는 함수.
        중심점(center_x, center_y, center_z)을 기준으로 점들이 생성됩니다.

        Parameters:
        center_x (float): 직육면체의 x축 중심
        center_y (float): 직육면체의 y축 중심
        center_z (float): 직육면체의 z축 중심
        length (float): 직육면체의 x축 길이
        width (float): 직육면체의 y축 길이
        height (float): 직육면체의 z축 높이
        num_points (int): 생성할 점의 개수

        Returns:
        numpy.ndarray: 생성된 점들의 3D 좌표 (num_points x 3)
        """
        
        center_x, center_y, center_z, length, width, height, num_points = args
        points = []

        # 직육면체의 각 축에 대해 절반 크기를 계산
        half_length = length / 2
        half_width = width / 2
        half_height = height / 2

        for _ in range(num_points):
            # 면을 랜덤으로 선택 (6개의 면)
            face = np.random.choice(6)
            
            if face == 0:  # x = -half_length (왼쪽 면)
                x = center_x - half_length
                y = np.random.uniform(center_y - half_width, center_y + half_width)
                z = np.random.uniform(center_z - half_height, center_z + half_height)
            elif face == 1:  # x = half_length (오른쪽 면)
                x = center_x + half_length
                y = np.random.uniform(center_y - half_width, center_y + half_width)
                z = np.random.uniform(center_z - half_height, center_z + half_height)
            elif face == 2:  # y = -half_width (앞면)
                x = np.random.uniform(center_x - half_length, center_x + half_length)
                y = center_y - half_width
                z = np.random.uniform(center_z - half_height, center_z + half_height)
            elif face == 3:  # y = half_width (뒷면)
                x = np.random.uniform(center_x - half_length, center_x + half_length)
                y = center_y + half_width
                z = np.random.uniform(center_z - half_height, center_z + half_height)
            elif face == 4:  # z = -half_height (아래면)
                x = np.random.uniform(center_x - half_length, center_x + half_length)
                y = np.random.uniform(center_y - half_width, center_y + half_width)
                z = center_z - half_height
            else:  # z = half_height (윗면)
                x = np.random.uniform(center_x - half_length, center_x + half_length)
                y = np.random.uniform(center_y - half_width, center_y + half_width)
                z = center_z + half_height
            
            new_points = np.array([x, y, z])
            if all(np.linalg.norm(new_points - np.array(p)) >= 0.5 for p in points):
                points.append(new_points)
        points_ = np.array(points)
        points_ = np.vstack((np.arange(start_idx_, start_idx_ + len(points_)), points_.T, np.tile(class_, points_.shape[0])))        
        return points_
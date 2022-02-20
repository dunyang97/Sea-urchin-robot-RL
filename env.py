import pybullet as p
import pybullet_data
import os
import math
import numpy as np
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import time
import matplotlib.pyplot as plt
import math

class RobotEnv2():
    metadata = {'render.modes': ['human']}

    def __init__(self):
        #<editor-fold desc="video测试时需要将这一步全部注释">
        self.xunlian = 2   # 2表示进行 sac测试， 1表示进行训练
        if self.xunlian == 1:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=0.05, cameraYaw=50,
                                     cameraPitch=-30, cameraTargetPosition=[0.6, -0.55, 0.5])
        #</editor-fold desc"全部注释">

        # <editor-fold desc="parameter setting">
        self.torque_bound = [-1, 1]    # 用于velosity、force
        self.action_space = spaces.Box(np.array([self.torque_bound[0]] * 12),np.array([self.torque_bound[1]] * 12))  # 12个关节动作幅度从-1 到 1
        self.state_space = spaces.Box(np.array([-1] * 45), np.array([1] * 45))  # 可观测状态有45个信息（12个关节长度+中心的XYZ坐标）

        self.done = False
        self.dist = 0.0
        self.flag = 0

        self.walk_target_x = 1e3
        self.walk_target_y = 0
        self.action_old = [0 for i in range(12)]
        self.index = []

        # </editor-fold desc"全部注释">

    def step(self, action):

        # <editor-fold desc="执行仿真步前：得到位置body信息">
        self.potential = self.calc_potential(self.robotuid)
        potential_old = self.potential
        self.body_old, _ = p.getBasePositionAndOrientation(self.robotuid)
        self.body_velocity_old, _ = p.getBaseVelocity(self.robotuid)
        # </editor-fold>

        #<editor-fold desc="replanning criteria = 45步">
        for i in range(45):
            for j in range(12):
                self.force = int(25)
                now_measure_length = p.getJointState(self.robotuid, j)[0]
                new_length = self.action_old[j] + action[j] * 0.08
                if now_measure_length >= 0.07:
                    if new_length <= 0.07:
                        p.setJointMotorControl2(self.robotuid, j, p.POSITION_CONTROL,
                                                targetPosition=new_length,
                                                force=self.force)  # 关节控制采用位置控制——虽然样机实验需要通过位置控制，但仿真中要用力控来生成平滑的步态方案
                    else:
                        p.setJointMotorControl2(self.robotuid, j, p.POSITION_CONTROL,
                                                targetPosition=0.07,
                                                force=self.force)
                elif now_measure_length <= 0.005:
                    if new_length >= 0.005:
                        p.setJointMotorControl2(self.robotuid, j, p.POSITION_CONTROL,
                                                targetPosition=new_length, force=self.force)
                    else:
                        p.setJointMotorControl2(self.robotuid, j, p.POSITION_CONTROL,
                                                targetPosition=0.005,
                                                force=self.force)
                else:
                    if new_length >= 0.07:
                        p.setJointMotorControl2(self.robotuid, j, p.POSITION_CONTROL,
                                                targetPosition=0.07,
                                                force=self.force)
                    elif new_length <= 0.005:
                        p.setJointMotorControl2(self.robotuid, j, p.POSITION_CONTROL,
                                                targetPosition=0.005,
                                                force=self.force)
                    else:
                        p.setJointMotorControl2(self.robotuid, j, p.POSITION_CONTROL,
                                                targetPosition=new_length, force=self.force)
            self.action_old = [p.getJointState(self.robotuid, k)[0] for k in range(12)]

            p.stepSimulation()
            if self.xunlian != 1:
                time.sleep(1/240)
        #</editor-fold>

        # <editor-fold desc="计算累计滞留flag">
        self.body_new, _ = p.getBasePositionAndOrientation(self.robotuid)
        dist = self.body_new[1] - self.body_old[1]
        if np.abs(dist) < 0.000001 :
            self.flag += 1
        # print(self.flag)
        # </editor-fold>

        # <editor-fold desc="奖励函数计算细节">
        # reward_1 动起来就给0.5
        self._alive = float(self.alive_bonus(self.body_old, self.body_new))

        # reward_2 向y正向运动奖励
        self.potential = self.calc_potential(self.robotuid)
        progress = (self.body_new[1] - self.body_old[1]) * 150
        # print(progress)

        # reward_3 接触力惩罚
        self.force_penalty = -np.abs(self.contact_force_penalty(self.floor, self.robotuid) * 0.15)
        # print(self.force_penalty)

        # reward_4 进门奖励
        gata_reward = 0
        if p.getBasePositionAndOrientation(self.robotuid)[0][1] > 4.5:
            gata_reward = 5

        # reward_5 触地腿少于3时给大惩罚
        offground_penalty = 0.0
        _, contact_link = self._get_state2(self.robotuid, self.floor)
        if sum(contact_link) < 2:
            offground_penalty = -20
        # print(offground_penalty)

        # reward_6 限制body在x方向的范围
        x_line_penality = 0
        if np.abs(p.getBasePositionAndOrientation(self.robotuid)[0][0]) >= 0.1:
            x_line_penality = -2
        # print(x_line_penality)

        # reward_7 限制机器人重心高度
        height_penalty = 0
        if p.getBasePositionAndOrientation(self.robotuid)[0][2] >= 0.14:
            height_penalty = -0.05
        # print(height_penalty)

        # reward_8 能量计算
        total_action_penality = - np.abs(sum(action) * 0.25)
        # print(total_action_penality)

        # reward_9 向前的速度奖励
        velocity_penality = (p.getBaseVelocity(self.robotuid)[0][1] - self.body_velocity_old[1]) * 1
        # print(velocity_penality)
        # </editor-fold>

        # <editor-fold desc="整合返回信息">
        self.rewards = [progress, self.force_penalty , x_line_penality, gata_reward, offground_penalty, height_penalty, total_action_penality, velocity_penality]

        # print(self.rewards)
        state = self._get_state(self.robotuid, self.floor)
        done = self._get_done(self.flag)
        info = {}  # 用于记录训练过程中的环境信息，便于观察训练状态
        # </editor-fold>

        return np.array(state), sum(self.rewards), done, info

    def reset(self):
        # <editor-fold desc="重要：基本参数、渲染设置">
        p.resetSimulation()  # 重置环境
        p.setTimeStep(1 / 240)  # 默认1/240
        p.setGravity(0, 0, -30)
        p.resetDebugVisualizerCamera(0.8, 0, -25, [0, 0, 0])  # 跟随坐标显示 3是距离， 50， -30是俯仰角，最后还是跟随坐标
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)        #参数0关闭可视化， 参数1打开可视化 = 不写这句话
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 关闭两侧的渲染工具
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)  # 关闭集成显卡：该语句的作用是禁用tinyrenderer，也就是不让CPU上的集成显卡来参与渲染工作。
        # </editor-fold>

        # <editor-fold desc="重要：加载urdf文件">
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.floor = p.loadURDF("plane.urdf", basePosition=[0, 0, 0], useMaximalCoordinates=True)  # 注释说明，将useMaximalCoordinates这个参数置为True会使得运行更快，但是个人猜测，这会使得模型运转时的精度下降
        self.robotuid = p.loadURDF("xin_12zu.urdf", [0, 0, 0.1], [0, 0.187, -0, 1] )    #   四元数 x_1 表示绕z轴旋转，x_2 表示绕y轴旋转，(0.55最合适)，x_3 表示绕x轴旋转， x-4未知 1最合适[-0, -0.55, -0, 1]--5腿全伸状态    3腿鼎立状态--[0, 0.187, -0, 1]最合适
        for i in range(12):
            p.setJointMotorControl2(self.robotuid, i, p.POSITION_CONTROL,targetPosition=0.07,force=30)
        p.changeVisualShape(self.robotuid, 0, rgbaColor=[0.745, 0, 0, 1])
        p.changeVisualShape(self.robotuid, 3, rgbaColor=[0.745, 0, 0, 1])
        # for i in [2, 9, 10]:
        #     p.resetJointState(self.robotuid, i, 0)  # 设置初始构型，保障每一次训练开始都是相通的state

        # <editor-fold desc="添加目标门">
        self.Draw_goal_gate()
        # </editor-fold>

        # self.Draw_goal_and_word(True)
        self.a = 0
        self.time_flag = 0
        # self.online_Draw_init()

        # <editor-fold desc="改变link的颜色,按W后生效">
        self.change_one_color(11)
        # </editor-fold>

        #<editor-fold desc="重要：动力学参数设置">
        p.changeDynamics(self.robotuid, -1, mass=500)
        for i in range(12):
            p.changeDynamics(self.robotuid, i, mass=1, lateralFriction=2, spinningFriction=2, rollingFriction=2,
                             linearDamping=2)
        p.changeDynamics(self.floor, -1, lateralFriction=2)
        # print(p.getDynamicsInfo(self.robotuid,0))
        #</editor-fold>

        # # <editor-fold desc ="球形障碍">
        # np.random.seed(1)
        # for i in range(100):
        #
        #     a_1 = np.random.uniform(-2, 2)
        #     a_2 = np.random.uniform(-2, 2)
        #     a_3 = np.random.uniform(-0.1, 0.05)
        #
        #     shift = [a_1, a_2, a_3]
        #     scale_1 = 0.15
        #     visual_shape_id = p.createVisualShape(
        #         shapeType=p.GEOM_SPHERE,
        #         rgbaColor=[1, 1, 1, 1],
        #         specularColor=[0.4, 0.4, 0],
        #         visualFramePosition=shift,
        #         radius=scale_1, )
        #     collision_shape_id = p.createCollisionShape(
        #         shapeType=p.GEOM_SPHERE,
        #         collisionFramePosition=shift,
        #         radius=scale_1, )
        #     p.createMultiBody(
        #         baseMass=0,
        #         baseCollisionShapeIndex=collision_shape_id,
        #         baseVisualShapeIndex=visual_shape_id,
        #         basePosition=[0, 0, 0],
        #         useMaximalCoordinates=True)
        # # </editor-fold>

        self.env_test = False
        # 调试env开关，如果为True则打开边框开始调试
        self.hand_control = 1  # 1：位置(目前最成熟)  2：速度   3：力    4：手动拖动施加力
        self.action_old =[np.random.uniform(-1, 1) for i in range(12)]
        # print(self.action_old)
        self.action_old = [0 for i in range(12)]

        # <editor-fold desc="添加控制BAR">
        if self.env_test == True:
            self.add_slider()
        # </editor-fold>

        # <editor-fold desc="BUTTON设置">
        if self.env_test == True:
            self.pre_control_button = self.read_slider()[1]
            self.pre_picture_button = self.read_slider()[2]
            self.pre_reset_button = self.read_slider()[3]
            self.pre_link_force_button = self.read_slider()[4]
            self.control_model = 0
            self.picture_model = 0
            self.reset_model = 0
            self.link_force_model = 0
            self.record_force = []
        # </editor-fold>

        observation = self._get_state(self.robotuid,self.floor)
        return np.array(observation), self.hand_control

    def alive_bonus(self, old, new):
        return +0.5 if np.linalg.norm([old[0] - new[0], old[1] - new[1], old[2] - new[2]]) > 0.00009 else -1

    def calc_potential(self, robot):
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - p.getBasePositionAndOrientation(robot)[0][1],
             self.walk_target_x - p.getBasePositionAndOrientation(robot)[0][0]])
        return -self.walk_target_dist / 0.00165

    def Draw_goal_gate(self):
        froms = [[1, 1.5, 0], [-1, 1.5, 0], [-1, 1.5, 1], [1, 1.5, 1]]
        tos = [[-1, 1.5, 0], [-1, 1.5, 1], [1, 1.5, 1], [1, 1.5, 0]]
        froms_1 = [0,0,0]
        to_1 = [1, 0, 0]
        froms_2 = [0, 0, 0]
        to_2 = [0, 1, 0]
        froms_3 = [0, 0, 0]
        to_3 = [0, 0, 1]
        for f, t in zip(froms, tos):
            p.addUserDebugLine(
                lineFromXYZ=f,
                lineToXYZ=t,
                lineColorRGB=[1, 0, 0],  # (1,1,1)白色，(0,1,0)绿色， (0.1.1)蓝色 (1.1.0)黄色 (1,0,0)红色 （0，0，1）深蓝色
                lineWidth=5
            )
        p.addUserDebugLine(
            lineFromXYZ=froms_1,
            lineToXYZ=to_1,
            lineColorRGB=[1, 0, 0],  # (1,1,1)白色，(0,1,0)绿色， (0.1.1)蓝色 (1.1.0)黄色 (1,0,0)红色 （0，0，1）深蓝色
            lineWidth=5
        )
        p.addUserDebugLine(
            lineFromXYZ=froms_2,
            lineToXYZ=to_2,
            lineColorRGB=[0, 1, 0],  # (1,1,1)白色，(0,1,0)绿色， (0.1.1)蓝色 (1.1.0)黄色 (1,0,0)红色 （0，0，1）深蓝色
            lineWidth=5
        )
        p.addUserDebugLine(
            lineFromXYZ=froms_3,
            lineToXYZ=to_3,
            lineColorRGB=[0, 0, 1],  # (1,1,1)白色，(0,1,0)绿色， (0.1.1)蓝色 (1.1.0)黄色 (1,0,0)红色 （0，0，1）深蓝色
            lineWidth=5
        )
        p.addUserDebugText(
            text="X",
            textPosition=[1, 0, 0],
            textColorRGB=[1, 0, 0],
            textSize=2,
        )
        p.addUserDebugText(
            text="Y",
            textPosition=[0, 1, 0],
            textColorRGB=[0, 1, 0],
            textSize=2,
        )
        p.addUserDebugText(
            text="Z",
            textPosition=[0, 0, 1],
            textColorRGB=[0, 0, 1],
            textSize=2,
        )
        p.addUserDebugText(
            text="Destination",
            textPosition=[0, 1, 1],
            textColorRGB=[0, 0, 1],
            textSize=1.2,
        )

    def change_one_color(self, i):
        p.setDebugObjectColor(
            objectUniqueId=self.robotuid,
            linkIndex=i,
            objectDebugColorRGB=[0, 1, 0]
        )
        return

    def _get_state(self, robot,floor):

        # <editor-fold desc="拉杆关节的长度信息（归一化） + 速度信息  12维度 + 12 维">
        state_lagan = []
        for i in range(12):
            length_mid = 0.5 * (0.08)  # 计算杆伸长的中间值
            length = 2 * (p.getJointState(robot, i)[0] - length_mid) / (0.08)  # 计算单位化的变化长度
            state_lagan.append(length)
            v = p.getJointState(robot, i)[1]
            state_lagan.append(0.1 * v)
        # </editor-fold>

        # <editor-fold desc="中心坐标信息:3 + 3 维度">
        body_pos = p.getBasePositionAndOrientation(robot)[0]  # body 3个位置坐标
        body_ori = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(robot)[1])  # body 3个欧拉角
        # </editor-fold>

        # <editor-fold desc="body速度信息（需要进行一次坐标系的转化）3">
        body_velocity = p.getBaseVelocity(robot)[0]  # 线速度
        yaw = body_ori[2]
        rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw), np.cos(-yaw), 0], [0, 0, 1]])
        vx, vy, vz = np.dot(rot_speed, body_velocity)  # 坐标转化到自身坐标
        v = np.array([vx, vy, vz])
        # </editor-fold>

        # <editor-fold desc="触地杆信息  12维度">
        contact_link = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a = p.getContactPoints(bodyA=floor, bodyB=robot)  # ！！注意：此处容易出现问题是刚开始tuple不含元素所以会有 超出索引的错误出现
        flag = int(np.size(a) / 14)  # 用来判断触地杆的触点数目
        if flag != 0:
            for i in range(flag):
                contact_link[(a[i][4])] = 1  # 触地杆为1，离地杆为0
        debug = 0
        if debug == 1:
            print("有(%s)" % (flag), "个点触地，触地杆有(%s)个" % sum(contact_link), "序列为%s" % contact_link)
        self.index_old = self.index
        self.index = self.get_land_bar_index(contact_link, 1)
        # print(index)

        # </editor-fold>

        # state信息一共有45 （24+6+3+12）个

        observation = list(state_lagan) + list(body_pos) + list(body_ori) + list(v) + list(contact_link)
        # print(contact_link)
        return observation

    def color_land_bar(self, floor, robot):  # video中给触地杆上色
        # <editor-fold desc="触地杆信息  12维度">
        contact_link = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a = p.getContactPoints(bodyA=floor, bodyB=robot)  # ！！注意：此处容易出现问题是刚开始tuple不含元素所以会有 超出索引的错误出现
        flag = int(np.size(a) / 14)  # 用来判断触地杆的触点数目
        if flag != 0:
            for i in range(flag):
                contact_link[(a[i][4])] = 1  # 触地杆为1，离地杆为0
        debug = 0
        if debug == 1:
            print("有(%s)" % (flag), "个点触地，触地杆有(%s)个" % sum(contact_link), "序列为%s" % contact_link)
        self.index_old = self.index
        self.index = self.get_land_bar_index(contact_link, 1)
        # print(index)
        for i in self.index:
            p.changeVisualShape(robot, i, rgbaColor = [1, 0, 0, 1])
        for i in self.index_old:
            p.changeVisualShape(robot, i, rgbaColor=[1, 1, 1, 1])

    def _get_state2(self, robot, floor):

        # <editor-fold desc="拉杆关节的长度信息（归一化） + 速度信息  12维度 + 12 维">
        state_lagan = []
        for i in range(12):
            length_mid = 0.5 * (0.08)  # 计算杆伸长的中间值
            length = 2 * (p.getJointState(robot, i)[0] - length_mid) / (0.08)  # 计算单位化的变化长度
            state_lagan.append(length)
            v = p.getJointState(robot, i)[1]
            state_lagan.append(0.1 * v)
        # </editor-fold>

        # <editor-fold desc="中心坐标信息:3 + 3 维度">
        body_pos = p.getBasePositionAndOrientation(robot)[0]  # body 3个位置坐标
        body_ori = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(robot)[1])  # body 3个欧拉角
        # </editor-fold>

        # <editor-fold desc="body速度信息（需要进行一次坐标系的转化）3">
        body_velocity = p.getBaseVelocity(robot)[0]  # 线速度
        yaw = body_ori[2]
        rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw), np.cos(-yaw), 0], [0, 0, 1]])
        vx, vy, vz = np.dot(rot_speed, body_velocity)  # 坐标转化到自身坐标
        v = np.array([vx, vy, vz])
        # </editor-fold>

        # <editor-fold desc="触地杆信息  12维度">
        contact_link = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a = p.getContactPoints(bodyA=floor, bodyB=robot)  # ！！注意：此处容易出现问题是刚开始tuple不含元素所以会有 超出索引的错误出现
        flag = int(np.size(a) / 14)  # 用来判断触地杆的触点数目
        if flag != 0:
            for i in range(flag):
                contact_link[(a[i][4])] = 1  # 触地杆为1，离地杆为0
        debug = 0
        if debug == 1:
            print("有(%s)" % (flag), "个点触地，触地杆有(%s)个" % sum(contact_link), "序列为%s" % contact_link)
        # </editor-fold>

        # state信息一共有45 （24+6+3+12）个

        observation = list(state_lagan) + list(body_pos) + list(body_ori) + list(v) + list(contact_link)

        return observation,contact_link

    def get_land_bar_index(self, lst = None, item = 2):
        tmp = []
        tag = 0
        for i in lst:
            if i == item:
                tmp.append(tag)
            tag += 1
        return tmp

    def _get_done(self, flag):
        ## 开放机器人运动的方向，鼓励它向四面八方奔跑！！
        # if p.getBasePositionAndOrientation(self.robotuid)[0][1] < -0.1:
        #     self.done = True
        # elif p.getBasePositionAndOrientation(self.robotuid)[0][0] < -0.2:
        #     self.done = True
        # elif p.getBasePositionAndOrientation(self.robotuid)[0][0] > 0.2:
        #     self.done = True
        if flag > 100:
            self.done = True
            self.flag = 0
        else:  ## 一定要有这一句！！
            self.done = False
        return self.done

    def close(self):
        p.disconnect()
        return

    def contact_force_penalty(self,robot,floor):
        a = p.getContactPoints(bodyA=floor, bodyB=robot)  # ！！注意：此处容易出现问题是刚开始tuple不含元素所以会有 超出索引的错误出现
        contact_force_penalty = 0
        link_flag = 0
        flag = int(np.size(a) / 14)  # 用来判断触地杆的触点数目
        if flag != 0:
            for i in range(flag):
                if a[i][10] < 100:
                    contact_force_penalty += a[i][10]
                else:
                    # print("力太大了，%f", a[i][10])
                    pass
        else:
            contact_force_penalty = 100
        return np.abs(contact_force_penalty)

    def Draw_Link_force_record_init(self, link_id):
        a = p.getContactPoints(bodyA=self.floor, bodyB=self.robotuid)  # ！！注意：此处容易出现问题是刚开始tuple不含元素所以会有 超出索引的错误出现
        link_flag = 0
        self.time_flag += 1
        self.new_force = 0
        flag = int(np.size(a) / 14)  # 用来判断触地杆的触点数目
        if self.time_flag % 5 == 0:
            if flag != 0:
                for i in range(flag):
                    if a[i][4] == link_id:
                        self.record_force.append(a[i][10])
                        link_flag += 1
                        self.new_force = a[i][10]
                if link_flag == 0:  # 如果触地杆里没有link_id杆，则记入0
                    self.record_force.append(0)
                    self.new_force = 0
            else:  # 如果没有触地杆，计入0
                self.record_force.append(0)
                self.new_force = 0
        return

    def Draw_Link_force_record(self, ):
        data = np.array(self.record_force)
        x = [i for i in range(np.size(data))]
        plt.figure(1)
        plt.subplot(1, 1, 1)
        plt.xlabel("time steps")
        plt.ylabel("normalForce ")
        # plt.imshow(a)

        plt.title("link contact force")
        # plt.axis("off")
        plt.plot(x, data, '-r')
        plt.show()
        # print(data)
        return

    def online_Draw_init(self):  # 实时绘制受力图初始化设置
        plt.ion()
        return


if __name__ == "__main__":
    env = RobotEnv2()
    for i in range(1):
        observation = env.reset()
        ep_r = 0
        for t in range(50):

            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            ep_r += reward
            time.sleep(1 / 240)  # 动画演示的时间被挂起/延迟 但仿真时长未变 1/240 时间挂起长度最合适

            # print(action)

            print(reward, ep_r)


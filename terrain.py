import pybullet as p
import pybullet_data as pd
import math
import time
import random


class terrain_plane():
    def __init__(self):

        #  #
        p.setAdditionalSearchPath(pd.getDataPath())
        self.textureId = -1
        self.heightfieldSource = 0

    def complex_terrain(self, open=1, row=256, colums = 256, height = 0.05, meshscale=None, position=None,
                        oritation=None, color=None):

        if color is None:
            color = [1, 1, 1, 1]
        if oritation is None:
            oritation = [0, 0, 0, 1]
        if position is None:
            position = [0, 0, 0]
        if meshscale is None:
            meshscale = [.05, .05, 1]

        self.useProgrammatic = open  # 加载平面complex terrain

        # <editor-fold desc="加载水波">
        if self.heightfieldSource == self.useProgrammatic:
            self.numHeightfieldRows = row  # 水波的长宽高
            self.numHeightfieldColumns = colums
            self.heightPerturbationRange = height     # 水波的高度
            self.heightfieldData = [0] * self.numHeightfieldRows * self.numHeightfieldColumns
            random.seed(10)
            for j in range(int(self.numHeightfieldColumns / 2)):
                for i in range(int(self.numHeightfieldRows / 2)):
                    height = random.uniform(0, self.heightPerturbationRange)
                    self.heightfieldData[2 * i + 2 * j * self.numHeightfieldRows] = height  # 生成地形数据方法:拉长256*256的矩阵,为了简化数据量,相邻的四个点采用同样的高度,且采用一维数据存储
                    self.heightfieldData[2 * i + 1 + 2 * j * self.numHeightfieldRows] = height
                    self.heightfieldData[2 * i + (2 * j + 1) * self.numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 + (2 * j + 1) * self.numHeightfieldRows] = height

            #  meshScale=[.5,.05,1] x方向被拉长, y方向被拉长,高度方向拉长
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=meshscale,
                                                  heightfieldTextureScaling=(self.numHeightfieldRows - 1) / 2,
                                                  heightfieldData=self.heightfieldData,
                                                  numHeightfieldRows=self.numHeightfieldRows,
                                                  numHeightfieldColumns=self.numHeightfieldColumns)  #fileName="heightmaps/wm_height_out.png" 使用png的贴图


            # textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")
            # terrain = p.createMultiBody(0, terrainShape)
            # p.changeVisualShape(terrain, -1, textureUniqueId=textureId)

            terrain = p.createMultiBody(0, terrainShape)  # 0是固定, 1是动的
            p.resetBasePositionAndOrientation(terrain, position, oritation)  # [0, 0, 0]为起始点  [0,0,0,1] 为方向
            p.changeVisualShape(terrain, -1, rgbaColor=color)  # 改变颜色

            return terrain


        # </editor-fold>

    def no_png_mountain_terrain(self, open=1, meshscale=None, position=None, oritation=None, lidu = 128, color=None):
        if color is None:
            color = [1, 1, 1, 1]
        if oritation is None:
            oritation = [0, 0, 0, 1]
        if position is None:
            position = [0, 0, 0]
        if meshscale is None:
            meshscale = [0.5, 0.5, 2.5]
        useDeepLocoCSV = open    # 0 开启无贴图地形 sacle默认合适尺寸
        #<editor-fold desc="加载无贴图地形">
        if self.heightfieldSource == useDeepLocoCSV:
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=meshscale,
                                                  fileName="heightmaps/ground6.txt", heightfieldTextureScaling=lidu)  # 更改粒度
            terrain = p.createMultiBody(0, terrainShape)
            p.resetBasePositionAndOrientation(terrain,position ,oritation)
            p.changeVisualShape(terrain, -1, rgbaColor=color)  # 改变颜色
        return terrain
        #</editor-fold>

    def png_mountain_terrain(self, open=1, meshscale=None, position=None, color=None, lidu = 18):
        if position is None:
            position = [0, 0, 0.5]
        if color is None:
            color = [1, 1, 1, 1]
        if meshscale is None:
            meshscale = [.1, .1, 24]
        useTerrainFromPNG = open  # 0 是开启贴图地形   sacle默认合适尺寸
        #<editor-fold desc="加载有贴图地形">

        #加载贴图的实物
        self.numHeightfieldRows = 256  # 水波的长宽高
        self.numHeightfieldColumns = 256
        self.heightPerturbationRange = 0.05     # 水波的高度
        self.heightfieldData = [0] * self.numHeightfieldRows * self.numHeightfieldColumns
        random.seed(10)
        for j in range(int(self.numHeightfieldColumns / 2)):
            for i in range(int(self.numHeightfieldRows / 2)):
                height = random.uniform(0, self.heightPerturbationRange)
                self.heightfieldData[2 * i + 2 * j * self.numHeightfieldRows] = height  # 生成地形数据方法:拉长256*256的矩阵,为了简化数据量,相邻的四个点采用同样的高度,且采用一维数据存储
                self.heightfieldData[2 * i + 1 + 2 * j * self.numHeightfieldRows] = height
                self.heightfieldData[2 * i + (2 * j + 1) * self.numHeightfieldRows] = height
                self.heightfieldData[2 * i + 1 + (2 * j + 1) * self.numHeightfieldRows] = height

        # print(self.numHeightfieldRows, self.heightfieldData, self.numHeightfieldRows, self.numHeightfieldColumns)
        if self.heightfieldSource == useTerrainFromPNG:
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=meshscale,
                                                  fileName="heightmaps/wm_height_out.png", heightfieldTextureScaling=lidu)
            textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")
            terrain = p.createMultiBody(0, terrainShape)
            p.changeVisualShape(terrain, -1, textureUniqueId=textureId)
            p.changeVisualShape(terrain, -1, rgbaColor=color)  # 改变颜色
            p.resetBasePositionAndOrientation(terrain, position, [0, 0, 0, 1])
            terrainShape2 = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, flags=0, meshScale=[.05, .05, 1],
                                                   heightfieldTextureScaling=(self.numHeightfieldRows- 1) / 2,    #
                                                   heightfieldData= self.heightfieldData,#,
                                                   numHeightfieldRows=self.numHeightfieldRows,
                                                   numHeightfieldColumns=self.numHeightfieldColumns,
                                                   replaceHeightfieldIndex=terrainShape)# 没有这一句贴图和地形就匹配不上了,而且

        #</editor-fold>

    def add_box(self):
        BOX = False
        if BOX==True:
            #<editor-fold desc="生成方块 3*3*3">
            sphereRadius = 0.05
            colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
            colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                              halfExtents=[sphereRadius, sphereRadius, sphereRadius])
            mass = 1
            visualShapeId = -1

            link_Masses = [1]
            linkCollisionShapeIndices = [colBoxId]
            linkVisualShapeIndices = [-1]
            linkPositions = [[0, 0, 0.11]]
            linkOrientations = [[0, 0, 0, 1]]
            linkInertialFramePositions = [[0, 0, 0]]
            linkInertialFrameOrientations = [[0, 0, 0, 1]]
            indices = [0]
            jointTypes = [p.JOINT_REVOLUTE]
            axis = [[0, 0, 1]]

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        basePosition = [
                            i * 5 * sphereRadius, j * 5 * sphereRadius, 1 + k * 5 * sphereRadius + 1
                        ]
                        baseOrientation = [0, 0, 0, 1]
                        if (k & 2):
                            sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, basePosition,
                                                          baseOrientation)
                        else:
                            sphereUid = p.createMultiBody(mass,
                                                          colBoxId,
                                                          visualShapeId,
                                                          basePosition,
                                                          baseOrientation,
                                                          linkMasses=link_Masses,
                                                          linkCollisionShapeIndices=linkCollisionShapeIndices,
                                                          linkVisualShapeIndices=linkVisualShapeIndices,
                                                          linkPositions=linkPositions,
                                                          linkOrientations=linkOrientations,
                                                          linkInertialFramePositions=linkInertialFramePositions,
                                                          linkInertialFrameOrientations=linkInertialFrameOrientations,
                                                          linkParentIndices=indices,
                                                          linkJointTypes=jointTypes,
                                                          linkJointAxis=axis)

                        p.changeDynamics(sphereUid,
                                         -1,
                                         spinningFriction=0.001,
                                         rollingFriction=0.001,
                                         linearDamping=0.0)
                        for joint in range(p.getNumJoints(sphereUid)):
                            p.setJointMotorControl2(sphereUid, joint, p.VELOCITY_CONTROL, targetVelocity=1, force=10)
            #</editor-fold>

    def deformable_complex_terrain(self, open=1, row=256, colums = 256, height_1 =0.04, meshscale=None, position=None,
                        oritation=None, color=None):

        if color is None:
            color = [1, 1, 1, 1]
        if oritation is None:
            oritation = [0, 0, 0, 1]
        if position is None:
            position = [0, 0, 0]
        if meshscale is None:
            meshscale = [.05, .05, 1]
        #<editor-fold desc="流动地形">
        if open == 1:
            updateHeightfield = True
        else:
            updateHeightfield = False


        self.numHeightfieldRows = row  # 水波的长宽高
        self.numHeightfieldColumns = colums
        self.heightPerturbationRange = 0     # 水波的高度
        self.heightfieldData = [0] * self.numHeightfieldRows * self.numHeightfieldColumns
        random.seed(10)
        for j in range(int(self.numHeightfieldColumns / 2)):
            for i in range(int(self.numHeightfieldRows / 2)):
                height = random.uniform(0, self.heightPerturbationRange)
                self.heightfieldData[2 * i + 2 * j * self.numHeightfieldRows] = height  # 生成地形数据方法:拉长256*256的矩阵,为了简化数据量,相邻的四个点采用同样的高度,且采用一维数据存储
                self.heightfieldData[2 * i + 1 + 2 * j * self.numHeightfieldRows] = height
                self.heightfieldData[2 * i + (2 * j + 1) * self.numHeightfieldRows] = height
                self.heightfieldData[2 * i + 1 + (2 * j + 1) * self.numHeightfieldRows] = height

        #  meshScale=[.5,.05,1] x方向被拉长, y方向被拉长,高度方向拉长
        self.terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=meshscale,
                                              heightfieldTextureScaling=(self.numHeightfieldRows - 1) / 2,
                                              heightfieldData=self.heightfieldData,
                                              numHeightfieldRows=self.numHeightfieldRows,
                                              numHeightfieldColumns=self.numHeightfieldColumns, )  #fileName="heightmaps/wm_height_out.png" 使用png的贴图


        # textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")
        # terrain = p.createMultiBody(0, terrainShape)
        # p.changeVisualShape(terrain, -1, textureUniqueId=textureId)

        terrain = p.createMultiBody(0, self.terrainShape)  # 0是固定, 1是动的
        p.resetBasePositionAndOrientation(terrain, position, oritation)  # 为起始点   为方向
        p.changeVisualShape(terrain, -1, rgbaColor=color)  # 改变颜色

        while True:
            keys = p.getKeyboardEvents()

            self.numHeightfieldRows = row  # 水波的长宽高
            self.numHeightfieldColumns = colums
            self.heightPerturbationRange = height_1  # 水波的高度
            self.heightfieldData = [0] * self.numHeightfieldRows * self.numHeightfieldColumns

            if updateHeightfield:
                for j in range(int(self.numHeightfieldColumns / 2)):
                    for i in range(int(self.numHeightfieldRows / 2)):
                        height = random.uniform(0, self.heightPerturbationRange)  # +math.sin(time.time())
                        self.heightfieldData[2 * i + 2 * j * self.numHeightfieldRows] = height
                        self.heightfieldData[2 * i + 1 + 2 * j * self.numHeightfieldRows] = height
                        self.heightfieldData[2 * i + (2 * j + 1) * self.numHeightfieldRows] = height
                        self.heightfieldData[2 * i + 1 + (2 * j + 1) * self.numHeightfieldRows] = height
                # GEOM_CONCAVE_INTERNAL_EDGE may help avoid getting stuck at an internal (shared) edge of the triangle/heightfield.
                # GEOM_CONCAVE_INTERNAL_EDGE is a bit slower to build though.
                # flags = p.GEOM_CONCAVE_INTERNAL_EDGE
                flags = 0
                terrainShape2 = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, flags=flags, meshScale=meshscale,
                                                       heightfieldTextureScaling=(self.numHeightfieldRows - 1) / 2,
                                                       heightfieldData=self.heightfieldData, numHeightfieldRows=self.numHeightfieldRows,
                                                       numHeightfieldColumns=self.numHeightfieldColumns,
                                                       replaceHeightfieldIndex=self.terrainShape)

            # print(keys)
            # getCameraImage note: software/TinyRenderer doesn't render/support heightfields!
            # p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            # time.sleep(0.01)
        #</editor-fold>

    def deformable_complex_terrain_2(self, open=1, row=256, colums = 256, height_1 =0.04, meshscale=None, position=None,
                        oritation=None, color=None):

        if color is None:
            color = [1, 1, 1, 1]
        if oritation is None:
            oritation = [0, 0, 0, 1]
        if position is None:
            position = [0, 0, 0]
        if meshscale is None:
            meshscale = [.05, .05, 1]
        #<editor-fold desc="流动地形">
        if open == 0:
            updateHeightfield = True
        else:
            updateHeightfield = False


        self.numHeightfieldRows = row  # 水波的长宽高
        self.numHeightfieldColumns = colums
        self.heightPerturbationRange = 0     # 水波的高度
        self.heightfieldData = [0] * self.numHeightfieldRows * self.numHeightfieldColumns
        random.seed(10)
        for j in range(int(self.numHeightfieldColumns / 2)):
            for i in range(int(self.numHeightfieldRows / 2)):
                height = random.uniform(0, self.heightPerturbationRange)
                self.heightfieldData[2 * i + 2 * j * self.numHeightfieldRows] = height  # 生成地形数据方法:拉长256*256的矩阵,为了简化数据量,相邻的四个点采用同样的高度,且采用一维数据存储
                self.heightfieldData[2 * i + 1 + 2 * j * self.numHeightfieldRows] = height
                self.heightfieldData[2 * i + (2 * j + 1) * self.numHeightfieldRows] = height
                self.heightfieldData[2 * i + 1 + (2 * j + 1) * self.numHeightfieldRows] = height

        #  meshScale=[.5,.05,1] x方向被拉长, y方向被拉长,高度方向拉长
        self.terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=meshscale,
                                              heightfieldTextureScaling=(self.numHeightfieldRows - 1) / 2,
                                              heightfieldData=self.heightfieldData,
                                              numHeightfieldRows=self.numHeightfieldRows,
                                              numHeightfieldColumns=self.numHeightfieldColumns, )  #fileName="heightmaps/wm_height_out.png" 使用png的贴图


        # textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")
        # terrain = p.createMultiBody(0, terrainShape)
        # p.changeVisualShape(terrain, -1, textureUniqueId=textureId)

        terrain = p.createMultiBody(0, self.terrainShape)  # 0是固定, 1是动的
        p.resetBasePositionAndOrientation(terrain, position, oritation)  # 为起始点   为方向
        p.changeVisualShape(terrain, -1, rgbaColor=color)  # 改变颜色

        keys = p.getKeyboardEvents()

        self.numHeightfieldRows = row  # 水波的长宽高
        self.numHeightfieldColumns = colums
        self.heightPerturbationRange = height_1  # 水波的高度
        self.heightfieldData = [0] * self.numHeightfieldRows * self.numHeightfieldColumns

        if updateHeightfield:
            for j in range(int(self.numHeightfieldColumns / 2)):
                for i in range(int(self.numHeightfieldRows / 2)):
                    height = random.uniform(0, self.heightPerturbationRange)  # +math.sin(time.time())
                    self.heightfieldData[2 * i + 2 * j * self.numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 + 2 * j * self.numHeightfieldRows] = height
                    self.heightfieldData[2 * i + (2 * j + 1) * self.numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 + (2 * j + 1) * self.numHeightfieldRows] = height
            # GEOM_CONCAVE_INTERNAL_EDGE may help avoid getting stuck at an internal (shared) edge of the triangle/heightfield.
            # GEOM_CONCAVE_INTERNAL_EDGE is a bit slower to build though.
            # flags = p.GEOM_CONCAVE_INTERNAL_EDGE
            flags = 0
            terrainShape2 = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, flags=flags, meshScale=meshscale,
                                                   heightfieldTextureScaling=(self.numHeightfieldRows - 1) / 2,
                                                   heightfieldData=self.heightfieldData, numHeightfieldRows=self.numHeightfieldRows,
                                                   numHeightfieldColumns=self.numHeightfieldColumns,
                                                   replaceHeightfieldIndex=self.terrainShape)

            # print(keys)
            # getCameraImage note: software/TinyRenderer doesn't render/support heightfields!
            # p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            # time.sleep(0.01)
        #</editor-fold>

if __name__ == "__main__":

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 关闭两侧的渲染工具
    plane = terrain_plane()
    p.resetSimulation()
    p.setGravity(0, 0, -30)
    # plane.complex_terrain()
    plane.deformable_complex_terrain()
    # plane.png_mountain_terrain(0)
    # plane.no_png_mountain_terrain(0)
    while True:
        p.stepSimulation()
        time.sleep(0.001)

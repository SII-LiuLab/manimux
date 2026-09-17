import pybullet as p,time,os,glob
f='generated_urdf/二指数采夹爪URDF/从夹爪组件0720/从夹爪组件0720.urdf'
# resolve mesh paths relative to this project
r=p.connect(p.GUI); p.setGravity(0,0,-9.81); p.loadURDF(f,useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
print('URDF loaded; close window to exit')
while p.isConnected(): time.sleep(0.1)

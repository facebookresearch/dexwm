# import mujoco
# x = mujoco.MjSpec.from_file("robosuite/models/assets/grippers/allegro_hand_right.urdf")
# for body in x.bodies:
#     body.explicitinertial = True
# breakpoint()
# mjmodel = x.compile()
# print(x.to_xml())
import mujoco
x = mujoco.MjSpec.from_file("/home/joanne/robosuite/robosuite/models/assets/grippers/allegro_hand_right.urdf")
x.compile() # first compilation ignores URDF inertias and infers from geoms
for body in x.bodies:
    body.explicitinertial = True
x.compiler.inertiafromgeom = 2
x.compiler.autolimits=True
breakpoint()
# mjmodel = x.compile() # not sure if you need this line, please try with and without :)
print(x.to_xml())
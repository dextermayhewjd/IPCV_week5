'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Lab Sheet 1
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def o3d_2_nparray(img):
    '''
    Change from open3d images to numpy array
    '''
    img = np.asarray(img)
    if len(img.shape) > 2:
        # rgb 2 bgr
        img = img[..., ::-1]
    img = np.ascontiguousarray(img)   # make it contiguously stored in memory, otherwise errors triggered when drawing circles with cv2.
    return img

# print("here", flush=True)
if __name__ == '__main__': 
    bDisplayAxis = True

    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    name_list.append('sphere_r')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=2)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, -4],
                     [0, 1, 0, 2],
                     [0, 0, 1, -2],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    name_list.append('sphere_g')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=2)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, -7],
                     [0, 1, 0, 2],
                     [0, 0, 1, 3],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    name_list.append('sphere_b')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, 4],
                     [0, 1, 0, 1.5],
                     [0, 0, 1, 4],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])


    #########################################
    '''
    Question 2: Add another sphere to the scene

    Write your code here to define another sphere
    in world coordinate frame
    '''
    #########################################


    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if bDisplayAxis:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * 45*5/180.
    # theta = 0.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * 80/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * 45*5/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]


    ###################################################
    '''
    Extra Question: Add an extra camera view here

    Write your code here to define camera poses
    '''
    ###################################################


    # set camera intrinsics
    K = o3d.camera.PinholeCameraIntrinsic(640, 480, 415.69219381653056, 415.69219381653056, 319.5, 239.5)
    # print(K)
    # print(K.intrinsic_matrix.shape)
    print('Pose_0:\n', H0_wc)
    print('Pose_1:\n', H1_wc)
    print('Intrinsics\n:', K.intrinsic_matrix)
    # o3d.io.write_pinhole_camera_intrinsic("test.json", K)


    ############################################################
    '''
    Question 4 & 5: Add sphere w.r.t. camera coordinate frames

    Write your code here to define the sphere
    in the camera coordinate frame
    '''
    ############################################################


    # Rendering RGB-D frames given camera poses
    render = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    for m, rgb, name in zip(obj_meshes, RGB_list, name_list):
        colour = o3d.visualization.rendering.MaterialRecord()
        colour.base_color = [rgb[0], rgb[1], rgb[2], 1.0]
        colour.shader = "defaultLit"
        render.scene.add_geometry(name, m, colour)
    
    for (H_wc, name, dname) in render_list:
        render.setup_camera(K, H_wc)
        img = o3d_2_nparray(render.render_to_image())
        cv2.imwrite(name, img)
        dep = o3d_2_nparray(render.render_to_depth_image(z_in_view_space=True))
        cv2.imwrite(dname, dep)


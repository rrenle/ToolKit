import numpy as np
import torch
import cv2

def get_affine_transform(center, # (2, )
                         scale, # (2, )
                         rot, # 0
                         output_size, # tuple, 例如 (832, 512)
                         shift=np.array([0, 0], dtype=np.float32), # 验证时，使用此默认
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180 # 度转化为弧度
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32) # 注意这个地方没有旋转，这样的话，仿射变换后就有旋转的效果了。因为这个问题卡了好久！！

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    # 根据原始图像中三个点的坐标 和 这三个点变换后的对应的坐标，计算2*3仿射变换矩阵
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_final_preds(poses, center, scale, heatmap_size):
    trans = get_affine_transform(center, scale, 0, heatmap_size, inv=1)
    final_poses = poses.copy()
    shape = poses[:, :, :2].shape
    poses = poses[:, :, :2].reshape(-1, 2)
    poses = np.concatenate((poses, poses[:, 0:1]*0+1), axis=1)
    poses = (np.dot(trans, poses.T)[:2]).T.reshape(shape)
    final_poses[:, :, :2] = poses
    return final_poses


def affine_joints(joints, mat):
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)


# 获取对称的点，根据前面两个点计算第三个点
# 不考虑shift的情况下，向上偏移多少，就向左偏移多少
def get_3rd_point(a, b):
    direct = a - b
    # 向上偏移了多少，就向左偏移多少；向左偏移了多少，就向上偏移多少。
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

# 获取旋转后的点，通过src_point （逆时针）旋转一定角度即可获得一个点。
def get_dir(src_point, rot_rad):
    # 通过两角和的正弦和余弦公式推导数
    # cos(α+β)=cosα·cosβ - sinα·sinβ
    # sin(α+β)=sinα.cosβ + cosα·sinβ
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn # rcos(α+β)展开计算可得
    src_result[1] = src_point[0] * sn + src_point[1] * cs # rsin(α+β)展开计算可得

    return src_result

def get_multi_scale_size(image, input_size, current_scale, min_scale, size_divisibility=64):
    h, w, _ = image.shape
    # center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)]) # (2,) 向上取整
    center = np.array([w / 2.0, h / 2.0])
    # calculate the size for min_scale
    # 当 min_scale=1, input_size=512, size_divisibility=64时, min_input_size=512
    min_input_size = int((min_scale * input_size + size_divisibility-1)//size_divisibility * size_divisibility)
    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(
            int((min_input_size/w*h+size_divisibility-1)//size_divisibility*size_divisibility)*current_scale/min_scale
        )
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else: # w > h
        h_resized = int(min_input_size * current_scale / min_scale) # 例512
        w_resized = int( # 例832
            int((min_input_size/h*w+size_divisibility-1)//size_divisibility*size_divisibility)*current_scale/min_scale
        )
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    return (w_resized, h_resized), center, np.array([scale_w, scale_h])

def resize_align_multi_scale(image, input_size, current_scale, min_scale, size_divisibility=64):
    # image: (h, w, 3)  e.g.(480, 640, 3) (425, 640, 3) (1000, 723, 3 )
    size_resized, center, scale = get_multi_scale_size(
        image, input_size, current_scale, min_scale, size_divisibility
    )
    # size_resized: (w=704, h=512), (w=832, h=512), (w=512, h=768)
    # 512/480*640=682.66,所以再填充为64的倍数？感觉应该是这样的。（以上只是自己的猜测,不过验证了以上三个例子都没错）
    trans = get_affine_transform(center, scale, 0, size_resized)
    image_resized = cv2.warpAffine(
        image,
        trans,
        size_resized
    )
    # cv2.imwrite('visualize/image.jpg',image)
    # cv2.imwrite('visualize/image_resized.jpg',image_resized)
    return image_resized, center, scale

def up_interpolate(x, size, mode='bilinear', aligh_corners=True):
    H=x.size()[2]
    W=x.size()[3]
    scale_h=int(size[0]/H)
    scale_w=int(size[1]/W)
    inter_x= torch.nn.functional.interpolate(x,size=[size[0]-scale_h+1,size[1]-scale_w+1],align_corners=aligh_corners,mode=mode)
    padd= torch.nn.ReplicationPad2d((0,scale_w-1,0,scale_h-1))
    return padd(inter_x)

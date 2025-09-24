import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainDecompOptions
from utils.plot_script import *

from networks.modules import *
from networks.trainers import DecompTrainerV3
from data.dataset import MotionDatasetV2
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator


def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == '__main__':
    parser = TrainDecompOptions()
    opt = parser.parse()

    opt.device = "cpu"  # 强制使用CPU
    # opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))

    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    # 编码相对运动，解码完整姿态
    # 编码阶段专注于学习运动本质：
    #     去除根节点的全局变换，让模型专注于学习相对肢体运动规律
    #     避免模型被根节点的大幅度位移干扰
    #     学习到的运动表示更具泛化性
    # 解码阶段需要完整输出：
    #     最终生成的运动必须是完整的、可用的
    #     包含根节点位置才能在实际3D空间中定位
    #     下游应用（如动画系统）需要完整的姿态数据
    # 我们注意到在数据预处理中，原始姿态数据的维度是dim_pose，而在MovementConvEncoder中，输入大小是dim_pose - 4。
    # 这通常意味着原始姿态数据中有4个维度被去除了。在HumanML3D和KIT-ML数据集中，原始姿态表示可能包含全局位置或方向信息，而局部运动信息则不需要这些。
    # 具体来说，在HumanML3D数据集中，每个姿态向量可能包含根节点的全局位置和旋转（例如3个位置和1个旋转，或者4个代表全局信息的数值），而其余维度代表局部关节信息。
    # 在训练运动编码器时，我们可能只关心局部运动，因此去除了全局信息（4个维度），只对局部姿态进行编码。
    movement_enc = MovementConvEncoder(input_size=dim_pose - 4,
                                       hidden_size=opt.dim_movement_enc_hidden,
                                       output_size=opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(input_size=opt.dim_movement_latent,
                                       hidden_size=opt.dim_movement_dec_hidden,
                                       output_size=dim_pose)

    all_params = 0
    pc_mov_enc = sum(param.numel() for param in movement_enc.parameters())
    print(movement_enc)
    print("Total parameters of prior net: {}".format(pc_mov_enc))
    all_params += pc_mov_enc

    pc_mov_dec = sum(param.numel() for param in movement_dec.parameters())
    print(movement_dec)
    print("Total parameters of posterior net: {}".format(pc_mov_dec))
    all_params += pc_mov_dec

    trainer = DecompTrainerV3(opt, movement_enc, movement_dec)

    print(
        f"mean = {mean}\n"
        f"std = {std}\n"
        f"train_split_file = {train_split_file}\n"
        f"val_split_file = {val_split_file}\n"
        f"Total parameters of model: {all_params}\n"
    )

    train_dataset = MotionDatasetV2(opt, mean, std, train_split_file)
    val_dataset = MotionDatasetV2(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              drop_last=True,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=opt.batch_size,
                            drop_last=True,
                            num_workers=4,
                            shuffle=True,
                            pin_memory=True)

    trainer.train(train_loader, val_loader, plot_t2m)

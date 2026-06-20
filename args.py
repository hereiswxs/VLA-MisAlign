
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Madv_VLA arguments (Notation for the arguments followed from paper)
    parser.add_argument('--model_family', type=str, default="openvla", 
                        help="VLA_x, libero")
    
    parser.add_argument('--dataset', type=str, default="libero", 
                        help="VLA_x, libero")
    parser.add_argument('--epoch', type=int, default=15)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--learn_rate', type=float, default=1e-4,
                        help="0.0001")

    parser.add_argument('--VLA_path', type=str, default="/home/student/DongXiaorong/openvla-main/openvla-7b-finetuned-libero-object",
                        help="VLAModel")
    parser.add_argument('--VLA_task_suite', type=str, default="libero_object",
                        help=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument('--log_path', type=str, default="./log_ours")

    parser.add_argument('--pre_train', type=bool, default=False,
                        help=["BLIPModel","Stable_diffusionModel"])
    parser.add_argument('--center_crop', type=bool, default=True,
                        help=["BLIPModel","Stable_diffusionModel"])

    parser.add_argument('--load_in_8bit', type=bool, default=False)
    parser.add_argument('--load_in_4bit', type=bool, default=False)
    
    parser.add_argument('--pre_train_weight', type=str, default="/home/student/DongXiaorong/Madv_VLA/log_ours/libero/libero_goal/EVAL-libero_goal-2025_09_08-00_51_10/codes/libero_goal_perturbation_generator.pth",
                        help="single image, single txt ")

    parser.add_argument('--batch_size', type=int, default=16,
                        help="number of users: K")
    parser.add_argument('--device', type=str, default="cuda:0", #选择设备
                        help=["cuda:0","cuda:1","cpu"])
    parser.add_argument('--bert_path', type=str, default="/home/student/DongXiaorong/MFAL/bert",
                        help="BertModel")

    args = parser.parse_args()
    return args
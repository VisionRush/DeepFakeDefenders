import torch
from core.dsproc_mclsmfolder import MultiClassificationProcessor_mfolder
from toolkit.yacs import CfgNode as CN
from toolkit.dtransform import create_transforms_inference, create_transforms_inference1,\
                    create_transforms_inference2,\
                    create_transforms_inference3,\
                    create_transforms_inference4,\
                    create_transforms_inference5
from toolkit.chelper import load_model
from tqdm import tqdm
import os
import csv



def extract_model_from_pth(params_path, net_model):
    checkpoint = torch.load(params_path)
    state_dict = checkpoint['state_dict']

    net_model.load_state_dict(state_dict, strict=True)

    return net_model


def main(model_path, dataset_dir):
    cfg = CN(new_allowed=True)
    cfg.test = CN(new_allowed=True)
    cfg.test.worker_num = 8
    cfg.test.ctg_list = './dataset/label.txt'

    # : network params
    cfg.network = CN(new_allowed=True)
    cfg.network.name = 'all'
    cfg.network.weights_path = model_path
    cfg.network.class_num = 2
    cfg.test.batch_size = 16

    # create transforms
    transform =[create_transforms_inference(h=512, w=512),
                create_transforms_inference1(h=512, w=512),
                create_transforms_inference2(h=512, w=512),
                create_transforms_inference3(h=512, w=512),
                create_transforms_inference4(h=512, w=512),
                create_transforms_inference5(h=512, w=512)]
    ctg_list = cfg.test.ctg_list

    print("load model ...")

    model = load_model(cfg.network.name, cfg.network.class_num)
    model_path = cfg.network.weights_path
    model = extract_model_from_pth(model_path, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    print("load model ok...")

    csv_output_folder = './'

    full_path = os.path.join(csv_output_folder, 'final_output.csv')

    with open(full_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['img_name', 'y_pred'])

        testset = MultiClassificationProcessor_mfolder(transform)
        dataset = MultiClassificationProcessor_mfolder()
        dataset.load_data_from_dir_test(dataset_dir)

        with open(f'./{cfg.network.name}_temporal.txt', 'w') as file:
            for k in range(len(dataset.img_names_)):
                img_path = dataset.img_paths_[k]
                file.write(img_path + ' ' + '0' + '\n')

        testset.load_data_from_txt(f'./{cfg.network.name}_temporal.txt', ctg_list)

        # create dataloader
        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                    batch_size=cfg.test.batch_size,
                                                    shuffle=False,
                                                    num_workers=cfg.test.worker_num,
                                                    pin_memory=True,
                                                    drop_last=False)

        with torch.no_grad():

            test_loader = tqdm(test_loader, desc='valid', ascii=True)

            for imgs_idx, (imgs_tensor, imgs_label, img_path, _) in enumerate(test_loader):
                imgs_tensor = imgs_tensor.cuda()

                preds = model(imgs_tensor)

                for k in range(len(img_path)):
                    filename = os.path.basename(img_path[k])

                    csv_writer.writerow([filename, round(float(preds[k]), 10)])


if __name__ == '__main__':
    model_weight = './final_model_csv/final_model.pth'
    dataset_dir = ''
    main(model_weight, dataset_dir)

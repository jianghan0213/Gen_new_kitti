import os
import pickle
import cv2
import numpy as np
import kitti_common as kitti
from PIL import Image

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Van': 3,
    'Person_sitting': 4,
    'Truck': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': -1,
}


class CreateNewDate():
    def __init__(self, kitti_root, database_save_path):
        super(CreateNewDate, self).__init__()
        self.kitti_root = kitti_root
        self.classes = ("Car", "Cyclist", "Pedestrian")
        self.save_path_1 = database_save_path + '/imgs_1'
        self.save_path_2 = database_save_path + '/imgs_2'
        self.class_aug_nums = {"Car": 24, "Pedestrian": 12, "Cyclist": 12}

        info_path = os.path.join(self.kitti_root, "kitti_infos_train.pkl")
        db_info_path = os.path.join(self.kitti_root, "kitti_dbinfos_train.pkl")
        with open(info_path, 'rb') as f:
            self.kitti_infos = pickle.load(f)
        with open(db_info_path, 'rb') as f:
            self.db_infos = pickle.load(f)

        self.num_samples = len(self.kitti_infos)

        self.sample_nums_hashmap = dict()
        self.sample_counter = dict()
        for class_name, class_db_infos in self.db_infos.items():
            self.sample_nums_hashmap[class_name] = {}
            self.sample_counter[class_name] = {}
            for img_shape_key, class_shape_db_infos in class_db_infos.items():
                self.sample_nums_hashmap[class_name][img_shape_key] = len(class_shape_db_infos)
                self.sample_counter[class_name][img_shape_key] = 0


    def reset_sample_counter(self):
        for class_name, class_counter in self.sample_counter.items():
            for img_shape_key, class_shape_counter in class_counter.items():
                self.sample_counter[class_name][img_shape_key] = 0

    def update_sample_counter(self, aug_class, img_shape_key):
        total_nums = self.sample_nums_hashmap[aug_class][img_shape_key]
        current_nums = self.sample_counter[aug_class][img_shape_key]
        if current_nums < total_nums - 1:
            self.sample_counter[aug_class][img_shape_key] = current_nums + 1
        else:
            self.sample_counter[aug_class][img_shape_key] = 0


    def __call__(self):
        for i in range(self.num_samples):
            for j in range(self.num_samples):
                if i == j:
                    continue

                info, annos, img = self.get_info(j)
                info_1, annos_1, img_1 = self.get_info(i)

                if (img.shape[0] != img_1.shape[0]) or (img.shape[1] != img_1.shape[1]):
                    continue

                embedding_annos = []
                init_bboxes = []


                # image_1 related info
                names_1 = annos_1['name']
                bboxes_1 = annos_1["bbox"]
                alphas_1 = annos_1["alpha"]
                dimensions_1 = annos_1["dimensions"]
                locations_1 = annos_1["location"]
                rotys_1 = annos_1["rotation_y"]
                difficulty_1 = annos_1["difficulty"]
                truncated_1 = annos_1["truncated"]
                occluded_1 = annos_1["occluded"]
                scores_1 = annos_1["score"]

                # image_2 related info
                names = annos["name"]
                alphas = annos["alpha"]
                rotys = annos["rotation_y"]
                bboxes = annos["bbox"]
                locs = annos["location"]
                dims = annos["dimensions"]
                difficulty = annos["difficulty"]
                truncated = annos["truncated"]
                occluded = annos["occluded"]
                scores = annos["score"]
                gt_idxes = annos["index"]
                num_obj = np.sum(annos["index"] >= 0)



                for k in range(len(names_1)):
                    init_bboxes.append(bboxes_1[k])
                    if names_1[k] not in self.classes: continue
                    ins_anno = {
                        "name": names_1[k],
                        "label": TYPE_ID_CONVERSION[names_1[k]],
                        "bbox": bboxes_1[k],
                        "alpha": alphas_1[k],
                        "dim": dimensions_1[k],
                        "loc": locations_1[k],
                        "roty": rotys_1[k],
                        "P": info_1["calib/P2"],
                        "difficulty": difficulty_1[k],
                        "truncated": truncated_1[k],
                        "occluded": occluded_1[k],
                        "flipped": False,
                        "score": scores_1[k]
                    }
                    embedding_annos.append(ins_anno)
                init_bboxes = np.array(init_bboxes)

                for k in range(num_obj):
                    if difficulty[k] != 0:
                        continue
                    box2d = bboxes[k]
                    cropImg = img[int(box2d[1]):int(box2d[3]), int(box2d[0]):int(box2d[2]), :]
                    if len(init_bboxes.shape) > 1:
                        ious = kitti.iou(init_bboxes, box2d[np.newaxis, ...])
                        if np.max(ious) > 0.0:
                            continue
                        init_bboxes = np.vstack((init_bboxes, box2d[np.newaxis, ...]))
                    else:
                        init_bboxes = box2d[np.newaxis, ...].copy()
                    img_1[int(box2d[1]):int(box2d[3]), int(box2d[0]):int(box2d[2]), :] = cropImg
                    ins_anno = {
                        "name": names[k],
                        "label": TYPE_ID_CONVERSION[names[k]],
                        "bbox": box2d,
                        "alpha": alphas[k],
                        "dim": dims[k],
                        "loc": locs[k],
                        "roty": rotys[k],
                        "P": info["calib/P2"],
                        "difficulty": difficulty[k],
                        "truncated": truncated[k],
                        "occluded": occluded[k],
                        "flipped": False,
                        "score": scores[k]
                    }
                    embedding_annos.append(ins_anno)

                img_shape_key = f"{img_1.shape[0]}_{img_1.shape[1]}"
                for aug_class, aug_nums in self.class_aug_nums.items():
                    if img_shape_key in self.db_infos[aug_class].keys():
                        class_db_infos = self.db_infos[aug_class][img_shape_key]
                        trial_num = aug_nums + 45
                        nums = 0
                        for _ in range(trial_num):
                            if nums >= aug_nums:
                                break
                            sample_id = self.sample_counter[aug_class][img_shape_key]
                            self.update_sample_counter(aug_class, img_shape_key)
                            ins = class_db_infos[sample_id]
                            patch_img_path = os.path.join(self.kitti_root, ins["path"])
                            # print('====================================')
                            # print(patch_img_path)
                            # if use_right:
                            #     box2d = ins["bbox_r"]
                            #     P = ins["P3"]
                            #     patch_img_path = patch_img_path.replace("image_2", "image_3")
                            # else:
                            box2d = ins["bbox_l"]
                            #     P = ins["P2"]

                            if ins["difficulty"] > 0:
                                continue

                            if len(init_bboxes.shape) > 1:
                                ious = kitti.iou(init_bboxes, box2d[np.newaxis, ...])
                                if np.max(ious) > 0.0:
                                    continue
                                init_bboxes = np.vstack((init_bboxes, box2d[np.newaxis, ...]))
                            else:
                                init_bboxes = box2d[np.newaxis, ...].copy()
                            patch_img = cv2.imread(patch_img_path)
                            img_1[int(box2d[1]):int(box2d[3]), int(box2d[0]):int(box2d[2]), :] = patch_img
                            ins_anno = {
                                "name": ins["name"],
                                "label": TYPE_ID_CONVERSION[ins["name"]],
                                "bbox": box2d,
                                "alpha": ins["alpha"],
                                "dim": ins["dim"],
                                "loc": ins["loc"],
                                "roty": ins["roty"],
                                "P": info_1["calib/P2"],
                                "difficulty": ins["difficulty"],
                                "truncated": ins["truncated"],
                                "occluded": ins["occluded"],
                                "flipped": False,
                                "score": ins["score"]
                            }
                            embedding_annos.append(ins_anno)
                            nums += 1

                # process image2
                for k in bboxes:
                    img[int(k[1]):int(k[3]), int(k[0]):int(k[2]), :] = \
                        img_1[int(k[1]):int(k[3]), int(k[0]):int(k[2]), :]
                for k in init_bboxes:
                    img[int(k[1]):int(k[3]), int(k[0]):int(k[2]), :] = \
                        img_1[int(k[1]):int(k[3]), int(k[0]):int(k[2]), :]

                img_1 = Image.fromarray(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
                filename_1 = f"{i}_{j}_1.png"
                filepath_1 = os.path.join(self.save_path_1, filename_1)
                # cv2.imwrite(filepath_1, img_1)
                img_1.save(filepath_1)

                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                filename_2 = f"{i}_{j}_2.png"
                filepath_2 = os.path.join(self.save_path_2, filename_2)
                # cv2.imwrite(filepath_2, img)
                img.save(filepath_2)
                print(f'Save image pairs {i}_{j} done!')


    def get_info(self, index):
        info = self.kitti_infos[index]
        annos = info["annos"]
        img_path = os.path.join(self.kitti_root, info["img_path"])
        print(img_path)
        img = cv2.imread(img_path)
        # print(img)
        return info, annos, img


kitti_root = "../datasets/kitti"
database_save_path = "../datasets/new_kitti"
d = CreateNewDate(kitti_root, database_save_path)
d()
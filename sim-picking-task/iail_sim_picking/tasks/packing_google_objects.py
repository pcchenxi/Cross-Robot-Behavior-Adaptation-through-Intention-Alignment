"""Packing Google Objects tasks."""

import os
import numpy as np
from iail_sim_picking.tasks.task import Task
from iail_sim_picking.utils import utils
from iail_sim_picking.tasks import primitives

import pybullet as p

large_scale = [
                'dinosaur figure',
                'lion figure',
                'mario figure',
                'yoshi figure',
                'android toy',
                'unicorn toy',              
]
middle_scale = [
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'grey soccer shoe with cleats',
                'light brown boot with golden laces',
                'black boot with leopard print',
                'porcelain cup',
                'red cup',                  
]

class PackingSeenGoogleObjectsSeq(Task):
    """Packing Seen Google Objects Group base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.lang_template = "{obj}"
        self.task_completed_desc = "done packing objects."
        self.object_names = self.get_object_names()
        self.primitive = primitives.PickPlace()
        self.goal_idx = 0
        self.task_name = 'seen'
        self.obj_names = []
        self.target_obj_names = []
        self.goal_condition = 'single'
        self.goal_selection = 'rotating'
        self.pick_only = True
        self.max_steps = 1

    def _get_goal_indices(self, num_objects):
        if num_objects <= 0:
            return []

        if self.goal_condition == 'single':
            if self.goal_selection == 'rotating':
                idx = self.goal_idx % num_objects
                self.goal_idx = (self.goal_idx + 1) % num_objects
                return [idx]
            return [0]

        if self.goal_selection == 'rotating_suffix':
            idx = self.goal_idx % num_objects
            self.goal_idx = (self.goal_idx + 1) % num_objects
            return list(range(idx, num_objects))

        return list(range(num_objects))

    def get_object_names(self):
        return {
            'train': [
                'porcelain cup',
                'yoshi figure',
                'android toy',
                'toy school bus',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'grey soccer shoe with cleats',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'red and white striped towel',
            ],
            'val': [
                'porcelain cup',
                'yoshi figure',
                'android toy',
                'toy school bus',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'grey soccer shoe with cleats',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'red and white striped towel',
            ],
            'test': [
                'porcelain cup',
                'yoshi figure',
                'android toy',
                'toy school bus',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'grey soccer shoe with cleats',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'red and white striped towel',
            ],            
        }

    def reset(self, env):
        super().reset(env)
        self.object_id_to_name = {}

        object_names = self.object_names[self.mode]

        # Add container box.
        zone_size = (0.5, 0.5, 0.2)
        zone_pose = ((0.0, -0.5, 0.01), (0.0, 0.0, 1, 0))
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf): os.remove(container_urdf)

        # Add Google Scanned Objects to scene.
        object_points = {}
        object_ids = []
        object_template = 'google/object-template.urdf'
        max_obj_num = np.minimum(len(object_names), 4)
        chosen_objs, repeat_category = self.choose_objects(object_names, max_obj_num)
        object_descs = []

        for i in range(max_obj_num):
            size = [0.15, 0.2, 0.05]
            pose= self.get_random_pose(env, size)
            if chosen_objs[i] in large_scale:
                scale_factor = 1.5
            elif chosen_objs[i] in middle_scale:
                scale_factor = 0.7
            else:
                scale_factor = 0.5

            # Add object only if valid pose found.
            if pose[0] is not None:
                object_name = chosen_objs[i]
                object_name_with_underscore = object_name.replace(" ", "_")
                mesh_file = os.path.join(self.assets_root,
                                         'google',
                                         'meshes_fixed',
                                         f'{object_name_with_underscore}.obj')
                texture_file = os.path.join(self.assets_root,
                                            'google',
                                            'textures',
                                            f'{object_name_with_underscore}.png')

                try:
                    replace = {'FNAME': (mesh_file,),
                               'SCALE': [scale_factor, scale_factor, scale_factor],
                               'COLOR': (0.2, 0.2, 0.2)}
                    urdf = self.fill_template(object_template, replace)
                    box_id = env.add_object(urdf, pose)
                    if os.path.exists(urdf):
                        os.remove(urdf)
                    object_ids.append((box_id, (0, None)))
                    self.object_id_to_name[box_id] = object_name

                    texture_id = p.loadTexture(texture_file)
                    p.changeVisualShape(box_id, -1, textureUniqueId=texture_id)
                    p.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1])
                    object_points[box_id] = self.get_mesh_object_points(box_id)

                    object_descs.append(object_name)
                except Exception as e:
                    print("Failed to load Google Scanned Object in PyBullet")
                    print(object_name_with_underscore, mesh_file, texture_file)
                    print(f"Exception: {e}")

        self.obj_names = list(object_descs)
        self.set_goals(object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size)

        for i in range(600):
            p.stepSimulation()

    def choose_objects(self, object_names, k):
        repeat_category = None
        return np.random.choice(object_names, k, replace=False), repeat_category

    def set_goals(self, object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size):
        goal_indices = self._get_goal_indices(len(object_ids))
        self.target_obj_names = [object_descs[idx] for idx in goal_indices]

        for obj_idx in goal_indices:
            object_id, _ = object_ids[obj_idx]
            if self.pick_only:
                self.goals.append(([(object_id, (0, None))], np.int32([[1]]), [zone_pose],
                                   False, True, 'pickup', None, 1))
            else:
                chosen_obj_pts = {object_id: object_points[object_id]}
                self.goals.append(([(object_id, (0, None))], np.int32([[1]]), [zone_pose],
                                   False, True, 'zone',
                                   (chosen_obj_pts, [(zone_pose, zone_size)]),
                                   1 / len(goal_indices)))
            self.lang_goals.append(self.lang_template.format(obj=object_descs[obj_idx]))

        self.max_steps = max(1, len(goal_indices))

class PackingUnseenGoogleObjectsSeq(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()
        self.task_name = 'unseen'

    def get_object_names(self):
        return {
            'train': [
                'red cup',
                'lion figure',
                'mario figure',
                'android toy',
                'black shoe with green stripes',
                'light brown boot with golden laces',
                'black boot with leopard print',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'green and white striped towel',
            ],
            'val': [
                'red cup',
                'lion figure',
                'mario figure',
                'android toy',
                'black shoe with green stripes',
                'light brown boot with golden laces',
                'black boot with leopard print',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'green and white striped towel',
            ],
            'test': [
                'red cup',
                'lion figure',
                'mario figure',
                'android toy',
                'black shoe with green stripes',
                'light brown boot with golden laces',
                'black boot with leopard print',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'green and white striped towel',
            ],
        }


class PackingExtraGoogleObjectsSeq(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()
        self.task_name = 'extra'

    def get_object_names(self):
        return {
            'train': [
                'red cup',
                'mario figure',
                'yoshi figure',
                'grey soccer shoe with cleats',
                'black boot with leopard print',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'red and white striped towel',
            ],
            'val': [
                'red cup',
                'mario figure',
                'yoshi figure',
                'grey soccer shoe with cleats',
                'black boot with leopard print',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'red and white striped towel',
            ],
            'test': [
                'red cup',
                'mario figure',
                'yoshi figure',
                'grey soccer shoe with cleats',
                'black boot with leopard print',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'red and white striped towel',
            ],
        }
    

class PackingTopGoogleObjectsSeq(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()
        self.task_name = 'top'

    def get_object_names(self):
        return {
            'train': [
                'porcelain cup',
                'red cup',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'green and white striped towel',
                'red and white striped towel',
            ],
            'val': [
                'porcelain cup',
                'red cup',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'green and white striped towel',
                'red and white striped towel',
            ],
            'test': [
                'porcelain cup',
                'red cup',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'green and white striped towel',
                'red and white striped towel',
            ],
        }


class PackingSeenGoogleObjectsGroup(PackingSeenGoogleObjectsSeq):
    """Packing Seen Google Objects Group task."""

    def __init__(self):
        super().__init__()
        self.lang_template = "pack all the {obj} objects in the brown box"
        self.max_steps = 3

    def choose_objects(self, object_names, k):
        chosen_objects = np.random.choice(object_names, k, replace=True)
        repeat_category, distractor_category = np.random.choice(chosen_objects, 2, replace=False)
        num_repeats = np.random.randint(2, 3)
        chosen_objects[:num_repeats] = repeat_category
        chosen_objects[num_repeats:2*num_repeats] = distractor_category

        return chosen_objects, repeat_category

    def set_goals(self, object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size):
        num_pack_objs = object_descs.count(repeat_category)
        true_poses = []

        chosen_obj_pts = dict()
        chosen_obj_ids = []
        for obj_idx, (object_id, info) in enumerate(object_ids):
            if object_descs[obj_idx] == repeat_category:
                true_poses.append(zone_pose)
                chosen_obj_pts[object_id] = object_points[object_id]
                chosen_obj_ids.append((object_id, info))

        self.goals.append((
            chosen_obj_ids, np.eye(len(chosen_obj_ids)), true_poses, False, True, 'zone',
            (chosen_obj_pts, [(zone_pose, zone_size)]), 1))
        self.lang_goals.append(self.lang_template.format(obj=repeat_category))
        self.max_steps = num_pack_objs+1


class PackingUnseenGoogleObjectsGroup(PackingSeenGoogleObjectsGroup):
    """Packing Unseen Google Objects Group task."""

    def __init__(self):
        super().__init__()

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with orange stripes',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hard drive',
                'honey dipper',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'porcelain cup',
                'purple tape',
                'red and white flashlight',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
            ],
            'val': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
            'test': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
        }

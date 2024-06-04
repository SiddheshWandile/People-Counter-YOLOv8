import math
from typing import List, Tuple, Dict

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points: Dict[int, Tuple[int, int]] = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count: int = 0

    def update(self, objects_rect: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int, int]]:
        # Objects boxes and ids
        objects_bbs_ids: List[Tuple[int, int, int, int, int]] = []

        # Get center point of new objects
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append((x1, y1, x2, y2, id))
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append((x1, y1, x2, y2, self.id_count))
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {obj_bb_id[4]: self.center_points[obj_bb_id[4]] for obj_bb_id in objects_bbs_ids}

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points
        return objects_bbs_ids

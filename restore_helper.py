#  2021, Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
#  Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah, CVPR
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import numpy as np
import utils


def get_history(history_filename):
    if utils.check_file_existence(history_filename) is False:
        return None
    content = np.loadtxt(history_filename, dtype=str)
    return content


def add_to_history(history_filename, video_name):
    if utils.check_file_existence(history_filename) is True:
        content = np.loadtxt(history_filename, dtype=str)
        content = np.append(content, video_name)
        np.savetxt(history_filename, content, fmt="%s")
    else:
        np.savetxt(history_filename, np.array(["justforsafety", video_name]), fmt="%s")


def restore_from_history(history_filename, video_names):
    history = get_history(history_filename)
    if history is None:
        return video_names
    else:
        # It's N squared, but it s safe. if "pop" is in O(1), this could be O(n)
        for video_name_history in history:
            for i, video_name in enumerate(video_names):
                if video_name == video_name_history:
                    video_names.pop(i)
                    break
        return video_names



from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode
from real.orbbec_utils import frame_to_bgr_image
import pdb

class CameraOrbbec:
    """
    Orbbec Astra Camera Wrapper for Asynchronous Processing
    """
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes
    
    def __init__(
            self, 
            debug=False,
        ):
        super().__init__()

        if debug is False:
            # Orbbec Camera Setting
            pipeline = Pipeline()
            device = pipeline.get_device()
            device_info = device.get_device_info()
            device_pid = device_info.get_pid()
            config = Config()
            try:
                profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)
                profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                assert profile_list is not None
                depth_profile = profile_list.get_default_video_stream_profile()
                assert depth_profile is not None
                print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                        color_profile.get_height(),
                                                        color_profile.get_fps(),
                                                        color_profile.get_format()))
                print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                        depth_profile.get_height(),
                                                        depth_profile.get_fps(),
                                                        depth_profile.get_format()))
                config.enable_stream(depth_profile)
                import pdb; pdb.set_trace()
                while pipeline.wait_for_frames(100) is None: continue
                print("Orbbec Pipeline Initialization Success.")
                exit(0)
                import pdb; pdb.set_trace()
            except:
                pass


if __name__ == "__main__":

    camera = CameraOrbbec()
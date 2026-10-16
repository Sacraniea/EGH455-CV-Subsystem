# Run from EGH455-CV-Subsystem root
# .pixi/envs/default/bin/python "Target Detection/src/only_detector.py"

import cv2
import sys
from pathlib import Path

from predict import Predictor, initialise_predictor
predictor = Predictor(str(Path(__file__).resolve().parents[1] / "models" / "50_epoch_best_ncnn_model"))

# change path just for depthai import because i'm lazy
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Camera Stream"))
print(sys.path)
from depthai_source import DepthAIFrameSource, frame_packet_from_message
from stream_types import get_stream_profile, FramePacket



def main():
    # get profile for stream
    profile = get_stream_profile("480p25")


    # setup objects
    camera = DepthAIFrameSource(profile)

    exit = False

    while not exit:


        # get frame from depth ai
        frame_packet = camera.read()
        raw_image = frame_packet.image

        # predict on it
        prediction = predictor.predict(raw_image)

        # annotate it
        predicted_image = predictor.annotate_image(raw_image, prediction)

        cv2.imshow("predicted_image", predicted_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit = True  
  
    cv2.destroyAllWindows()
    camera.close()

if __name__ == "__main__":
    main()


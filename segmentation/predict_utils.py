import glob
import json
import os
import numpy as np
import traceback
from .inference_pipeline_mid import InferencePipeLine


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ScanSegmentation():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self, model):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.chl_pipeline = model

        #self.model = load_model()
        #sef.device = "cuda"

        pass

    @staticmethod
    def load_input(input_dir):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """

        # iterate over files in input_dir, assuming only 1 file available
        inputs = glob.glob(f'{input_dir}/*.obj')
        print("scan to process:", inputs)
        return inputs

    @staticmethod
    def write_output(labels, instances, jaw, output_path):
        """
        Write to /output/dental-labels.json your predicted labels and instances
        Check https://grand-challenge.org/components/interfaces/outputs/
        """
        pred_output = {'id_patient': "",
                       'jaw': jaw,
                       'labels': labels,
                       'instances': instances
                       }

        # just for testing
        #with open('./test/test_local/expected_output.json', 'w') as fp:
        with open(output_path, 'w') as fp:
            json.dump(pred_output, fp, cls=NpEncoder)

        return

    @staticmethod
    def get_jaw(scan_path):
        try:
            # read jaw from filename
            _, jaw = os.path.basename(scan_path).split('.')[0].split('_')
        except:
            # read from first line in obj file
            try:
                with open(scan_path, 'r') as f:
                    jaw = f.readline()[2:-1]
                if jaw not in ["upper", "lower"]:
                    return None
            except Exception as e:
                print(str(e))
                print(traceback.format_exc())
                return None

        return jaw

    def predict(self, scan_path, jaw='upper'):
        """
        Your algorithm goes here
        """

        #print(f"loading scan : {scan_path}")
        # read input 3D scan .obj
        try:
            # you can use trimesh or other any loader we keep the same order
            pred_result = self.chl_pipeline(scan_path)
            if jaw == "lower":
                pred_result["sem"][pred_result["sem"]>0] += 20
            elif jaw=="upper":
                pass
            else:
                raise "jaw name error"
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            raise
        # preprocessing if needed
        # prep_data = preprocess_function(mesh)
        # inference data here
        # labels, instances = self.model(mesh, jaw=None)


        # just for testing : generate dummy output instances and labels
        # instances = pred_result["ins"].astype(int).tolist()
        labels = pred_result["sem"].astype(int).tolist()


        return labels

    def process(self, input_mesh, jaw='upper'):
        labels = self.predict(input_mesh, jaw)
        return labels
        # self.write_output(labels=labels, instances=instances, jaw=jaw, output_path=output_path)
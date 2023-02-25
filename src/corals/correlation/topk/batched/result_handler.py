import pathlib
import tempfile
import numpy as np
import pickle

from corals.correlation.topk.batched.base import ResultHandler, joint_array_result_dtype


# class NoopResultHandler(ResultHandler):
#     """
#       Calling `load` and `save` of the NoopResultHandler (even though it does nothing)
#       can slow down result processing significantly (particularly during iterative reduce implementations).
#       so we manually check whether the result handler is None 
#       and just don't process results in that case.
#       Thanks Python -.-
#       We leave this class and comment here for future reference.
#     """
#     def init(self):
#         pass
# 
#     def save(self, result, offset):
#         return result
#   
#     def load(self, result_pointer):
#         return result_pointer
#   
#     def remove(self, result_pointer):
#         pass
# 
#     def close(self):
#         pass


class AbstractFileResultHandler(ResultHandler):
  
    def __init__(self, output_folder=None, overwrite=False, cleanup=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.overwrite = overwrite
        self.cleanup = cleanup

    def init(self):
        if self.output_folder is None:
            self.output_folder_ = pathlib.Path(tempfile.mkdtemp())
        else:
            self.output_folder_ = pathlib.Path(self.output_folder)
            self.output_folder_.mkdir(exist_ok=self.overwrite, parents=True)

    def remove(self, result_pointer):
        if self.cleanup:
            result_pointer.unlink()

    def close(self):
        if self.cleanup:
            self.output_folder_.rmdir()


class PickleResultHandler(AbstractFileResultHandler):
    
    def __init__(self, output_folder=None, overwrite=False, cleanup=True) -> None:
        super().__init__(
            output_folder=output_folder,
            overwrite=overwrite,
            cleanup=cleanup
        )

    def save(self, result, offset):
        result_file = self.output_folder_ /  f"batch_{offset}.pickle"
        pickle.dump(result, open(result_file, "wb"))
        return result_file
    
    def load(self, result_pointer):
        return pickle.load(open(result_pointer, "rb"))


class JointMemmapResultHandler(ResultHandler):
    
    def save(self, result, offset):

        cor, (idx_r, idx_c) = result

        result_file = self.out_folder_ /  f"batch_{offset}.memmap"

        memmap = np.memmap(
            result_file, 
            dtype=joint_array_result_dtype, 
            mode="w+", 
            offset=0, 
            shape=cor.size, 
            order="C")
            
        memmap["cor"] = cor
        memmap["idx_row"] = idx_r
        memmap["idx_col"] = idx_c
        memmap.flush()

        return result_file
    
    def load(self, result_pointer):
        array = np.load(open(result_pointer, "rb"))
        return array["cor"], (array["idx_row"], array["idx_col"])
    
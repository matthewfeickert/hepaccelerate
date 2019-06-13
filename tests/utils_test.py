from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend, JaggedStruct
import uproot
import numpy as np
import unittest

class TestJaggedStruct(unittest.TestCase):
    def test_jaggedstruct(self):
        attr_names_dtypes = [("Muon_pt", "float64")]
        js = JaggedStruct([0,2,3], {"pt": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])}, "Muon_", np, attr_names_dtypes)
        js.attr_names_dtypes = attr_names_dtypes
        js.save("cache")
    
        js2 = JaggedStruct.load("cache", "Muon_", attr_names_dtypes, np)
    
        np.all(js.offsets == js2.offsets)
        for k in js.attrs_data.keys():
            np.all(getattr(js, k) == getattr(js2, k))


class TestDataset(unittest.TestCase):
    NUMPY_LIB, ha = choose_backend(use_cuda=False)
    
    @staticmethod
    def load_dataset():
        fi = uproot.open("data/HZZ.root")
        #print(fi.keys())
        #print(fi.get("events").keys())
        
        datastructures = {
                "Muon": [
                    ("Muon_Px", "float32"),
                    ("Muon_Py", "float32"),
                    ("Muon_Pz", "float32"), 
                    ("Muon_E", "float32"),
                    ("Muon_Charge", "int32"),
                    ("Muon_Iso", "float32")
                ],
                "Jet": [
                    ("Jet_Px", "float32"),
                    ("Jet_Py", "float32"),
                    ("Jet_Pz", "float32"),
                    ("Jet_E", "float32"),
                    ("Jet_btag", "float32"),
                    ("Jet_ID", "bool")
                ],
                "EventVariables": [
                    ("NPrimaryVertices", "int32"),
                    ("triggerIsoMu24", "bool"),
                    ("EventWeight", "float32")
                ]
            }
        dataset = Dataset("HZZ", ["data/HZZ.root"], datastructures, cache_location="./mycache/", treename="events", datapath="")
        assert(dataset.filenames[0] == "data/HZZ.root")
        assert(len(dataset.structs["Jet"]) == 0)
        assert(len(dataset.eventvars) == 0)
        return dataset

    def setUp(self):
        self.dataset = self.load_dataset()

    def test_dataset_to_cache(self):
        dataset = self.dataset
    
        dataset.load_root()
        assert(len(dataset.data_host) == 1)
        
        assert(len(dataset.structs["Jet"]) == 1)
        assert(len(dataset.eventvars) == 1)
    
        dataset.to_cache()
        return dataset
    
    def test_dataset_from_cache(self):
        dataset = self.dataset
        dataset.load_root()
        dataset.to_cache()
        del dataset
        dataset = self.load_dataset()
        dataset.from_cache()
        
        dataset2 = self.load_dataset()
        dataset2.load_root()
    
        assert(dataset.num_objects_loaded("Jet") == dataset2.num_objects_loaded("Jet"))
        assert(dataset.num_events_loaded("Jet") == dataset2.num_events_loaded("Jet"))
   
    @staticmethod
    def map_func(dataset, ifile):
        mu = dataset.structs["Muon"][ifile]
        mu_pt = np.sqrt(mu.Px**2 + mu.Py**2)
        mu_pt_pass = mu_pt > 20
        mask_rows = np.ones(mu.numevents(), dtype=np.bool)
        mask_content = np.ones(mu.numobjects(), dtype=np.bool)
        ret = TestDataset.ha.sum_in_offsets(mu, mu_pt_pass, mask_rows, mask_content, dtype=np.int8) 
        return ret
    
    def test_dataset_map(self):
        dataset = self.load_dataset()
        dataset.load_root()
    
        rets = dataset.map(self.map_func)
        assert(len(rets) == 1)
        assert(len(rets[0]) == dataset.structs["Muon"][0].numevents())
        assert(np.sum(rets[0]) > 0)
        return rets
    
    def test_dataset_compact(self):
        dataset = self.dataset
        dataset.load_root()
    
        memsize1 = dataset.memsize()
        rets = dataset.map(self.map_func)
        dataset.compact(rets)
        memsize2 = dataset.memsize()
        assert(memsize1 > memsize2)
        print("compacted memory size ratio:", memsize2/memsize1)

if __name__ == "__main__":
    unittest.main()
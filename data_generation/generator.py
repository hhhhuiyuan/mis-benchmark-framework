from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path
import os
import shutil
import pickle
from solvers.gurobi import Gurobi
from solvers.kamis import KaMIS
import json
import numpy as np
from logzero import logger

class DataGenerator(ABC):
    def _call_gurobi_solver(self, G, timeout=30, weighted=False, pid=None, subopt_flag=False):
        if self.output_path is None:
            raise ValueError("This function can only be called if an output path is set!")
        # create temp directories
        #  
        if pid is None:
            tmp_input_folder = self.output_path / "gurobi_input"
            tmp_output_folder = self.output_path / "gurobi_output"
        else:
            tmp_input_folder = self.output_path / f"gurobi_input_{pid}"
            tmp_output_folder = self.output_path / f"gurobi_output_{pid}"

        if tmp_input_folder.exists() and tmp_input_folder.is_dir():
            shutil.rmtree(tmp_input_folder)

        if tmp_output_folder.exists() and tmp_output_folder.is_dir():
            shutil.rmtree(tmp_output_folder)
        
        
        os.mkdir(tmp_input_folder)
        os.mkdir(tmp_output_folder)

        # write input file for gurobi
        input_file = tmp_input_folder / "input.gpickle"
        with open(input_file, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        
        # call gurobi
        solver = Gurobi()
        parameters = { "time_limit": timeout, "loglevel": "INFO", "num_threads": 16,
                      "SolutionNumber": 3, "SubOptimal": subopt_flag }

        if weighted:
            parameters["weighted"] = "yes"
     
        solver.solve(tmp_input_folder, tmp_output_folder, parameters)

        # read output from gurobi
        with open(tmp_output_folder / "results.json") as f:
            results = json.load(f)

        mis = results["input"]["mwis"]
        status = results["input"]["gurobi_status"]
        obj = results["input"]["mwis_weight"] 
 
        # cleanup
        shutil.rmtree(tmp_input_folder)
        shutil.rmtree(tmp_output_folder)
        
        return mis, status, obj

    def _call_kamis_solver(self, G, timeout=30, weighted=False, pid=None):
        if self.output_path is None:
            raise ValueError("This function can only be called if an output path is set!")

        # create temp directories
        tmp_input_folder = self.output_path / "kamis_input"
        tmp_output_folder = self.output_path / "kamis_output"

        if tmp_input_folder.exists() and tmp_input_folder.is_dir():
            shutil.rmtree(tmp_input_folder)

        if tmp_output_folder.exists() and tmp_output_folder.is_dir():
            shutil.rmtree(tmp_output_folder)

        os.mkdir(tmp_input_folder)
        os.mkdir(tmp_output_folder)

        # write input file for gurobi
        input_file = tmp_input_folder / "input.gpickle"
        with open(input_file, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

        solver = KaMIS()
        parameters = { "time_limit": timeout, "loglevel": "INFO", "num_threads": 16 }

        if weighted:
            parameters["weighted"] = "yes"

        solver.solve(tmp_input_folder, tmp_output_folder, parameters)

        # read output from gurobi
        with open(tmp_output_folder / "results.json") as f:
            results = json.load(f)

        if weighted:
            mis = results["input"]["mwis"]
            status = results["input"]["mwis_found"]
            obj = results["input"]["mwis_weight"]
        else:
            mis = results["input"]["mis"]
            status = results["input"]["found_mis"]
            obj = results["input"]["vertices"]  
 
        # cleanup
        shutil.rmtree(tmp_input_folder)
        shutil.rmtree(tmp_output_folder)

        return mis, status, obj
    

    def random_weight(self, n, mu = 5, sigma = 2):
        return np.around(np.random.normal(mu, sigma, n)).astype(int).clip(min=0)

    @abstractmethod
    def generate(self, gen_labels = False, weighted = False):
        pass

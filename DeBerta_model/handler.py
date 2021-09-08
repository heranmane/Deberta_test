# Copyright (c) Facebook, Inc. and its affiliates.

"""
Instructions:
Please work through this file to construct your handler. Here are things
to watch out for:
- TODO blocks: you need to fill or modify these according to the instructions.
   The code in these blocks are for demo purpose only and they may not work.
- NOTE inline comments: remember to follow these instructions to pass the test.
For expected task I/O, please check dynalab/tasks/README.md
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

from dynalab.handler.base_handler import BaseDynaHandler, ROOTPATH
from dynalab.tasks.hs import TaskIO


# NOTE: use the following line to import modules from your repo
sys.path.append(ROOTPATH)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

class Handler(BaseDynaHandler):
    def initialize(self, context):
        """
        load model and extra files
        """
        model_pt_path, model_file_dir, device = self._handler_initialize(context)
        self.taskIO = TaskIO()

        # ############TODO 1: Initialize model ############
        """
        Load model and read relevant files here.
        """
        config = AutoConfig.from_pretrained('.')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            '.', config=config
        )
        self.tokenizer = AutoTokenizer.from_pretrained('.')
        self.model.to(device)
        self.model.eval()
        # #################################################

        self.initialized = True

    def preprocess(self, data):
        """
        preprocess data into a format that the model can do inference on
        """
        example = self._read_data(data)
        statement = example["statement"]
        input_data = self.tokenizer(statement, return_tensors="pt")
        # #################################################

        return input_data

    def inference(self, input_data):
        """
        do inference on the processed example
        """
        """
        Run model prediction using the processed data
        """
        with torch.no_grad():
            inference_output = self.model(**input_data)
            inference_output = F.softmax(inference_output[0]).squeeze()
        # #################################################

        return inference_output

    def postprocess(self, inference_output, data):
        """
        post process inference output into a response.
        response should be a single element list of a json
        the response format will need to pass the validation in
        ```
        dynalab.tasks.hs.TaskIO().verify_response(response, data)
        ```
        """
        response = dict()
        example = self._read_data(data)
        # ############TODO 4: postprocess response ########
        """
        Add attributes to response
        """
        response["id"] = example["uid"]
        response["label"] = {0: "not-hateful", 1: "hateful"}[int(inference_output.argmax())]
        response["prob"] = {"not-hateful": float(inference_output[0]), "hateful": float(inference_output[1])}
        # #################################################
        response = self.taskIO.sign_response(response, example)
        return [response]


_service = Handler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None

    # ############TODO 5: assemble inference pipeline #####
    """
    Normally you don't need to change anything in this block.
    However, if you do need to change this part (e.g. function name, argument, etc.),
    remember to make corresponding changes in the Handler class definition.
    """
    input_data = _service.preprocess(data)
    output = _service.inference(input_data)
    response = _service.postprocess(output, data)
    # #####################################################

    return response

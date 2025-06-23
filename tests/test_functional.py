import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader
import io
import sys
import numpy as np
from main import *
from tests.TestUtils import TestUtils

class TestMentalHealthModelYaksha(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_obj = TestUtils()
        try:
            cls.X_train, cls.y_train, cls.X_test, cls.y_test, cls.scaler = load_data_from_csv("mental_health_data.csv")
            cls.df = pd.read_csv("mental_health_data.csv")
            cls.dataset = MentalHealthDataset(cls.X_train, cls.y_train)
            cls.dataloader = DataLoader(cls.dataset, batch_size=8, shuffle=False)
            cls.train_loader = DataLoader(MentalHealthDataset(cls.X_train, cls.y_train), batch_size=8, shuffle=True)
            cls.test_loader = DataLoader(MentalHealthDataset(cls.X_test, cls.y_test), batch_size=8)
            cls.input_size = cls.X_train.shape[1]
            cls.model = build_model(input_size=cls.input_size, num_classes=3)
        except Exception as e:
            # Handle cases where functions return None or raise exceptions
            print(f"Setup failed due to unimplemented functions: {e}")
            cls.X_train = torch.randn(80, 4, dtype=torch.float32)
            cls.y_train = torch.randint(0, 3, (80,), dtype=torch.long)
            cls.X_test = torch.randn(20, 4, dtype=torch.float32)
            cls.y_test = torch.randint(0, 3, (20,), dtype=torch.long)
            cls.scaler = None
            cls.df = pd.read_csv("mental_health_data.csv")
            cls.dataset = None
            cls.dataloader = None
            cls.train_loader = None
            cls.test_loader = None
            cls.input_size = 4
            cls.model = None

    def test_dataset_length(self):
        try:
            result = len(self.dataset) == len(self.X_train)
            self.test_obj.yakshaAssert("TestDatasetLength", result, "functional")
            print("TestDatasetLength =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestDatasetLength", False, "functional")
            print("TestDatasetLength = Failed | Exception:", e)

    def test_model_output_shape(self):
        try:
            sample_input = self.X_train[0:1]
            output = self.model(sample_input)
            result = output.shape == torch.Size([1, 3])
            self.test_obj.yakshaAssert("TestModelOutputShape", result, "functional")
            print("TestModelOutputShape =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelOutputShape", False, "functional")
            print("TestModelOutputShape = Failed | Exception:", e)


    def test_bin_wellbeing_classification_matches_expected(self):
        try:
            actual_classes = bin_wellbeing(self.df['productivity_score'], self.df['stress_level'])
            scores = self.df['productivity_score'] - (self.df['stress_level'] * 5)
            expected_classes = pd.cut(scores, bins=[-float('inf'), 40, 70, float('inf')], labels=[0, 1, 2]).astype(int)
            result = actual_classes.equals(expected_classes)
            self.test_obj.yakshaAssert("TestBinWellbeingFromActualData", result, "functional")
            print("TestBinWellbeingFromActualData =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestBinWellbeingFromActualData", False, "functional")
            print("TestBinWellbeingFromActualData = Failed | Exception:", e)

    def test_model_accuracy_above_80_percent(self):
        try:
            model = build_model(input_size=self.input_size)
            train_model(model, self.train_loader, val_loader=None, epochs=15)
            accuracy = evaluate_model(model, self.test_loader)
            result = accuracy >= 0.80
            self.test_obj.yakshaAssert("TestModelAccuracyAbove80", result, "functional")
            print("TestModelAccuracyAbove80 =", "Passed" if result else f"Failed (Accuracy: {accuracy:.2%})")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelAccuracyAbove80", False, "functional")
            print("TestModelAccuracyAbove80 = Failed | Exception:", e)

    def test_dataset_getitem_valid(self):
        try:
            x, y = self.dataset[0]
            correct_shape = isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            correct_dims = x.shape[0] == self.X_train.shape[1]
            result = correct_shape and correct_dims
            self.test_obj.yakshaAssert("TestDatasetGetItemValid", result, "functional")
            print("TestDatasetGetItemValid =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestDatasetGetItemValid", False, "functional")
            print("TestDatasetGetItemValid = Failed | Exception:", e)

    def test_prediction_is_class_2_from_main_input(self):
        try:
            # This assumes 'new_user_input.txt' contains a record that should predict class 2
            predicted = predict_new_user(
                model_path="mental_model_class.pth",
                scaler=self.scaler,
                input_size=self.input_size,
                num_classes=3,
                file_path='new_user_input.txt'
            )
            result = predicted == 2
            self.test_obj.yakshaAssert("TestPredictionIsClass2FromMainInput", result, "functional")
            print("TestPredictionIsClass2FromMainInput =", "Passed" if result else f"Failed (Predicted: {predicted})")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictionIsClass2FromMainInput", False, "functional")
            print("TestPredictionIsClass2FromMainInput = Failed | Exception:", e)

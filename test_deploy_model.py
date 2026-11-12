
import unittest
from unittest.mock import MagicMock, patch

from src.pipeline.deploy_model import deploy_model


class TestControlledDeployment(unittest.TestCase):

    @patch("src.pipeline.deploy_model.mlflow.set_tracking_uri")
    @patch("src.pipeline.deploy_model.MlflowClient")
    @patch("src.pipeline.deploy_model.promote_model")
    @patch("src.pipeline.deploy_model.reload_api_model")
    @patch("src.pipeline.deploy_model.verify_deployment")
    def test_successful_deployment(
        self,
        mock_verify,
        mock_reload,
        mock_promote,
        mock_client_class,
        mock_tracking_uri
    ):
        client = MagicMock()
        mock_client_class.return_value = client

        current = MagicMock()
        current.version = "1"
        client.get_model_version_by_alias.return_value = current

        mock_verify.return_value = True

        result = deploy_model("5", approved=True)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["previous_version"], "1")
        self.assertEqual(result["deployed_version"], "5")

        mock_promote.assert_called_once_with(
            "5", approved=True
        )
        mock_reload.assert_called_once()
        mock_verify.assert_called_once_with("5")

        print("\nSUCCESS TEST PASSED")

    @patch("src.pipeline.deploy_model.mlflow.set_tracking_uri")
    @patch("src.pipeline.deploy_model.MlflowClient")
    @patch("src.pipeline.deploy_model.promote_model")
    @patch("src.pipeline.deploy_model.reload_api_model")
    @patch("src.pipeline.deploy_model.verify_deployment")
    def test_failed_deployment_rolls_back(
        self,
        mock_verify,
        mock_reload,
        mock_promote,
        mock_client_class,
        mock_tracking_uri
    ):
        client = MagicMock()
        mock_client_class.return_value = client

        current = MagicMock()
        current.version = "1"
        client.get_model_version_by_alias.return_value = current

        # Candidate fails health validation.
        # Restored production version passes validation.
        mock_verify.side_effect = [False, True]

        result = deploy_model("5", approved=True)

        self.assertEqual(result["status"], "rolled_back")
        self.assertEqual(result["failed_version"], "5")
        self.assertEqual(result["restored_version"], "1")

        client.set_registered_model_alias.assert_called_once_with(
            "MLPilot-Demand-Forecasting",
            "production",
            "1"
        )

        self.assertEqual(mock_reload.call_count, 2)

        self.assertEqual(
            mock_verify.call_args_list[0].args,
            ("5",)
        )
        self.assertEqual(
            mock_verify.call_args_list[1].args,
            ("1",)
        )

        print("\nROLLBACK TEST PASSED")

    def test_deployment_requires_approval(self):
        with self.assertRaises(PermissionError):
            deploy_model("5", approved=False)

        print("\nAPPROVAL TEST PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
